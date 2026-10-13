
import os
import io
import json
import sys
import zipfile
from functools import lru_cache
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2. Backbone for embeddings, built on first use rather than at import time
@lru_cache(maxsize=1)
def get_backbone():
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()
    return backbone

@torch.no_grad()
def get_embedding_from_pil(img: Image.Image):
    x = preprocess(img).unsqueeze(0)
    e = get_backbone()(x).squeeze()
    return e / e.norm()

def create_archive_with_index(classes_root: str, archive_path: str = "images.bin"):
    """
    Zip all images under classes_root, plus an 'index.json' mapping
    each class to its list of paths inside the archive.
    """
    index = {}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls in sorted(os.listdir(classes_root)):
            folder = os.path.join(classes_root, cls)
            if not os.path.isdir(folder):
                continue
            index[cls] = []
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                rel = f"{cls}/{fname}"
                zf.write(os.path.join(folder, fname), arcname=rel)
                index[cls].append(rel)
        # write the index file
        zf.writestr("index.json", json.dumps(index))
    print(f"Archive '{archive_path}' created with index for {len(index)} classes.")

@lru_cache(maxsize=4)
def load_class_embeddings(archive_path: str):
    """
    Embed every reference image in the archive once and return
    {class: mean embedding}. The average cosine similarity of a query q
    against a class is mean_i(q . e_i) = q . mean_i(e_i), so a query only
    needs one dot product per class instead of re-embedding every image.
    Classes without readable images are left out.
    """
    class_means = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        index = json.loads(zf.read("index.json").decode("utf-8"))
        for cls, rel_list in index.items():
            embeds = []
            for rel in rel_list:
                try:
                    data = zf.read(rel)
                    img  = Image.open(io.BytesIO(data)).convert("RGB")
                    embeds.append(get_embedding_from_pil(img))
                except KeyError:
                    print(f"  Missing entry in archive: {rel}")
                except Exception as ex:
                    print(f"  Error {rel}: {ex}")
            if embeds:
                class_means[cls] = torch.stack(embeds).mean(dim=0)
    return class_means

def class_similarities(archive_path: str, query_img: Image.Image):
    """Return {class: average cosine similarity} of query_img against each class."""
    q_embed = get_embedding_from_pil(query_img)
    return {cls: float(torch.dot(q_embed, mean))
            for cls, mean in load_class_embeddings(archive_path).items()}

def avg_cosine_scores_from_archive(archive_path: str, query_img_path: str):
    """
    Print the average cosine similarity of query_img_path against each class.
    """
    query_img = Image.open(query_img_path).convert("RGB")
    for cls, avg in class_similarities(archive_path, query_img).items():
        print(f"Class '{cls}': avg cosine = {avg:.4f}")

def similarity(archive_path: str, query_img):
    """
    Return the highest per-class average cosine similarity of query_img
    (a PIL image) against the reference images in the archive.
    """
    scores = class_similarities(archive_path, query_img)
    if not scores:
        raise ValueError(f"No readable reference images in '{archive_path}'")
    return max(scores.values())



if __name__ == "__main__":
    # 1) Create archive with embedded index
    #create_archive_with_index("images", "images.bin")
    # 2) Run similarity check solely from archive
    #avg_cosine_scores_from_archive("images.bin", "test/3.jpg")
    if len(sys.argv) != 2:
        sys.exit("usage: python cosinsim.py <query image>")
    print(similarity("images.bin", Image.open(sys.argv[1]).convert("RGB")))
