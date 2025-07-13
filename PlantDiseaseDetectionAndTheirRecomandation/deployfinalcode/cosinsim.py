import os
import io
import json
import zipfile
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

# 2. Backbone for embeddings
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.eval()

@torch.no_grad()
def get_embedding_from_pil(img: Image.Image):
    x = preprocess(img).unsqueeze(0)
    e = backbone(x).squeeze()
    return e / e.norm()

def create_archive_with_index(classes_root: str, archive_path: str = "images.bin"):
    """
    Zip all images under classes_root, plus an 'index.json' mapping
    each class to its list of paths inside the archive.
    """
    index = {}
    with zipfile.ZipFile(archive_path, "w", ZIP_DEFLATED := zipfile.ZIP_DEFLATED) as zf:
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

def avg_cosine_scores_from_archive(archive_path: str, query_img_path: str):
    """
    Load 'index.json' from the archive, then for each class compute
    and print the average cosine similarity against query_img_path.
    """
    # load query embedding
    query_img = Image.open(query_img_path).convert("RGB")
    q_embed   = get_embedding_from_pil(query_img)

    with zipfile.ZipFile(archive_path, "r") as zf:
        # load the index
        index = json.loads(zf.read("index.json").decode("utf-8"))

        for cls, rel_list in index.items():
            sims = []
            for rel in rel_list:
                try:
                    data = zf.read(rel)
                    img  = Image.open(io.BytesIO(data)).convert("RGB")
                    e    = get_embedding_from_pil(img)
                    sims.append(float(torch.dot(q_embed, e)))
                except KeyError:
                    print(f"  Missing entry in archive: {rel}")
                except Exception as ex:
                    print(f"  Error {rel}: {ex}")
            if sims:
                avg = sum(sims) / len(sims)
                print(f"Class '{cls}': avg cosine = {avg:.4f}")
            else:
                print(f"Class '{cls}': no images in archive.")

def similarity(archive_path: str, query_img):
    """
    Load 'index.json' from the archive, then for each class compute
    and print the average cosine similarity against query_img_path.
    """
    # load query embedding
    #query_img = Image.open(query_img_path).convert("RGB")
    q_embed   = get_embedding_from_pil(query_img)

    with zipfile.ZipFile(archive_path, "r") as zf:
        # load the index
        index = json.loads(zf.read("index.json").decode("utf-8"))
        avsims=[]
        for cls, rel_list in index.items():
            
            sims = []
            for rel in rel_list:
                try:
                    data = zf.read(rel)
                    img  = Image.open(io.BytesIO(data)).convert("RGB")
                    e    = get_embedding_from_pil(img)
                    sims.append(float(torch.dot(q_embed, e)))
                except KeyError:
                    print(f"  Missing entry in archive: {rel}")
                except Exception as ex:
                    print(f"  Error {rel}: {ex}")
            if sims:
                avg = sum(sims) / len(sims)
                avsims.append(avg)

    return max(avsims)



if __name__ == "__main__":
    # 1) Create archive with embedded index
    #create_archive_with_index("images", "images.bin")
    # 2) Run similarity check solely from archive
    #avg_cosine_scores_from_archive("images.bin", "test/3.jpg")
    print(similarity("images.bin", "test/mango.jpg"))
