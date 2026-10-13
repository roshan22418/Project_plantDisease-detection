import io
import json
import os
import zipfile

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50

import cosinsim
from conftest import APP_DIR, SAMPLES_DIR

ARCHIVE = os.path.join(APP_DIR, "images.bin")


@pytest.fixture
def random_backbone(monkeypatch):
    # Deterministic weights without downloading ImageNet ones; the maths does not depend on them
    torch.manual_seed(0)
    backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-1]).eval()
    monkeypatch.setattr(cosinsim, "get_backbone", lambda: backbone)
    cosinsim.load_class_embeddings.cache_clear()
    yield
    cosinsim.load_class_embeddings.cache_clear()


def test_class_similarities_equal_average_per_image_cosine(random_backbone):
    query = Image.open(os.path.join(SAMPLES_DIR, "AppleScab1.JPG")).convert("RGB")
    scores = cosinsim.class_similarities(ARCHIVE, query)

    q = cosinsim.get_embedding_from_pil(query)
    with zipfile.ZipFile(ARCHIVE) as zf:
        index = json.loads(zf.read("index.json"))
        for cls, entries in index.items():
            if not entries:
                assert cls not in scores
                continue
            sims = [float(torch.dot(q, cosinsim.get_embedding_from_pil(
                Image.open(io.BytesIO(zf.read(rel))).convert("RGB")))) for rel in entries]
            assert scores[cls] == pytest.approx(sum(sims) / len(sims), abs=1e-5)

    assert cosinsim.similarity(ARCHIVE, query) == max(scores.values())
