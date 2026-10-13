import json
import os
import shutil

import pytest
import torch

import evaluate_samples
import plant_model
from conftest import SAMPLES_DIR


@pytest.fixture(scope="module")
def shipped():
    return plant_model.load_model(device=torch.device("cpu"))


def test_shipped_checkpoint_is_resnet50_with_38_labels(shipped):
    assert shipped.arch == "resnet50"
    assert len(shipped.class_names) == 38
    assert os.path.basename(shipped.checkpoint_path) == "best_model.pth"


def test_resnet_labels_are_sorted_folder_names():
    # code.ipynb: class_names = sorted(os.listdir(TRAIN_DIR))
    labels = plant_model.load_class_names("resnet50")
    assert labels == sorted(labels)
    assert len(set(labels)) == len(labels)


def test_custom_cnn_labels_kept_their_underscores():
    labels = plant_model.load_class_names("custom_cnn")
    assert len(labels) == len(set(labels)) == 45
    # every label is "<plant>___<condition>", as produced by format_label() in training
    assert all("___" in label for label in labels)
    assert "orange___haunglongbing_(citrus_greening)" in labels
    assert "corn_(maize)___common_rust_" in labels


def test_sample_images_accuracy(shipped):
    results = evaluate_samples.evaluate(shipped, SAMPLES_DIR)
    correct = sum(predicted.lower() == expected for _, expected, predicted, _ in results)
    assert len(results) == 33
    assert correct >= 31, results


def test_custom_cnn_checkpoint_is_detected(tmp_path):
    torch.save(plant_model.PlantDiseaseModel(45).state_dict(), tmp_path / "cnn.pth")
    shutil.copy(os.path.join(plant_model.MODEL_DIR, "class_names.json"), tmp_path)

    loaded = plant_model.load_model(str(tmp_path / "cnn.pth"), device=torch.device("cpu"))

    assert loaded.arch == "custom_cnn"
    assert len(loaded.class_names) == 45
    assert loaded.transform is plant_model.TRANSFORMS["custom_cnn"]


def test_label_count_mismatch_is_reported(tmp_path):
    torch.save(plant_model.PlantDiseaseModel(45).state_dict(), tmp_path / "cnn.pth")
    (tmp_path / "class_names.json").write_text(json.dumps(["only", "three", "labels"]))

    with pytest.raises(ValueError, match="has 3 labels but the checkpoint predicts 45"):
        plant_model.load_model(str(tmp_path / "cnn.pth"), device=torch.device("cpu"))


def test_unknown_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised checkpoint"):
        plant_model.detect_architecture({"fc.weight": torch.zeros(1)})


def test_find_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("PLANT_MODEL_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        plant_model.find_checkpoint(str(tmp_path))

    (tmp_path / "best_model.pth").touch()
    assert plant_model.find_checkpoint(str(tmp_path)).endswith("best_model.pth")

    # the 45-class model wins when both are present
    (tmp_path / "plant_disease_modelFinal_45Class15.pth").touch()
    assert plant_model.find_checkpoint(str(tmp_path)).endswith("plant_disease_modelFinal_45Class15.pth")

    monkeypatch.setenv("PLANT_MODEL_PATH", "/elsewhere/model.pth")
    assert plant_model.find_checkpoint(str(tmp_path)) == "/elsewhere/model.pth"


@pytest.mark.parametrize("label, expected", [
    ("Tomato___Early_blight", "Tomato - Early blight"),
    ("corn_(maize)___common_rust_", "Corn (maize) - Common rust"),
    ("pepper,_bell___healthy", "Pepper, bell - Healthy"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn (maize) - Cercospora leaf spot Gray leaf spot"),
])
def test_format_label(label, expected):
    assert plant_model.format_label(label) == expected


def test_supported_plants():
    plants = plant_model.supported_plants(plant_model.load_class_names("custom_cnn"))
    assert len(plants) == 16
    assert "Rice" in plants and "Wheat" in plants and "Cotton" not in plants
