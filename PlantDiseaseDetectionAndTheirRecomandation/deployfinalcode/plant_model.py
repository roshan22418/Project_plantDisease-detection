# plant_model.py - model architectures, checkpoint loading and preprocessing
#
# Two kinds of checkpoint come out of this project:
#   * "resnet50"   - ResNet-50 transfer-learning model from code.ipynb, 38 PlantVillage
#                    classes, 224x224 input, ImageNet normalisation (model_output/best_model.pth)
#   * "custom_cnn" - 5-block CNN from cv-projectnewupdated.ipynb, 45 classes (PlantVillage +
#                    rice + wheat), 128x128 input, mean/std 0.5 (plant_disease_modelFinal_45Class15.pth)
# The architecture is detected from the state-dict keys, so either file can be dropped into
# model_output/ and the app picks the matching network, labels and preprocessing.
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output")

# Checked in this order; the first one that exists is used
CHECKPOINT_CANDIDATES = ["plant_disease_modelFinal_45Class15.pth", "best_model.pth"]

CLASS_NAME_FILES = {
    "resnet50": "class_names_resnet50.json",
    "custom_cnn": "class_names.json",
}

TRANSFORMS = {
    # Same as test_transforms in code.ipynb
    "resnet50": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # Same as ImageDataset.default_transform in cv-projectnewupdated.ipynb
    "custom_cnn": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
}


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=45):
        super(PlantDiseaseModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.4)

        # Block 4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.4)

        # Block 5 (Extra Layers for Depth)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dropout5 = nn.Dropout(0.5)

        # Fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 1024)
        self.dropout6 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        x = self.dropout3(x)

        x = self.pool4(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))))
        x = self.dropout4(x)

        x = self.pool5(F.relu(self.bn10(self.conv10(F.relu(self.bn9(self.conv9(x)))))))
        x = self.dropout5(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout6(x)

        x = self.fc2(x)

        return x


class TransferLearningModel(nn.Module):
    """ResNet-50 with the classifier head used in code.ipynb."""

    def __init__(self, num_classes=38):
        super(TransferLearningModel, self).__init__()
        # weights=None: every weight comes from the checkpoint, so skip the ImageNet download
        self.resnet = resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


@dataclass
class LoadedModel:
    model: nn.Module
    arch: str
    class_names: list
    transform: transforms.Compose
    checkpoint_path: str


def detect_architecture(state_dict):
    if any(key.startswith("resnet.") for key in state_dict):
        return "resnet50"
    if "conv10.weight" in state_dict and "fc2.weight" in state_dict:
        return "custom_cnn"
    raise ValueError("Unrecognised checkpoint: expected ResNet-50 or custom CNN weights")


def num_classes_in(state_dict, arch):
    final_layer = "resnet.fc.3.weight" if arch == "resnet50" else "fc2.weight"
    return state_dict[final_layer].shape[0]


def find_checkpoint(model_dir=MODEL_DIR):
    """Return the checkpoint to serve: $PLANT_MODEL_PATH, else the first candidate in model_dir."""
    env_path = os.getenv("PLANT_MODEL_PATH")
    if env_path:
        return env_path
    for name in CHECKPOINT_CANDIDATES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No model checkpoint found in {model_dir}; expected one of {CHECKPOINT_CANDIDATES}"
    )


def load_class_names(arch, model_dir=MODEL_DIR):
    with open(os.path.join(model_dir, CLASS_NAME_FILES[arch]), "r") as f:
        return json.load(f)


def load_model(checkpoint_path=None, device=None):
    checkpoint_path = checkpoint_path or find_checkpoint()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    arch = detect_architecture(state_dict)
    num_classes = num_classes_in(state_dict, arch)

    class_names = load_class_names(arch, os.path.dirname(checkpoint_path))
    if len(class_names) != num_classes:
        raise ValueError(
            f"{CLASS_NAME_FILES[arch]} has {len(class_names)} labels but the checkpoint "
            f"predicts {num_classes} classes"
        )

    model = TransferLearningModel(num_classes) if arch == "resnet50" else PlantDiseaseModel(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return LoadedModel(model, arch, class_names, TRANSFORMS[arch], checkpoint_path)


@torch.no_grad()
def predict(loaded, image, top_k=3):
    """Return the top_k (class_name, probability) pairs for a PIL image."""
    device = next(loaded.model.parameters()).device
    input_tensor = loaded.transform(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = F.softmax(loaded.model(input_tensor)[0], dim=0)
    top_p, top_class = torch.topk(probabilities, min(top_k, len(loaded.class_names)))
    return [(loaded.class_names[i], p) for p, i in zip(top_p.tolist(), top_class.tolist())]


def format_label(class_name):
    """'corn_(maize)___common_rust_' -> 'Corn (maize) - Common rust'"""
    plant, _, disease = class_name.partition("___")
    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    plant = plant[:1].upper() + plant[1:]
    disease = disease[:1].upper() + disease[1:]
    return f"{plant} - {disease}" if disease else plant


def supported_plants(class_names):
    """Sorted plant names covered by the labels, e.g. ['Apple', 'Blueberry', ...]"""
    return sorted({format_label(name).split(" - ")[0] for name in class_names})
