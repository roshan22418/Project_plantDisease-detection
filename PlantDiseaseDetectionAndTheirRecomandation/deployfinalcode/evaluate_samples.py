# evaluate_samples.py - sanity-check a checkpoint on the labelled sample images
#
#   python evaluate_samples.py [--checkpoint PATH] [--images DIR]
#
# The sample images (PlantDiseaseDetection2024/test/test) are named <Label><n>.JPG,
# e.g. AppleCedarRust1.JPG; the prefix is mapped to the class name below.
import argparse
import os
import re
import sys

from PIL import Image

import plant_model

DEFAULT_IMAGES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "PlantDiseaseDetection2024", "test", "test"
)

FILENAME_LABELS = {
    "AppleCedarRust": "apple___cedar_apple_rust",
    "AppleScab": "apple___apple_scab",
    "CornCommonRust": "corn_(maize)___common_rust_",
    "PotatoEarlyBlight": "potato___early_blight",
    "PotatoHealthy": "potato___healthy",
    "TomatoEarlyBlight": "tomato___early_blight",
    "TomatoHealthy": "tomato___healthy",
    "TomatoYellowCurlVirus": "tomato___tomato_yellow_leaf_curl_virus",
}


def expected_label(filename):
    prefix = re.match(r"[A-Za-z]+", filename).group(0)
    return FILENAME_LABELS[prefix]


def evaluate(loaded, image_dir=DEFAULT_IMAGES):
    """Return a list of (filename, expected, predicted, confidence) for every image in image_dir."""
    results = []
    for filename in sorted(os.listdir(image_dir)):
        with Image.open(os.path.join(image_dir, filename)) as image:
            predicted, confidence = plant_model.predict(loaded, image, top_k=1)[0]
        results.append((filename, expected_label(filename), predicted, confidence))
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the labelled sample images")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    args = parser.parse_args()

    loaded = plant_model.load_model(args.checkpoint)
    print(f"checkpoint: {loaded.checkpoint_path}")
    print(f"architecture: {loaded.arch}, classes: {len(loaded.class_names)}")

    results = evaluate(loaded, args.images)
    correct = 0
    for filename, expected, predicted, confidence in results:
        ok = predicted.lower() == expected
        correct += ok
        print(f"{'OK ' if ok else 'BAD'} {filename:28} {predicted:45} {confidence:6.1%}")
    print(f"accuracy: {correct}/{len(results)} = {correct / len(results):.1%}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
