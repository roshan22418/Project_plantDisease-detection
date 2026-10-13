import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(PROJECT_DIR, "deployfinalcode")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "..", "PlantDiseaseDetection2024", "test", "test")

sys.path.insert(0, APP_DIR)
