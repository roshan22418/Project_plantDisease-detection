import ast
import os
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(PROJECT_DIR, "site.py")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "test", "test")

FILENAME_LABELS = {
    "AppleCedarRust": "Apple___Cedar_apple_rust",
    "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust_",
    "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy",
    "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy",
    "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def class_names():
    # site.py cannot be imported by name (it would shadow the stdlib `site` module)
    tree = ast.parse(open(SITE).read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CLASS_NAMES")
    return ast.literal_eval(node.value)


def recognition_page():
    at = AppTest.from_file(SITE, default_timeout=300).run()
    at.sidebar.selectbox[0].select("Disease Recognition").run()
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_class_names_are_in_keras_folder_order():
    names = class_names()
    assert len(names) == 38
    assert names == sorted(names)


@pytest.mark.parametrize("page", ["Home", "About", "Disease Recognition"])
def test_pages_render(page):
    at = AppTest.from_file(SITE, default_timeout=300).run()
    at.sidebar.selectbox[0].select(page).run()
    assert not at.exception


@pytest.mark.parametrize("button", ["Predict", "Show Image"])
def test_buttons_without_upload_ask_for_an_image(button):
    at = recognition_page()
    click(at, button)
    assert not at.exception
    assert [w.value for w in at.warning] == ["Please upload an image first."]


def test_predict_sample_image():
    at = recognition_page()
    with open(os.path.join(SAMPLES_DIR, "AppleCedarRust1.JPG"), "rb") as f:
        at.file_uploader[0].upload("leaf.jpg", f.read(), "image/jpeg").run()
    click(at, "Predict")
    assert not at.exception
    assert "Apple___Cedar_apple_rust" in at.success[0].value


def test_unreadable_upload_shows_error():
    at = recognition_page()
    at.file_uploader[0].upload("broken.jpg", b"definitely not a jpeg", "image/jpeg").run()
    click(at, "Predict")
    assert not at.exception
    assert len(at.error) == 1


def test_sample_images_accuracy():
    model = tf.keras.models.load_model(os.path.join(PROJECT_DIR, "trainedFuture.keras"))
    names = class_names()
    correct = 0
    for filename in sorted(os.listdir(SAMPLES_DIR)):
        image = tf.keras.preprocessing.image.load_img(os.path.join(SAMPLES_DIR, filename), target_size=(128, 128))
        batch = np.array([tf.keras.preprocessing.image.img_to_array(image)])
        predicted = names[int(np.argmax(model.predict(batch, verbose=0)))]
        correct += predicted == FILENAME_LABELS[re.match(r"[A-Za-z]+", filename).group(0)]
    assert correct >= 26  # 27/33 with TensorFlow 2.21
