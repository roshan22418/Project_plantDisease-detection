# 🌿 Plant Disease Detection and Recommendation System

This project is an AI-based system to detect plant diseases from leaf images and provide suitable recommendations. Deep learning models classify the leaf, and a Streamlit web app shows the result together with treatment advice from an LLM (OpenAI or a local Hugging Face model) or from built-in offline advice.

---

## 🔍 Problem Statement

Plant diseases can drastically affect crop yield and quality. Early and accurate detection is essential for timely intervention. Our system aims to automate disease recognition and assist farmers and agronomists with appropriate treatment suggestions.

---

## 🚀 Features

- 🔬 **Plant-disease classifiers**
  - **ResNet-50** (transfer learning, PyTorch): 38 PlantVillage classes across 14 crops. This is the model shipped with the web app.
  - **Custom CNN** (PyTorch): 45 classes, the 38 above plus rice and wheat diseases. Its checkpoint is not included in this repository; drop `plant_disease_modelFinal_45Class15.pth` into `model_output/` and the app uses it automatically.
  - **Keras CNN** (TensorFlow, 2024 version): 38 PlantVillage classes.

- 🛠 **Techniques**:
  - Image preprocessing
  - Data augmentation (flipping, rotation, blur, brightness/contrast, etc.)
  - Hyperparameter tuning and learning-rate scheduling (ReduceLROnPlateau)

- 🌐 **Web Application**:
  Built using **Streamlit**: top-3 predictions with confidence, a confidence threshold, and downloadable treatment advice from GPT-4o-mini / GPT-3.5 Turbo, Llama-2 / Phi-2, or offline mode.

- 📱 **Mobile Application**:
  Developed using **Kotlin** for Android (not included in this repository).

---

## 🖼 Sample Classes

- **Tomato**: Bacterial Spot, Late Blight, Yellow Leaf Curl Virus
- **Potato**: Early Blight, Late Blight
- **Corn (maize)**: Common Rust, Northern Leaf Blight
- **Rice** (45-class model): Brown Spot, Leaf Blast, Neck Blast
- **Wheat** (45-class model): Brown Rust, Yellow Rust
- _(and many more...)_

---

## 📊 Model Performance

| Model | Classes | Reported accuracy | Source | 33 sample images* |
|-------|---------|-------------------|--------|-------------------|
| ResNet-50 (shipped) | 38 | ≈97.4% on the 17,572-image validation split | diagonal of `model_output/confusion_matrix.png` | 32/33 |
| Custom CNN | 45 | 96.8% on a 12,435-image held-out split | classification report in `cv-projectnewupdated (9).ipynb` | checkpoint not in repo |
| Keras CNN | 38 | 95.2% on the 17,572-image validation split | `LoadAndTrainTheModel.ipynb` | 27/33 |

\* `PlantDiseaseDetection2024/test/test`, scored with `deployfinalcode/evaluate_samples.py` and the Keras test suite.

See [Known limitations](#-known-limitations) before reading too much into these numbers.

---

## 🧰 Tech Stack

- **Python**, **PyTorch / torchvision** for the ResNet-50 and custom CNN
- **TensorFlow / Keras** for the 2024 Keras CNN
- **OpenCV**, **PIL**, **albumentations** for image preprocessing and augmentation
- **Streamlit** for the web interfaces
- **OpenAI API** or **Hugging Face Transformers** for treatment advice
- **Kotlin** & **Android Studio** for the mobile application
- **Kaggle** for the training environment, **Weights & Biases** for experiment tracking

---

## 🗂 Repository Layout

```
PlantDiseaseDetectionAndTheirRecomandation/   PyTorch models + LLM advice app
  app.py                                      entry point (runs deployfinalcode/deploy.py)
  deployfinalcode/
    deploy.py                                 Streamlit app
    plant_model.py                            architectures, checkpoint detection, preprocessing
    evaluate_samples.py                       accuracy on the labelled sample images
    cosinsim.py                               cosine similarity to reference images (images.bin)
    model_output/                             best_model.pth (ResNet-50), label files, confusion matrix
    code.ipynb                                ResNet-50 training
    cv-projectnewupdated (8).ipynb            custom CNN training (copy)
  cv-projectnewupdated (9).ipynb              custom CNN training, EDA, augmentation
  tests/
PlantDiseaseDetection2024/                    Keras CNN + Streamlit app
  site.py, trainedFuture.keras, LoadAndTrainTheModel*.ipynb, TestDataset.ipynb
  test/test/                                  33 labelled sample images
  tests/
tests/                                        repository checks (notebook syntax, secret scan)
```

---

## 👨‍💻 Team Members

- Roshan Kumar Mahto
- Tarandeep Singh
- Kshitij
- Himanshu Kumar

**Project Guide:** Prof. Saket Anand

---

## 📝 How to Run

### 🔧 Web App (PyTorch + treatment advice)

```bash
cd PlantDiseaseDetectionAndTheirRecomandation
pip install -r requirements.txt
streamlit run app.py
```

- Advice defaults to **offline mode**. To use GPT models, enter an OpenAI key in the sidebar (kept only for your session) or set `OPENAI_API_KEY` on the server.
- For Llama-2 / Phi-2, also `pip install transformers` (Llama-2 is gated on Hugging Face and needs an access token).
- `PLANT_MODEL_PATH=/path/to/model.pth` serves a different checkpoint; the architecture and labels are detected automatically.
- Check a checkpoint on the sample images: `python deployfinalcode/evaluate_samples.py`

### 🔧 Web App (Keras, 2024)

```bash
cd PlantDiseaseDetection2024
pip install -r requirements.txt
streamlit run site.py
```

### ✅ Tests

```bash
pip install pytest
pytest tests                                       # repository checks
pytest PlantDiseaseDetectionAndTheirRecomandation/tests   # needs requirements.txt (+ transformers for one test)
pytest PlantDiseaseDetection2024/tests                    # needs requirements.txt (TensorFlow)
```

The same suites run in GitHub Actions (`.github/workflows/tests.yml`).

---

## ⚠️ Known Limitations

- **Optimistic accuracy.** The training data is the offline-augmented *New Plant Diseases Dataset*, which contains several augmented copies of each original leaf. The custom CNN's train/validation/test sets come from a `random_split` of those images (without a fixed seed), so copies of the same leaf can end up in both training and test data. The Keras notebooks evaluate their "test" set on the `valid` folder that was also used for validation during training. Real-world accuracy, especially on field photos with cluttered backgrounds, is likely lower.
- **Weak classes.** In the custom CNN's report, rice leaf blast has 0.40 recall and rice brown spot 0.64 precision.
- **Offline advice** has specific entries only for apple scab, grape black rot, tomato early/late blight and healthy plants; other diseases get general advice.
- `cosinsim.py` needs the ImageNet ResNet-50 weights (downloaded by torchvision on first use) and is not wired into the app yet.
