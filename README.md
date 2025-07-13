# 🌿 Plant Disease Detection and Recommendation System

This project is an AI-based system to detect plant diseases from leaf images and provide suitable recommendations. It supports **45 different classes** across multiple crop types like **tomato, potato, rice, maize, corn**, and more. The system uses deep learning models and provides an interface via a web and Android application.

---

## 🔍 Problem Statement

Plant diseases can drastically affect crop yield and quality. Early and accurate detection is essential for timely intervention. Our system aims to automate disease recognition and assist farmers and agronomists with appropriate treatment suggestions.

---

## 🚀 Features

- 🔬 **Supports 45 Plant-Disease Classes**  
  Covers a diverse set of crops and diseases (e.g., bacterial spots, blight, mildew).

- 🧠 **Models Used**:
  - **Custom CNN**: Achieved **96% accuracy** on test data.
  - **ResNet-100**: Achieved **95% accuracy** on test data.

- 🛠 **Techniques**:
  - Image preprocessing
  - Data augmentation (flipping, rotation, zoom, etc.)
  - Hyperparameter tuning

- 🌐 **Web Application**:
  Built using **Streamlit** for interactive disease detection.

- 📱 **Mobile Application**:
  Developed using **Kotlin** for easy accessibility on Android devices.

---

## 🖼 Sample Classes

- **Tomato**: Bacterial Spot, Late Blight, Yellow Leaf Curl Virus  
- **Potato**: Early Blight, Late Blight  
- **Maize**: Common Rust, Northern Leaf Blight  
- **Rice**: Brown Spot, Leaf Smut  
- _(and many more...)_

---

## 📊 Model Performance

| Model       | Test Accuracy |
|-------------|---------------|
| Custom CNN  | 96%           |
| ResNet-100  | 95%           |

---

## 🧰 Tech Stack

- **Python**, **TensorFlow/Keras** for model training  
- **OpenCV**, **PIL** for image preprocessing  
- **Streamlit** for web interface  
- **Kotlin** & **Android Studio** for mobile application  
- **kaggle** for training environment  

---

## 👨‍💻 Team Members

- Roshan Kumar Mahto  
- Tarandeep Singh  
- Kshitij  
- Himanshu Kumar  

**Project Guide:** Prof. Saket Anand

---

## 📝 How to Run

### 🔧 Web App
```bash
streamlit run app.py
