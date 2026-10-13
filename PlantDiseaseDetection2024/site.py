import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import UnidentifiedImageError

# Next to this file, so the app works whatever directory it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainedFuture.keras")

# Labels in training order (image_dataset_from_directory sorts the class folders)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

#Load the model once per server process instead of on every prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr, verbose=0)[0]
    return int(np.argmax(predictions)), float(np.max(predictions)) #index and probability of the top class

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Classifier Showcase!

    Are you curious about the intersection of artificial intelligence and plant health? Dive into our project showcase, where we unveil the results of our endeavor to train an AI model for classifying plant diseases.

    Explore Our Project:

    Training Process: Delve into the methodologies and techniques we employed to train our AI model. From data collection to algorithm selection, discover the intricacies of our training journey.

    Model Architecture: Gain insights into the architecture of our AI model. Explore the neural networks and algorithms that power our disease classification system, and learn how they contribute to accurate diagnosis.

    Performance Metrics: Discover the performance metrics of our AI model. From accuracy rates to precision and recall, explore how our model performs in identifying various plant diseases compared to traditional methods.

    Case Studies: Explore real-world case studies showcasing our AI model in action. From common plant diseases to rare afflictions, witness how our model accurately identifies and diagnoses issues, aiding in timely intervention.

    Future Directions: Peek into the future of our project. Learn about potential enhancements, applications, and collaborations that could further advance the field of AI-driven plant disease classification.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on [this github repo](https://github.com/spMohanty/PlantVillage-Dataset).
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if(st.button("Show Image")):
        if test_image is None:
            st.warning("Please upload an image first.")
        else:
            st.image(test_image, width="stretch")
    #Predict button
    if(st.button("Predict")):
        if test_image is None:
            st.warning("Please upload an image first.")
        else:
            try:
                result_index, confidence = model_prediction(test_image)
            except (UnidentifiedImageError, OSError):
                st.error("Could not read this file as an image. Please upload a JPG or PNG photo of a leaf.")
            else:
                st.snow()
                st.write("Our Prediction")
                st.success("Model is Predicting it's a {} ({:.1%} confidence)".format(CLASS_NAMES[result_index], confidence))
