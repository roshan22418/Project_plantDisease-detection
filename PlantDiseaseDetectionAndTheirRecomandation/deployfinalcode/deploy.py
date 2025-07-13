# app.py - Enhanced Streamlit web application for plant disease detection with LLM integration
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import requests
import io 
import os
import hashlib
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


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

# Load the model and class names
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlantDiseaseModel()  # Replace with your actual model class
    model.load_state_dict(torch.load("model_output/plant_disease_modelFinal_45Class15.pth", map_location=device))
    model.eval()
    return model

@st.cache_data
def load_class_names():
    with open('model_output/class_names.json', 'r') as f:
        class_names = json.load(f)
    return class_names

# LLM Options Configuration
LLM_CONFIG = {
    "gpt4o_mini": {
        "name": "GPT-4o-mini",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-4o-mini",
        "temperature": 0.7,
        "needs_api_key": True
    },
    "gpt35_turbo": {
        "name": "GPT-3.5 Turbo",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-3.5-turbo",
        "temperature": 0.7,
        "needs_api_key": True
    },
    "llama2": {
        "name": "Llama-2-7b",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "quantization": True,
        "max_length": 512,
        "temperature": 0.7,
        "needs_api_key": False
    },
    "phi2": {
        "name": "Phi-2 (Lightweight)",
        "model_id": "microsoft/phi-2",
        "quantization": True,
        "max_length": 512,
        "temperature": 0.7,
        "needs_api_key": False
    },
    "offline": {
        "name": "Offline Mode (No LLM)",
        "needs_api_key": False
    }
}

# Load LLM based on configuration settings
@st.cache_resource
def load_llm(llm_option="gpt4o_mini"):
    if llm_option not in LLM_CONFIG:
        llm_option = "gpt4o_mini"
        
    config = LLM_CONFIG[llm_option]
    
    # If using an API-based LLM
    if "api_url" in config:
        if config.get("needs_api_key", False) and not os.getenv("OPENAI_API_KEY"):
            st.warning(f"{config['name']} requires an API key. Please enter it in the Settings page.")
            return None
        return config
        
    # If using offline mode
    if llm_option == "offline":
        return None
        
    # Using HuggingFace models
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], 
            load_in_8bit=config.get("load_in_8bit", False),
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_length=config.get("max_length", 512), 
            temperature=config.get("temperature", 0.7)
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

# Create a hash for caching responses
def create_response_hash(disease_name, llm_option):
    return hashlib.md5(f"{disease_name}_{llm_option}".encode()).hexdigest()

# Cache for storing generated advice
class AdviceCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        return self.cache.get(key, None)
        
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simplistic approach)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
        
# Initialize cache
advice_cache = AdviceCache()

# Image transformation
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

# Create a prompt for the LLM based on disease name
def create_disease_prompt(disease_name):
    return f"""
    You are an agricultural expert specialized in plant diseases. Provide detailed advice about the following:
    
    Plant Disease: {disease_name}
    
    Please include:
    1. Brief description of the disease
    2. Causes and conditions that favor this disease
    3. Preventive measures farmers should take
    4. Treatment options including organic and chemical solutions
    5. Environmental considerations
    6. When to apply treatments for best results
    

    If the disease_name is Unknown , then just return response of "Unknown Plant Disease". 
    If the plant disease is healthy , then recommend preventive measures.
    Format your response in clear sections with practical advice that farmers can implement.
    Use markdown formatting for headers and bullet points.
    """

# Get treatment advice from LLM
def get_advice(disease_name, llm_option="gpt4o_mini"):
    # Check cache first
    cache_key = create_response_hash(disease_name, llm_option)
    cached_response = advice_cache.get(cache_key)
    if cached_response:
        return cached_response
    
    # Format disease name for better readability
    formatted_disease = disease_name.replace('___', ' - ').replace('_', ' ')
    
    # Create prompt template for LLM
    prompt = create_disease_prompt(formatted_disease)
    
    try:
        # Check which LLM option is selected
        if "api_url" in LLM_CONFIG.get(llm_option, {}):
            response = get_api_advice(prompt, LLM_CONFIG[llm_option])
        elif llm_option == "offline":
            response = get_fallback_advice(disease_name)
        else:
            llm = load_llm(llm_option)
            if llm is None:
                return get_fallback_advice(disease_name)
                
            generated_text = llm(prompt)[0]['generated_text']
            # Clean up the response to get only the generated part
            response = generated_text.split(prompt)[-1].strip()
        
        # Cache the response
        advice_cache.set(cache_key, response)
        return response
        
    except Exception as e:
        st.error(f"Error generating advice: {e}")
        # Fall back to dictionary-based advice
        return get_fallback_advice(disease_name)

# Get advice using API-based LLM (like OpenAI)
def get_api_advice(prompt, config):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": "You are an agricultural expert that specializes in plant diseases and their treatments."},
            {"role": "user", "content": prompt}
        ],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            config["api_url"],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")

# Alternative advice function using a dictionary for demo purposes (if LLM is unavailable)
@st.cache_data
def get_fallback_advice(disease_name):
    advice_dict = {
        "Apple___Apple_scab": """
        # Apple Scab Treatment
        
        ## Description
        Apple scab is a fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scaly lesions on leaves and fruit.
        
        ## Causes
        - Fungal spores that overwinter in fallen leaves
        - Wet, cool spring weather promotes infection
        - Prolonged leaf wetness (6+ hours) at 55-75°F
        
        ## Prevention
        1. Plant resistant apple varieties
        2. Ensure proper spacing between trees for air circulation
        3. Prune trees regularly to improve airflow
        4. Remove and destroy fallen leaves in autumn
        5. Apply dormant sprays before bud break
        
        ## Treatment
        - **Organic options**: Sulfur sprays, neem oil, or copper-based fungicides applied every 7-10 days during wet weather
        - **Chemical options**: Myclobutanil, captan, or propiconazole-based fungicides
        - Begin applications at green tip stage and continue until 2-3 weeks after petal fall
        
        ## Best Application Timing
        - Apply preventive sprays during bud break
        - Continue protection during primary infection periods (spring rains)
        - Monitor weather and apply before rain events when possible
        """,
        
        "Tomato___Early_blight": """
        # Tomato Early Blight Treatment
        
        ## Description
        Early blight is caused by the fungus Alternaria solani, creating characteristic bull's-eye patterned lesions on lower leaves.
        
        ## Causes
        - Fungal pathogen surviving in soil and plant debris
        - Warm, humid conditions (75-85°F)
        - Overhead irrigation that wets foliage
        - Poor air circulation
        
        ## Prevention
        1. Use crop rotation (3-4 year cycle)
        2. Use disease-free seeds and resistant varieties
        3. Space plants properly for air circulation
        4. Mulch around plants to prevent soil splash
        5. Water at the base of plants, preferably in morning
        
        ## Treatment
        - **Organic options**: Copper fungicides, Bacillus subtilis products, compost tea sprays
        - **Chemical options**: Chlorothalonil, mancozeb, or azoxystrobin products
        - Remove and destroy affected leaves immediately
        
        ## Best Application Timing
        - Begin preventive treatments early in the season
        - Apply every 7-10 days during humid weather
        - Reapply after heavy rains
        - Always treat at first sign of disease
        """,
        
        # Add more predefined advice for common diseases
        "Tomato___Late_blight": """
        # Tomato Late Blight Treatment
        
        ## Description
        Late blight is a devastating disease caused by the oomycete pathogen Phytophthora infestans. It causes water-soaked lesions that quickly turn brown or black, and can destroy plants within days.
        
        ## Causes
        - Airborne spores from infected plants
        - Cool, wet weather (60-70°F) with high humidity
        - Water on leaves for extended periods
        - The pathogen can survive in infected potato tubers
        
        ## Prevention
        1. Plant resistant varieties when available
        2. Ensure good air circulation around plants
        3. Avoid overhead irrigation
        4. Remove volunteer potato plants (can harbor disease)
        5. Use clean seeds and transplants
        
        ## Treatment
        - **Organic options**: Copper-based fungicides, applied before infection
        - **Chemical options**: Chlorothalonil, mancozeb, or products containing cymoxanil
        - Remove and destroy infected plants immediately to prevent spread
        
        ## Best Application Timing
        - Apply preventative sprays before disease appears
        - Reapply after rain events
        - Apply more frequently during cool, wet conditions
        - Begin treatment immediately upon first symptoms in the area
        """,
        
        "Grape___Black_rot": """
        # Grape Black Rot Treatment
        
        ## Description
        Black rot is a fungal disease caused by Guignardia bidwellii that affects grapes. It causes circular lesions on leaves and can rot the fruit.
        
        ## Causes
        - Fungal spores overwinter in mummified berries and lesions
        - Warm, humid weather (70-80°F)
        - Rainfall or high humidity during fruit development
        - Poor air circulation
        
        ## Prevention
        1. Prune vines properly to improve air circulation
        2. Remove mummified fruits and infected canes during dormant season
        3. Use a trellis system that promotes good airflow
        4. Clean up fallen leaves and fruit
        5. Select resistant varieties when possible
        
        ## Treatment
        - **Organic options**: Copper fungicides, sulfur sprays, potassium bicarbonate
        - **Chemical options**: Myclobutanil, tebuconazole, or mancozeb products
        - Apply at bud break and continue through fruit development
        
        ## Best Application Timing
        - Begin applications when shoots are 2-3 inches long
        - Most critical period is from immediate pre-bloom through 4-5 weeks after bloom
        - Apply before rain events when possible
        """,
        
        "default": """
        # Treatment Advice
        
        ## Description
        This appears to be a plant disease that affects plant health by interfering with normal plant functions. Specific symptoms may include discoloration, lesions, wilting, or abnormal growth patterns.
        
        ## Likely Causes
        - Fungal, bacterial, or viral pathogens
        - Environmental stress conditions
        - Insect vectors that may transmit disease
        - Poor cultural practices that weaken plant defense
        
        ## Prevention
        1. Use resistant varieties when available
        2. Ensure proper spacing for good air circulation
        3. Avoid overhead watering to keep foliage dry
        4. Practice crop rotation with unrelated plants
        5. Remove and destroy infected plant material
        6. Maintain plant vigor through proper nutrition
        
        ## Treatment Options
        - **Organic options**: Consider copper-based fungicides, neem oil, or sulfur sprays for fungal diseases
        - **Biological control**: Products containing beneficial microorganisms may help
        - **Chemical options**: Specific fungicides, bactericides, or insecticides depending on the causal agent
        - Prune and remove infected parts to reduce spread
        
        ## Environmental Considerations
        - Apply treatments in early morning or evening to reduce impact on beneficial insects
        - Follow label instructions carefully for all products
        - Consider integrated pest management (IPM) approaches
        
        ## Application Timing
        - Begin preventative treatments at first sign of disease
        - Reapply as needed, especially after rainfall
        - Coordinate applications with plant growth stages
        """
    }
    
    return advice_dict.get(disease_name, advice_dict["default"])

def generate_unrecognized_markdown(confidence, threshold=20.0):
    return f"""
    ### ❌ Unable to Recognize Disease

    The model's confidence is **{confidence:.2f}%**, which is below the recognition threshold of **{threshold:.0f}%**.

    Please try uploading a clearer image or another leaf image with more visible symptoms.

    ---
    **Tips for Better Detection:**
    - Make sure the leaf is in focus
    - Avoid glare or shadows
    - Capture more of the affected area
    - Use a uniform background if possible
    """

# Main Streamlit app
def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
    
    # Initialize session state
    if 'llm_option' not in st.session_state:
        st.session_state.llm_option = "gpt4o_mini"
    
    if 'detection_threshold' not in st.session_state:
        st.session_state.detection_threshold = 20.0
    
    st.title("🌿 AI-Powered Plant Disease Detection")
    st.markdown("""
    Upload an image of a plant leaf to identify diseases and get treatment recommendations.
    This system can identify diseases across 50+ plant species including wheat, rice, corn, apple, tomato, and more.
    """)
    
    # Sidebar for app navigation and settings
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Disease Detection", "About", "How It Works", "Settings"])
    
    # Add LLM dropdown directly in the sidebar
    st.sidebar.markdown("### LLM Selection")
    llm_options = {key: config["name"] for key, config in LLM_CONFIG.items()}
    
    selected_option = st.sidebar.selectbox(
        "Select LLM for treatment advice:",
        options=list(llm_options.keys()),
        format_func=lambda x: llm_options[x],
        index=list(llm_options.keys()).index(st.session_state.llm_option)
    )
    
    if selected_option != st.session_state.llm_option:
        st.session_state.llm_option = selected_option
        st.sidebar.success(f"Using: {llm_options[selected_option]}")
    
    # API Key input if needed
    if LLM_CONFIG[st.session_state.llm_option].get("needs_api_key", False):
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password", 
                                       help="Required for GPT models")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("✅ API key configured")
        else:
            st.sidebar.warning("⚠️ API key required for this model")
    
    if page == "Settings":
        st.subheader("Application Settings")
        
        # Advanced LLM Settings
        st.markdown("### LLM Configuration")
        st.markdown(f"**Current LLM**: {llm_options[st.session_state.llm_option]}")
        
        if LLM_CONFIG[st.session_state.llm_option].get("needs_api_key", False):
            st.info("This model requires an OpenAI API key. You can set it in the sidebar.")
            
            # Model-specific settings for API models
            if st.session_state.llm_option in ["gpt4o_mini", "gpt35_turbo"]:
                temperature = st.slider(
                    "Temperature (Creativity)",
                    min_value=0.0,
                    max_value=1.0,
                    value=LLM_CONFIG[st.session_state.llm_option].get("temperature", 0.7),
                    step=0.1,
                    help="Higher values make output more creative but potentially less accurate"
                )
                
                # Update the config
                LLM_CONFIG[st.session_state.llm_option]["temperature"] = temperature
                
        # Detection Threshold
        st.markdown("### Detection Settings")
        detection_threshold = st.slider(
            "Detection confidence threshold (%)",
            min_value=40.0,
            max_value=90.0,
            value=st.session_state.detection_threshold,
            step=5.0,
            help="Minimum confidence level required to confirm disease detection"
        )
        st.session_state.detection_threshold = detection_threshold
        
        # Cache Settings
        st.markdown("### Cache Settings")
        if st.button("Clear Advice Cache"):
            global advice_cache
            advice_cache = AdviceCache()
            st.success("Advice cache cleared successfully")
    
    elif page == "Disease Detection":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Upload Plant Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            # Display currently selected LLM
            st.info(f"Using {llm_options[st.session_state.llm_option]} for advice generation")
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Make prediction when button is clicked
                if st.button('Detect Disease'):
                    with st.spinner('Analyzing image...'):
                        # Load model and class names
                        model = load_model()
                        class_names = load_class_names()
                        
                        # Transform image and predict
                        input_tensor = transform_image(image)
                        
                        # Move input to the same device as model
                        device = next(model.parameters()).device
                        input_tensor = input_tensor.to(device)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(output[0], dim=0)
                            
                        # Get top 3 predictions
                        top_p, top_class = torch.topk(probabilities, 3)
                        
                        # Display results
                        st.success('Analysis complete!')
                        
                        # Display prediction results in the second column
                        with col2:
                            st.subheader("Detection Results")
                            
                            # Display top prediction with confidence
                            predicted_class = class_names[top_class[0]]
                            confidence = top_p[0].item() * 100
                            threshold = st.session_state.detection_threshold

                            if confidence < threshold:
                                st.warning("🔍 Low Confidence Prediction")
                                st.markdown(generate_unrecognized_markdown(confidence, threshold))
                            else:
                                st.markdown(f"### Detected Disease: **{predicted_class.replace('___', ' - ').replace('_', ' ')}**")
                                st.progress(confidence / 100)
                                st.markdown(f"Confidence: **{confidence:.2f}%**")

                                # Show alternative possibilities
                                st.markdown("### Alternative Possibilities")
                                for i in range(1, 3):
                                    alt_class = class_names[top_class[i]]
                                    alt_confidence = top_p[i].item() * 100
                                    st.markdown(f"- {alt_class.replace('___', ' - ').replace('_', ' ')}: {alt_confidence:.2f}%")

                                # Get treatment advice for the detected disease with loading indicator
                                st.subheader("Treatment Advisory")
                                with st.spinner("Generating treatment advice..."):
                                    try:
                                        # Use the selected LLM option from session state
                                        treatment_advice = get_advice(
                                            predicted_class, 
                                            llm_option=st.session_state.llm_option
                                        )
                                        st.markdown(treatment_advice)
                                        
                                        # Add option to download the advice as PDF
                                        st.download_button(
                                            label="Download Treatment Plan",
                                            data=treatment_advice,
                                            file_name=f"{predicted_class}_treatment.md",
                                            mime="text/markdown"
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating treatment advice: {e}")
                                        st.markdown(get_fallback_advice(predicted_class))

                            
        # If no file is uploaded, show some information
        if uploaded_file is None:
            with col2:
                st.info("👈 Upload an image to get started!")
                st.markdown("""
                ### Supported Plants
                This system can detect diseases in various plants including:
                - Fruit plants (Apple, Grape, Orange, Peach, Strawberry)
                - Vegetables (Potato, Tomato, Pepper, Squash, Corn)
                - Crops (Rice, Wheat, Soybean, Cotton)
                
                ### How to Get Best Results
                - Take clear photos in good lighting
                - Focus on affected leaves or parts
                - Include multiple affected areas in the image
                - Avoid shadows and glare
                """)
        
        elif page == "About":
            st.subheader("About This Project")
            st.markdown("""
            ## Plant Disease Detection with AI
            
            This application uses deep learning to detect plant diseases from images. It combines:
            
            1. **Convolutional Neural Networks (CNN)** trained on 50+ plant species and their diseases
            2. **Transfer Learning** with ResNet50 architecture for improved accuracy
            3. **Natural Language Processing (NLP)** to provide detailed treatment recommendations
            
            ### Data Source
            The model was trained on the Plant Village dataset containing thousands of images across multiple plant species and disease categories.
            
            ### LLM Integration
            This application leverages state-of-the-art language models to provide personalized treatment advice:
            
            - **Multiple LLM Options**: Choose between different language models based on your needs
            - **GPT-4o-mini**: Latest OpenAI model for high quality agricultural advice
            - **Offline Fallback**: Even without internet access, the system provides reliable advice
            
            ### Team
            Developed as part of an agricultural AI initiative to help farmers identify and treat plant diseases more effectively and sustainably.
            
            ### Future Updates
            - Mobile app for offline use
            - Integration with weather data for contextualized advice
            - Region-specific treatment recommendations
            - Disease progression prediction
            """)

    
        elif page == "How It Works":
            st.subheader("How It Works")
            
            st.markdown("""
            ### Machine Learning Pipeline
            
            This application uses a sophisticated AI pipeline to detect plant diseases:
            
            1. **Image Processing**:
            - Your uploaded image is resized and normalized
            - Data augmentation techniques improve model robustness
            
            2. **Disease Detection**:
            - A CNN-based neural network analyzes the image
            - The model was trained on 38 disease classes across different plants
            - Transfer learning improves accuracy even with limited training data
            
            3. **Advisory System**:
            - Large Language Models generate contextual treatment advice
            - Multiple LLM options available based on your needs:
                - Llama-2-7b: High quality, comprehensive advice
                - Phi-2: Faster, lightweight option
                - GPT-3.5: API-based option with excellent response quality
                - Offline mode: Uses predefined responses for key diseases
            - Recommendations include organic and chemical treatment options
            - Preventive measures and best practices are provided
            
            ### Technologies Used
            
            - **PyTorch**: Deep learning framework for model development
            - **Streamlit**: Web application framework
            - **Transformers**: NLP models for treatment recommendations
            - **Transfer Learning**: Leveraging pre-trained neural networks
            
            ### Technical Performance
            
            - Model Accuracy: ~96% on test dataset
            - Inference Time: 0.2-0.5 seconds per image
            - Support for 38+ different plant disease categories
            """)
            # Add a sample confusion matrix or performance visualization
            MODEL_DIR = 'model_output'
            try:
                # Load confusion matrix visualization
                cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
                if os.path.exists(cm_path):
                    st.subheader("Confusion Matrix")
                    st.image(cm_path)
                else:
                    st.info("Confusion matrix visualization not available.")
                    
            except Exception as e:
                st.error(f"Error loading performance metrics: {e}")
                st.warning("Performance visualizations may not be available.")

if __name__ == "__main__":
    main()
    
    