# deploy.py - Streamlit web application for plant disease detection with LLM integration
import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import os
import hashlib
import threading

import plant_model
from plant_model import format_label


# Load the model; architecture, labels and preprocessing are detected from the checkpoint
@st.cache_resource
def load_model():
    return plant_model.load_model()

# LLM Options Configuration
LLM_CONFIG = {
    "gpt4o_mini": {
        "name": "GPT-4o-mini",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-4o-mini",
        "needs_api_key": True
    },
    "gpt35_turbo": {
        "name": "GPT-3.5 Turbo",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model_id": "gpt-3.5-turbo",
        "needs_api_key": True
    },
    "llama2": {
        "name": "Llama-2-7b",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "max_new_tokens": 512,
        "needs_api_key": False
    },
    "phi2": {
        "name": "Phi-2 (Lightweight)",
        "model_id": "microsoft/phi-2",
        "max_new_tokens": 512,
        "needs_api_key": False
    },
    "offline": {
        "name": "Offline Mode (No LLM)",
        "needs_api_key": False
    }
}

# Load a local Hugging Face text-generation pipeline. Errors propagate to get_advice(),
# and st.cache_resource does not cache them, so a failed load is retried next time.
@st.cache_resource
def load_llm(llm_option):
    # Imported here so the offline and OpenAI modes do not need transformers installed
    from transformers import pipeline
    return pipeline("text-generation", model=LLM_CONFIG[llm_option]["model_id"])

# Create a hash for caching responses
def create_response_hash(disease_name, llm_option):
    return hashlib.md5(f"{disease_name}_{llm_option}".encode()).hexdigest()

# Cache for storing generated advice
class AdviceCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()  # shared by all session threads
        
    def get(self, key):
        with self.lock:
            return self.cache.get(key, None)
        
    def set(self, key, value):
        with self.lock:
            if key not in self.cache and len(self.cache) >= self.max_size:
                # Remove oldest item (simplistic approach)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value

# One cache per server process. A module-level instance would be recreated
# on every rerun, because Streamlit re-executes this script each time.
@st.cache_resource
def get_advice_cache():
    return AdviceCache()

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
def get_advice(disease_name, llm_option="gpt4o_mini", api_key=None, temperature=0.7):
    # Check cache first
    cache_key = create_response_hash(disease_name, llm_option)
    cached_response = get_advice_cache().get(cache_key)
    if cached_response:
        return cached_response
    
    # Create prompt template for LLM with a readable disease name
    prompt = create_disease_prompt(format_label(disease_name))
    
    try:
        # Check which LLM option is selected
        if "api_url" in LLM_CONFIG.get(llm_option, {}):
            if not api_key:
                st.info("No OpenAI API key set, so showing offline advice instead.")
                return get_fallback_advice(disease_name)
            response = get_api_advice(prompt, LLM_CONFIG[llm_option], api_key, temperature)
        elif llm_option == "offline":
            response = get_fallback_advice(disease_name)
        else:
            generated = load_llm(llm_option)(
                prompt,
                max_new_tokens=LLM_CONFIG[llm_option]["max_new_tokens"],
                do_sample=True,
                temperature=temperature,
                return_full_text=False  # only the answer, not the echoed prompt
            )
            response = generated[0]["generated_text"].strip()
        
        # Cache the response
        get_advice_cache().set(cache_key, response)
        return response
        
    except Exception as e:
        st.error(f"Error generating advice: {e}")
        # Fall back to dictionary-based advice
        return get_fallback_advice(disease_name)

# Get advice using API-based LLM (like OpenAI)
def get_api_advice(prompt, config, api_key, temperature=0.7):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": config["model_id"],
        "messages": [
            {"role": "system", "content": "You are an agricultural expert that specializes in plant diseases and their treatments."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
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
        
        "healthy": """
        # Healthy Plant
        
        ## Assessment
        No disease was detected on this leaf. Keep monitoring, since problems are easiest to manage when caught early.
        
        ## Preventive Care
        1. Inspect plants weekly, checking both sides of the leaves
        2. Water at the base of plants, preferably in the morning, to keep foliage dry
        3. Keep enough spacing and prune to maintain good air circulation
        4. Remove fallen leaves and plant debris that can harbour pathogens
        5. Rotate crops and use disease-free seed or transplants
        6. Maintain balanced nutrition and avoid excess nitrogen
        
        ## When to Act
        - Re-check the plant if spots, discoloration, wilting or unusual growth appear
        - Isolate or remove plants with suspicious symptoms to limit spread
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
    
    # Model labels differ in case (e.g. "Tomato___Early_blight" vs "tomato___early_blight")
    lookup = {name.lower(): advice for name, advice in advice_dict.items()}
    key = disease_name.lower()
    if key not in lookup and key.endswith("___healthy"):
        key = "healthy"
    return lookup.get(key, advice_dict["default"])

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

# Render a stored detection result and its treatment advice
def show_results(result):
    st.subheader("Detection Results")

    # Display top prediction with confidence
    predicted_class, top_probability = result["predictions"][0]
    confidence = top_probability * 100
    threshold = st.session_state.detection_threshold

    if confidence < threshold:
        st.warning("🔍 Low Confidence Prediction")
        st.markdown(generate_unrecognized_markdown(confidence, threshold))
        return

    st.markdown(f"### Detected Disease: **{format_label(predicted_class)}**")
    st.progress(confidence / 100)
    st.markdown(f"Confidence: **{confidence:.2f}%**")

    # Show alternative possibilities
    st.markdown("### Alternative Possibilities")
    for alt_class, alt_probability in result["predictions"][1:]:
        st.markdown(f"- {format_label(alt_class)}: {alt_probability * 100:.2f}%")

    # Generate advice once per detection and LLM choice; reruns reuse it
    st.subheader("Treatment Advisory")
    if result.get("advice_llm") != st.session_state.llm_option:
        with st.spinner("Generating treatment advice..."):
            try:
                result["advice"] = get_advice(
                    predicted_class,
                    llm_option=st.session_state.llm_option,
                    api_key=get_api_key(),
                    temperature=st.session_state.temperature
                )
            except Exception as e:
                st.error(f"Error generating treatment advice: {e}")
                result["advice"] = get_fallback_advice(predicted_class)
        result["advice_llm"] = st.session_state.llm_option
    st.markdown(result["advice"])

    st.download_button(
        label="Download Treatment Plan",
        data=result["advice"],
        file_name=f"{predicted_class}_treatment.md",
        mime="text/markdown"
    )

# The visitor's own OpenAI key, else the server's OPENAI_API_KEY. Kept in session state,
# never in os.environ, which is shared by every session on the server.
def get_api_key():
    return st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
    
    # Initialize session state
    if 'llm_option' not in st.session_state:
        # GPT models need a key, so start in offline mode unless the server provides one
        st.session_state.llm_option = "gpt4o_mini" if os.getenv("OPENAI_API_KEY") else "offline"

    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
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
        st.sidebar.text_input("OpenAI API Key:", type="password", key="openai_api_key",
                              help="Required for GPT models. Used only for your session.")
        if get_api_key():
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
                st.session_state.temperature = st.slider(
                    "Temperature (Creativity)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Higher values make output more creative but potentially less accurate"
                )
                
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
            get_advice_cache.clear()
            st.success("Advice cache cleared successfully")
    
    elif page == "Disease Detection":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Upload Plant Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            # Display currently selected LLM
            st.info(f"Using {llm_options[st.session_state.llm_option]} for advice generation")
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
                    image = None
                    st.error("Could not read this file as an image. Please upload a JPG or PNG photo of a leaf.")

                if image is not None:
                    st.image(image, caption='Uploaded Image', width="stretch")

                    # Make prediction when button is clicked
                    if st.button('Detect Disease'):
                        with st.spinner('Analyzing image...'):
                            st.session_state.result = {
                                "file_id": uploaded_file.file_id,
                                # Top 3 (class name, probability) pairs
                                "predictions": plant_model.predict(load_model(), image, top_k=3),
                            }
                        st.success('Analysis complete!')

                    # Keep showing this image's result on later reruns (download
                    # button, sidebar changes, ...) instead of only right after the click
                    result = st.session_state.get("result")
                    if result is not None and result["file_id"] == uploaded_file.file_id:
                        with col2:
                            show_results(result)

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
        try:
            # Load confusion matrix visualization
            cm_path = os.path.join(plant_model.MODEL_DIR, 'confusion_matrix.png')
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
    
    