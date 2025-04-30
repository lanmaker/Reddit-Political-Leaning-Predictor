import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import re
import requests
import io
import tempfile
import time
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config for better appearance
st.set_page_config(
    page_title="Reddit Political Leaning Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Improved detection of Streamlit Cloud environment
is_streamlit_cloud = os.getenv("STREAMLIT_SHARING") or os.getenv("STREAMLIT_CLOUD") or os.getenv("IS_STREAMLIT_CLOUD")
# Additional check - typically Streamlit Cloud has this path
if not is_streamlit_cloud and os.path.exists("/mount/src"):
    is_streamlit_cloud = True

# Define paths conditionally
if not is_streamlit_cloud:
    # Only use local directories when not in cloud
    model_dir = os.path.join(script_dir, "models")
    distilbert_dir = os.path.join(model_dir, "distilbert_model")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
else:
    # Placeholder paths for cloud deployment (won't be directly accessed)
    model_dir = None
    distilbert_dir = None
    vectorizer_path = None

# GitHub repo details
GITHUB_USERNAME = "your-github-username"  
GITHUB_REPO = "your-repo-name"  
GITHUB_BRANCH = "main"  

# Define which models are small enough for GitHub (<25MB)
GITHUB_COMPATIBLE_MODELS = [
    "linear_svm.pkl",           # 105 KB
    "logistic_regression.pkl",  # 105 KB
    "multinomial_naive_bayes.pkl", # 417 KB
    "smote_+_logistic_regression.pkl", # 1.5 MB
    "tfidf_vectorizer.pkl"      # 515 KB
]

# Function to get file from GitHub
def download_file_from_github(filename):
    github_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename}"
    st.info(f"Downloading from: {github_url}")
    
    # Add a retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                content_size = len(response.content)
                st.success(f"Downloaded {filename} ({content_size} bytes)")
                
                # Basic validation for pkl files
                if filename.endswith('.pkl'):
                    try:
                        # Try to load the model to verify it's valid
                        model_obj = joblib.load(io.BytesIO(response.content))
                        if hasattr(model_obj, '__class__'):
                            st.success(f"Successfully validated {filename} as {model_obj.__class__.__name__}")
                        return response.content
                    except Exception as e:
                        st.error(f"Downloaded file appears to be invalid: {str(e)}")
                        raise Exception(f"Invalid model file: {str(e)}")
                return response.content
            else:
                st.warning(f"Failed to download {filename}: HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    st.info(f"Retrying download (attempt {attempt+2}/{max_retries})...")
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    raise Exception(f"Failed to download {filename} after {max_retries} attempts: {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Error during download: {str(e)}. Retrying...")
                time.sleep(1)
            else:
                raise Exception(f"Failed to download {filename}: {str(e)}")
                
    raise Exception(f"Failed to download {filename} after {max_retries} attempts")

# Define preprocessing function
def preprocess_text(text):
    """Preprocess text without NLTK dependency"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

# Load the model based on user selection
@st.cache_resource
def load_distilbert_model():
    # Default model
    model_name = "distilbert-base-uncased"
    
    if is_streamlit_cloud:
        # When in Streamlit Cloud, download from Hugging Face
        try:
            st.info("Loading DistilBERT model from Hugging Face Hub (this may take a moment)...")
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            preprocessing_info = {'max_length': 128}
            st.success(f"Successfully loaded DistilBERT model from Hugging Face Hub")
            print(f"Using DistilBERT model: {model_name}")
        except Exception as e:
            st.error(f"Error loading model from Hugging Face: {str(e)}")
            # Create a dummy model with random weights as fallback
            tokenizer = DistilBertTokenizer.from_pretrained(model_name, cache_dir=tempfile.gettempdir())
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2,
                cache_dir=tempfile.gettempdir()
            )
            preprocessing_info = {'max_length': 128}
    else:
        # Local environment - try to load from local directory first
        try:
            if os.path.exists(distilbert_dir):
                tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)
                model = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)
                preprocessing_info = joblib.load(os.path.join(distilbert_dir, 'preprocessing_info.pkl'))
                print("Loaded DistilBERT model from local directory")
            else:
                raise FileNotFoundError(f"Local model directory {distilbert_dir} not found")
        except Exception as e:
            # If local loading fails, download from Hugging Face Hub
            st.warning(f"Failed to load local model: {str(e)}")
            st.info("Loading DistilBERT model from Hugging Face Hub (this may take a moment)...")
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            preprocessing_info = {'max_length': 128}
            print(f"Downloaded DistilBERT model from Hugging Face Hub: {model_name}")
    
    return model, tokenizer, preprocessing_info

@st.cache_resource
def load_traditional_model(model_filename):
    if is_streamlit_cloud:
        try:
            # First, ensure we have a working vectorizer before loading any models
            if not hasattr(load_traditional_model, 'vectorizer'):
                st.info("Downloading vectorizer from GitHub first...")
                try:
                    vectorizer_content = download_file_from_github("tfidf_vectorizer.pkl")
                    vectorizer = joblib.load(io.BytesIO(vectorizer_content))
                    
                    # Validate vectorizer works by trying a simple transform
                    _ = vectorizer.transform(["test vectorizer"])
                    st.success("Vectorizer validated successfully!")
                    
                    # Cache the vectorizer for future use
                    load_traditional_model.vectorizer = vectorizer
                except Exception as e:
                    st.error(f"Vectorizer validation failed: {str(e)}")
                    st.warning("Creating a new vectorizer instead...")
                    vectorizer = create_new_vectorizer()
                    load_traditional_model.vectorizer = vectorizer
            else:
                vectorizer = load_traditional_model.vectorizer
                
            # Now download and load the model
            st.info(f"Downloading {model_filename} from GitHub...")
            model_content = download_file_from_github(model_filename)
            
            # Load model from binary content
            model = joblib.load(io.BytesIO(model_content))
            
            # Verify model is properly fitted by testing a prediction
            verify_model(model, vectorizer)
                
            return model, vectorizer
        except Exception as e:
            st.error(f"Error loading model from GitHub: {str(e)}")
            # Fallback to a simple LogisticRegression model
            st.warning("Using a simple fallback model instead.")
            return create_fallback_model(vectorizer=load_traditional_model.vectorizer if hasattr(load_traditional_model, 'vectorizer') else None)
    else:
        # Local environment
        try:
            model_path = os.path.join(model_dir, model_filename)
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Verify model is properly fitted
            verify_model(model, vectorizer)
            
            return model, vectorizer
        except Exception as e:
            st.error(f"Error loading local model: {str(e)}")
            # Fallback to a simple LogisticRegression model
            st.warning("Using a simple fallback model instead.")
            return create_fallback_model(vectorizer)
            
def create_new_vectorizer():
    """Create a new TF-IDF vectorizer with basic political text content"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    st.info("Creating new TF-IDF vectorizer from scratch...")
    vectorizer = TfidfVectorizer(max_features=5000)
    # Create some diverse political texts to fit the vectorizer
    texts = [
        "conservative right wing republican freedom liberty constitution america patriot",
        "liberal left wing democrat equality diversity inclusion progressive rights",
        "taxes economy jobs government regulation business market freedom capitalism",
        "healthcare education climate welfare social security immigration reform",
        "military defense border security law enforcement police states rights",
        "abortion gun control environment civil rights voting equality justice"
    ]
    vectorizer.fit(texts)
    st.success("New vectorizer created and fitted successfully")
    return vectorizer

def verify_model(model, vectorizer):
    """Verify that the model is properly fitted by attempting a simple prediction"""
    try:
        # Display model type
        model_type = type(model).__name__
        st.info(f"Verifying {model_type} model...")
        
        # Create a simple sample
        sample_text = "this is a test message for political classification"
        
        # Check if vectorizer transforms work
        st.info("Testing vectorizer transformation...")
        X = vectorizer.transform([sample_text])
        st.success(f"Vectorizer transform successful: shape {X.shape}")
        
        # Try predicting
        st.info("Testing model prediction...")
        
        # Check for fitted attributes
        if hasattr(model, 'coef_') and model.coef_ is not None:
            st.success("Model appears to be properly fitted (has coef_ attribute)")
        elif hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            st.success("Model appears to be properly fitted (has feature_importances_ attribute)")
        
        # Try actual prediction
        prediction = model.predict(X)
        st.success(f"Prediction successful: {prediction[0]}")
        
        # Try prediction probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            st.success(f"Probability prediction successful: {probs[0]}")
        
        # If we got here, the model is likely good
        return True
    except Exception as e:
        st.error(f"Model verification failed: {str(e)}")
        # Show more details about the model to help debug
        st.error("Model details:")
        for attr in ['fit', 'predict', 'classes_', 'n_features_in_']:
            if hasattr(model, attr):
                st.info(f"- Has '{attr}' attribute/method")
            else:
                st.warning(f"- Missing '{attr}' attribute/method")
        raise e

def create_fallback_model(vectorizer=None):
    """Create a simple fallback model for when the main model fails to load"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create a new vectorizer if none provided or if the provided one fails
    try:
        if vectorizer is None or not hasattr(vectorizer, 'transform'):
            st.info("Creating new TF-IDF vectorizer for fallback model...")
            vectorizer = TfidfVectorizer(max_features=5000)
            # Fit with some sample data
            sample_texts = [
                "conservative republican right wing freedom",
                "liberal democrat left wing equality",
                "government taxes spending economy jobs",
                "healthcare education climate immigration",
                "sample text for testing the fallback model"
            ]
            vectorizer.fit(sample_texts)
        else:
            # Test the provided vectorizer
            _ = vectorizer.transform(["test"])
    except Exception as e:
        st.warning(f"Vectorizer issue: {str(e)}. Creating new vectorizer.")
        vectorizer = TfidfVectorizer(max_features=5000)
        sample_texts = [
            "conservative republican right wing freedom",
            "liberal democrat left wing equality", 
            "test"
        ]
        vectorizer.fit(sample_texts)
    
    # Create a simple model with minimal weights
    model = LogisticRegression(random_state=42)
    
    # Prepare some training data that roughly represents political tendencies
    train_texts = [
        "government should help people healthcare education welfare",
        "freedom lower taxes less government regulation individual liberty"
    ]
    X_train = vectorizer.transform(train_texts)
    y_train = [0, 1]  # 0 for liberal, 1 for conservative
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model, vectorizer

# Prediction functions
def predict_with_distilbert(text, model, tokenizer, max_length):
    # Preprocess text
    text = preprocess_text(text)
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    # Get probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence = probs[0][prediction].item()
    
    return prediction, confidence

def predict_with_traditional(text, model, vectorizer):
    try:
        # Preprocess text
        text = preprocess_text(text)
        
        # Vectorize
        X = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            confidence = probs[0][prediction]
        else:
            confidence = 0.7  # Default confidence if not available
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # Return a default prediction
        st.warning("Using fallback prediction due to error")
        # Randomly assign with slightly higher chance of liberal (due to example)
        import random
        prediction = 0 if random.random() < 0.55 else 1
        confidence = 0.51  # Low confidence since this is a fallback
        return prediction, confidence

# App UI
st.title("📊 Reddit Political Leaning Predictor")

st.markdown("""
This app predicts whether a Reddit post is more likely to be from a Liberal or Conservative subreddit.
Enter your text below and select a model to make a prediction!
""")

# Add information about the GitHub deployment
if is_streamlit_cloud:
    st.sidebar.markdown("### GitHub Deployment")
    st.sidebar.info(
        "This app is running from GitHub with models under 25MB. "
        "For optimal performance, use the Logistic Regression or Linear SVM models."
    )

# Filter available models based on whether we're in cloud environment
if is_streamlit_cloud:
    # On cloud - only use GitHub-compatible models
    traditional_model_files = GITHUB_COMPATIBLE_MODELS
    # Remove vectorizer from the list as it's not a prediction model
    if "tfidf_vectorizer.pkl" in traditional_model_files:
        traditional_model_files.remove("tfidf_vectorizer.pkl")
else:
    # Only try to access local directory when not in cloud
    try:
        traditional_model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and not f.startswith('tfidf')]
        if not traditional_model_files:  # If directory exists but empty
            st.warning("No model files found in local directory. Using default model list.")
            traditional_model_files = GITHUB_COMPATIBLE_MODELS
    except (FileNotFoundError, TypeError):
        st.warning("Could not access local model directory. Using default model list.")
        traditional_model_files = GITHUB_COMPATIBLE_MODELS

# Add model information in the sidebar
st.sidebar.markdown("### Model Information")
st.sidebar.markdown("""
**Traditional ML** models are effective for text classification:

- Fast prediction time
- Good accuracy
- Small model size for cloud deployment
""")

# Model selection in sidebar
selected_model = st.sidebar.selectbox(
    "Select Model:",
    traditional_model_files,
    format_func=lambda x: x.replace('_', ' ').replace('.pkl', '').title()
)

# Load the selected model
model, vectorizer = load_traditional_model(selected_model)

# Initialize session state for text input
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Text input
text_input = st.text_area("Enter Reddit post text:", value=st.session_state.text_input, height=150)

# Add example texts
st.markdown("### Or try one of these examples:")
example_texts = [
    "I believe we need to strengthen our social safety net and provide healthcare for all citizens. The government needs to do more to protect vulnerable populations.",
    "Government regulations are strangling small businesses. We need to cut taxes and let the free market handle the economy better than bureaucrats ever could.",
    "Climate change is the most pressing issue of our time and we need immediate action to prevent catastrophic consequences.",
    "Second amendment rights are fundamental and should not be infringed. Law-abiding citizens should have the right to own firearms."
]

# Create a 2x2 grid of example buttons
cols = st.columns(2)
for i, example in enumerate(example_texts):
    col_idx = i % 2
    if cols[col_idx].button(f"Example {i+1}", key=f"ex_{i}"):
        st.session_state.text_input = example
        st.rerun()

# Make prediction when button is clicked
if st.button("Predict Political Leaning", type="primary"):
    if not text_input:
        st.error("Please enter some text to analyze.")
    else:
        # Display spinner while processing
        with st.spinner("Analyzing text..."):
            prediction, confidence = predict_with_traditional(text_input, model, vectorizer)
            
            # Display results with improved visualization
            st.markdown("### Prediction Results")
            
            # Get label from prediction
            if prediction == 0:
                label = "Liberal"
                color = "#3498db"  # Blue
            else:
                label = "Conservative"
                color = "#e74c3c"  # Red
                
            # Create results container with custom styling
            result_container = st.container()
            with result_container:
                st.markdown(f"<h2 style='color:{color};text-align:center;'>This post appears to be <span style='font-size:1.3em;'>{label}</span></h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;font-size:1.1em;'>Confidence: {confidence*100:.1f}%</p>", unsafe_allow_html=True)
                
                # Create a progress bar for visualization
                st.progress(confidence)
                
            # Add explanation of results
            st.markdown("### Analysis")
            st.write(f"The model has classified this text as {label} with {confidence*100:.1f}% confidence.")
            st.write("The model identifies important words and phrases that are statistically associated with political leanings based on historical data from Reddit.")

# Add footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>This model was trained on Reddit posts from political subreddits. It may not be accurate for all political content.</small>
</div>
""", unsafe_allow_html=True) 

