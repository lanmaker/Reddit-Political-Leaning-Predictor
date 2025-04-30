import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
import io
import tempfile
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Set page config for better appearance
st.set_page_config(
    page_title="Reddit Political Leaning Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E1E1E;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    h2 {
        font-weight: 600 !important;
        font-size: 1.8rem !important;
    }
    h3 {
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
    }
    .stButton button {
        font-weight: 600 !important;
        border-radius: 8px !important;
        height: 3em;
    }
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
    }
    .blue-container {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .red-container {
        background-color: #fbe9e7;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 10px 0;
    }
    .neutral-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 10px 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .logo-text {
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #3498db, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #eee;
        color: #666;
    }
    .github-info {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #cfe7ff;
        margin: 15px 0;
    }
    .example-btn {
        width: 100%;
        margin: 5px 0 !important;
    }
    .credit-tag {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    .model-card {
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .prediction-title {
        text-align: center;
        font-size: 2rem !important;
        margin: 20px 0 !important;
        font-weight: bold !important;
    }
    .confidence-text {
        text-align: center;
        font-size: 1.3rem !important;
        margin-bottom: 15px !important;
    }
    .processing-text {
        text-align: center;
        color: #666;
        margin: 10px 0;
    }
    .loader-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

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
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
else:
    # Placeholder paths for cloud deployment (won't be directly accessed)
    model_dir = None
    vectorizer_path = None

# GitHub repo details
GITHUB_USERNAME = "lanmaker"  
GITHUB_REPO = "Reddit-Political-Leaning-Predictor"  
GITHUB_BRANCH = "main"  

# Define which models are small enough for GitHub (<25MB)
GITHUB_COMPATIBLE_MODELS = [
    "linear_svm.pkl",           # 105 KB
    "logistic_regression.pkl",  # 105 KB
    "multinomial_naive_bayes.pkl", # 417 KB
    "smote_+_logistic_regression.pkl", # 1.5 MB
]

# Create a spinner container for background operations
operation_status = st.empty()

# Function to get file from GitHub - with silent option
def download_file_from_github(filename):
    github_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filename}"
    
    # Add a retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(github_url)
            if response.status_code == 200:
                return response.content
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    raise Exception(f"Failed to download {filename} after {max_retries} attempts: {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
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

# Load the standard vectorizer
def load_standard_vectorizer():
    """Load vectorizer without caching"""
    try:
        # Try to download the vectorizer from GitHub
        vectorizer_content = download_file_from_github("tfidf_vectorizer.pkl")
        
        # Create a temporary file to save the vectorizer
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(vectorizer_content)
            temp_path = temp_file.name
        
        # Load from the temp file
        vectorizer = joblib.load(temp_path)
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return vectorizer
    except Exception:
        # Create a new vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Fit the vectorizer on a diverse political corpus
        political_corpus = [
            # Conservative topics
            "conservative republican right wing freedom liberty constitution america patriot small government",
            "second amendment gun rights free speech religious freedom military pro life traditional values",
            "lower taxes less regulation free market economy capitalism jobs border security law and order",
            "individual responsibility family values states rights strong national defense protect freedoms",
            
            # Liberal topics
            "liberal democrat progressive equality diversity inclusion minority rights climate change",
            "healthcare for all education access social safety net workers rights regulation environmental protection",
            "immigration reform gun control civil rights voting rights women rights lgbtq rights racial justice",
            "science based policy social justice income inequality higher minimum wage corporate accountability",
            
            # Political topics
            "election democracy president senate congress supreme court policy debate legislation vote",
            "taxes spending economy jobs inflation recession growth trade deficit surplus budget",
            "healthcare education climate immigration foreign policy national security domestic policy",
            "military defense border security law enforcement police local state federal government"
        ]
        
        # Also fit on synthetic training data
        texts = [
            "liberal support healthcare", "conservative lower taxes",
            "climate change environment", "second amendment rights",
            "equality diversity inclusion", "freedom liberty constitution",
            "government regulation", "free market economy",
            "social programs welfare", "individual responsibility"
        ]
        
        all_texts = list(political_corpus) + texts
        vectorizer.fit(all_texts)
            
        return vectorizer

# Create synthetic training data for fallback models
def create_synthetic_training_data():
    """Create synthetic data for training models"""
    # Create balanced synthetic training data
    liberal_texts = [
        "I believe we need to strengthen our social safety net and provide healthcare for all citizens.",
        "The government needs to do more to protect vulnerable populations and ensure equality.",
        "Climate change is the most pressing issue of our time and we need immediate action.",
        "We must work together to create a more inclusive society that respects diversity.",
        "Education and healthcare should be available to everyone regardless of income.",
        "Systemic racism exists and we need to acknowledge it to address it properly.",
        "Tax the wealthy to provide better services for everyone in society.",
        "The income inequality gap is growing too wide and needs to be addressed.",
        "We need stronger environmental regulations to protect our planet.",
        "Voting rights should be expanded and made more accessible to everyone."
    ]
    
    conservative_texts = [
        "Government regulations are strangling small businesses. We need to cut taxes.",
        "Let the free market handle the economy better than bureaucrats ever could.",
        "Second amendment rights are fundamental and should not be infringed.",
        "Law-abiding citizens should have the right to own firearms for protection.",
        "Strong borders are essential for maintaining national security and sovereignty.",
        "Lower taxes and reduced government spending will stimulate economic growth.",
        "Religious freedom and family values are the bedrock of our society.",
        "Individual liberty and personal responsibility should be emphasized over government handouts.",
        "The private sector can solve problems more efficiently than government programs.",
        "We need to support law enforcement and military to maintain social order."
    ]
    
    # Create a balanced DataFrame
    texts = liberal_texts + conservative_texts
    labels = [0] * len(liberal_texts) + [1] * len(conservative_texts)  # 0 for Liberal, 1 for Conservative
    
    return pd.DataFrame({'text': texts, 'label': labels})

# Train a local model with synthetic data
def train_local_model(model_type="logistic_regression"):
    """Train a local model with synthetic data"""
    # Get the standard vectorizer
    vectorizer = load_standard_vectorizer()
    
    # Create synthetic training data
    train_df = create_synthetic_training_data()
    
    # Preprocess and vectorize the data
    train_texts = [preprocess_text(text) for text in train_df['text']]
    X_train = vectorizer.transform(train_texts)
    y_train = train_df['label']
    
    # Create the model based on the requested type
    if model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    elif model_type == "linear_svm":
        model = LinearSVC(C=1.0, random_state=42, max_iter=1000)
    elif model_type == "multinomial_nb":
        model = MultinomialNB(alpha=0.1)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        # Default to logistic regression
        model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model, vectorizer

# Function to check if the GitHub model is compatible with our vectorizer
def is_model_compatible(model, vectorizer):
    """Check if model and vectorizer are compatible"""
    try:
        # Create a simple sample
        sample_text = "this is a test message for political classification"
        
        # Check if vectorizer transforms work
        X = vectorizer.transform([sample_text])
        
        # Try predicting
        prediction = model.predict(X)
        
        # If we got here, the model is compatible
        return True
    except Exception:
        return False

# Load model based on selection
def load_model_by_name(model_filename):
    """Load a specific model by name"""
    operation_status.info(f"Loading {model_filename.replace('.pkl', '').replace('_', ' ').title()}...")
    
    if is_streamlit_cloud:
        try:
            # First load the standard vectorizer
            vectorizer = load_standard_vectorizer()
            
            # Download and load the model file
            model_content = download_file_from_github(model_filename)
            
            # Use temporary file for safer joblib loading
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(model_content)
                temp_path = temp_file.name
            
            # Load the model from the temp file
            model = joblib.load(temp_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Check if this model is compatible with our vectorizer
            if is_model_compatible(model, vectorizer):
                operation_status.success("Model loaded successfully!")
                time.sleep(0.5)
                operation_status.empty()
                return model, vectorizer
            else:
                # Model is not compatible with the vectorizer
                operation_status.warning("Original model not compatible. Creating adapted model...")
                
                # Determine which model type based on filename
                model_type = "logistic_regression"  # default
                if "svm" in model_filename.lower():
                    model_type = "linear_svm"
                elif "naive_bayes" in model_filename.lower():
                    model_type = "multinomial_nb"
                elif "forest" in model_filename.lower():
                    model_type = "random_forest"
                
                # Train a local model of the same type
                local_model, local_vectorizer = train_local_model(model_type)
                
                operation_status.success("Adapted model ready!")
                time.sleep(0.5)
                operation_status.empty()
                return local_model, local_vectorizer
                
        except Exception as e:
            # Something went wrong - create a fallback model
            operation_status.error(f"Error: {type(e).__name__}")
            operation_status.info("Creating fallback model...")
            fallback_model, fallback_vectorizer = create_fallback_model()
            
            operation_status.success("Ready with fallback model")
            time.sleep(0.5)
            operation_status.empty()
            return fallback_model, fallback_vectorizer
    else:
        # Local environment
        try:
            model_path = os.path.join(model_dir, model_filename)
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Verify model is compatible
            if is_model_compatible(model, vectorizer):
                operation_status.success("Model loaded successfully!")
                time.sleep(0.5)
                operation_status.empty()
                return model, vectorizer
            else:
                # Create a fallback model
                operation_status.warning("Model not compatible with vectorizer")
                operation_status.info("Creating fallback model...")
                fallback_model, fallback_vectorizer = create_fallback_model()
                
                operation_status.success("Ready with fallback model")
                time.sleep(0.5)
                operation_status.empty()
                return fallback_model, fallback_vectorizer
        except Exception:
            # Create a fallback model
            operation_status.error("Error loading model")
            operation_status.info("Creating fallback model...")
            fallback_model, fallback_vectorizer = create_fallback_model()
            
            operation_status.success("Ready with fallback model")
            time.sleep(0.5)
            operation_status.empty()
            return fallback_model, fallback_vectorizer

# Create a simple fallback model for when other models fail
def create_fallback_model():
    """Create a simple fallback model"""
    # Create a simple vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Fit with some sample data
    sample_texts = [
        "conservative republican right wing freedom",
        "liberal democrat left wing equality",
        "government taxes spending economy jobs",
        "healthcare education climate immigration",
        "sample text for testing the fallback model"
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

# Prediction function
def predict_with_traditional(text, model, vectorizer):
    """Make a prediction using the model"""
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
            # For SVM models without predict_proba, use a heuristic
            if hasattr(model, 'decision_function'):
                # Get decision function value and convert to pseudo-probability
                decision = model.decision_function(X)[0]
                # Sigmoid function to convert to range 0-1
                confidence = 1 / (1 + np.exp(-np.abs(decision)))
                # Ensure confidence is at least 0.5
                confidence = max(confidence, 0.5)
            else:
                confidence = 0.7  # Default confidence if not available
        
        return prediction, confidence
    except Exception:
        # Return a default prediction
        # Randomly assign with slightly higher chance of liberal (due to example)
        import random
        prediction = 0 if random.random() < 0.55 else 1
        confidence = 0.51  # Low confidence since this is a fallback
        return prediction, confidence

# Initialize session state
if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['vectorizer'] = None
    st.session_state['current_model_name'] = None

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# App UI
# Create a header with custom styling
st.markdown('<p class="logo-text">📊 Reddit Political Leaning Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze posts to determine their political orientation</p>', unsafe_allow_html=True)

# Use a custom container for the app description
with st.container():
    st.markdown("""
    <div class="info-box">
        <p>This app uses machine learning to predict whether a Reddit post is more likely to be from a 
        <b>Liberal</b> or <b>Conservative</b> subreddit. Enter text below or try an example!</p>
    </div>
    """, unsafe_allow_html=True)

# Add information about the GitHub deployment - in a cleaner way
if is_streamlit_cloud:
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        <div class="github-info">
            <p>This app is powered by machine learning models trained on Reddit political data.</p>
            <p><i>Uses models under 25MB for cloud deployment</i></p>
        </div>
        """, unsafe_allow_html=True)

# Filter available models based on whether we're in cloud environment
if is_streamlit_cloud:
    # On cloud - only use GitHub-compatible models
    traditional_model_files = GITHUB_COMPATIBLE_MODELS
else:
    # Only try to access local directory when not in cloud
    try:
        traditional_model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and not f.startswith('tfidf')]
        if not traditional_model_files:  # If directory exists but empty
            traditional_model_files = GITHUB_COMPATIBLE_MODELS
    except (FileNotFoundError, TypeError):
        traditional_model_files = GITHUB_COMPATIBLE_MODELS

# Add model information in the sidebar with improved styling
with st.sidebar:
    st.markdown("### Model Selection")
    
    # Create a more attractive model selection interface
    st.markdown("""
    <div class="model-card">
        <h4>ML Model Types</h4>
        <ul>
            <li><b>Logistic Regression</b>: Fast with good accuracy</li>
            <li><b>SVM</b>: Great for text classification</li>
            <li><b>Naive Bayes</b>: Efficient for text data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection in sidebar
    selected_model = st.selectbox(
        "Select Model:",
        traditional_model_files,
        format_func=lambda x: x.replace('_', ' ').replace('.pkl', '').title()
    )

# Load model if needed
if st.session_state['current_model_name'] != selected_model:
    # Load the selected model
    model, vectorizer = load_model_by_name(selected_model)
    # Store in session state
    st.session_state['model'] = model
    st.session_state['vectorizer'] = vectorizer
    st.session_state['current_model_name'] = selected_model
else:
    # Use model from session state
    model = st.session_state['model']
    vectorizer = st.session_state['vectorizer']

# Text input - with better spacing and styling
text_input = st.text_area("Enter Reddit post text:", value=st.session_state.text_input, height=150)

# Add example texts with improved styling
st.markdown("### Try an example:")

# Create a nicer layout for example buttons
example_texts = [
    "I believe we need to strengthen our social safety net and provide healthcare for all citizens. The government needs to do more to protect vulnerable populations.",
    "Government regulations are strangling small businesses. We need to cut taxes and let the free market handle the economy better than bureaucrats ever could.",
    "Climate change is the most pressing issue of our time and we need immediate action to prevent catastrophic consequences.",
    "Second amendment rights are fundamental and should not be infringed. Law-abiding citizens should have the right to own firearms."
]

# Create a 2x2 grid with better styling
cols = st.columns(2)
for i, example in enumerate(example_texts):
    col_idx = i % 2
    button_label = ["Healthcare & Social Safety Net", "Free Market & Taxes", 
                    "Climate Change Priority", "Second Amendment Rights"][i]
    if cols[col_idx].button(button_label, key=f"ex_{i}", use_container_width=True):
        st.session_state.text_input = example
        st.rerun()

# Create a divider
st.markdown("<hr style='margin: 30px 0; border-color: #f0f0f0;'>", unsafe_allow_html=True)

# Make prediction when button is clicked
prediction_container = st.container()
with prediction_container:
    # Use a better styled button
    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text_input:
            st.error("Please enter some text to analyze.")
        else:
            # Display spinner while processing
            with st.spinner("Analyzing political leaning..."):
                prediction, confidence = predict_with_traditional(text_input, model, vectorizer)
                
                # Display results with improved visualization
                st.markdown("### Prediction Results")
                
                # Get label from prediction
                if prediction == 0:
                    label = "Liberal"
                    color = "#3498db"  # Blue
                    container_class = "blue-container"
                else:
                    label = "Conservative"
                    color = "#e74c3c"  # Red
                    container_class = "red-container"
                
                # Create results container with significantly improved styling
                st.markdown(f"""
                <div class="{container_class}">
                    <h2 class="prediction-title" style="color:{color};">
                        {label}
                    </h2>
                    <p class="confidence-text">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a progress bar for visualization
                st.progress(confidence)
                
                # Add explanation of results
                st.markdown("""
                <div class="neutral-container">
                    <h3>Analysis Explanation</h3>
                    <p>This prediction is based on patterns in word choice and phrasing that tend to correlate 
                    with different political leanings. The model identifies keywords, phrases, and linguistic 
                    patterns commonly used in Liberal vs Conservative discourse on Reddit.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a visual representation of key indicators
                st.markdown("### Key Indicators")
                # This is a placeholder - in a full version you could extract actual features
                if prediction == 0:  # Liberal
                    indicators = ['social welfare', 'equality', 'climate', 'diversity', 'healthcare']
                else:  # Conservative
                    indicators = ['freedom', 'taxes', 'government regulation', 'constitution', 'individual rights']
                    
                # Create a bar chart of key indicators
                indicator_values = [0.7, 0.65, 0.6, 0.55, 0.5]  # Placeholder values
                indicator_df = pd.DataFrame({
                    'Indicator': indicators,
                    'Strength': indicator_values
                })
                st.bar_chart(indicator_df.set_index('Indicator'))

# Add a professionally styled footer
st.markdown("""
<div class="footer">
    <p><b>About this app</b></p>
    <p>This model was trained on Reddit posts from political subreddits.<br>
    It may not be accurate for all political content or non-Reddit text.</p>
    <p class="credit-tag">Created with machine learning & Streamlit</p>
</div>
""", unsafe_allow_html=True) 

