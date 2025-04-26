import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Set page config for better appearance
st.set_page_config(
    page_title="Reddit Political Leaning Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "models")
distilbert_dir = os.path.join(model_dir, "distilbert_model")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
best_model_path = os.path.join(model_dir, "logistic_regression.pkl")  # or whatever your best model was

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
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)
    model = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)
    preprocessing_info = joblib.load(os.path.join(distilbert_dir, 'preprocessing_info.pkl'))
    return model, tokenizer, preprocessing_info

@st.cache_resource
def load_traditional_model(model_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
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
        confidence = 0.5  # Default confidence if not available
    
    return prediction, confidence

# App UI
st.title("📊 Reddit Political Leaning Predictor")

st.markdown("""
This app predicts whether a Reddit post is more likely to be from a Liberal or Conservative subreddit.
Enter your text below and select a model to make a prediction!
""")

# Model selection
model_type = st.sidebar.radio(
    "Select Model Type:",
    ["DistilBERT (Transformer)", "Traditional ML (Faster)"]
)

# Add some model information in the sidebar
st.sidebar.markdown("### Model Information")
if model_type == "DistilBERT (Transformer)":
    st.sidebar.markdown("""
    **DistilBERT** is a smaller, faster transformer model that retains 97% of BERT's performance.
    
    - Higher accuracy
    - Better understanding of context
    - Slower prediction time
    """)
else:
    st.sidebar.markdown("""
    **Traditional ML** models are simpler but very effective for text classification.
    
    - Faster prediction time
    - Good accuracy
    - Less memory usage
    """)

# Load appropriate model
if model_type == "DistilBERT (Transformer)":
    model, tokenizer, preprocessing_info = load_distilbert_model()
    max_length = preprocessing_info.get('max_length', 128)
else:
    # For Traditional ML, let user select which model to use
    traditional_model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and not f.startswith('tfidf')]
    selected_model = st.sidebar.selectbox(
        "Select Traditional Model:",
        traditional_model_files,
        format_func=lambda x: x.replace('_', ' ').replace('.pkl', '').title()
    )
    model_path = os.path.join(model_dir, selected_model)
    model, vectorizer = load_traditional_model(model_path)

# Text input
text_input = st.text_area("Enter Reddit post text:", height=150)

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
        text_input = example
        st.session_state.text_input = example
        # Need to rerun to update the text area
        st.rerun()

# Make prediction when button is clicked
if st.button("Predict Political Leaning", type="primary"):
    if not text_input:
        st.error("Please enter some text to analyze.")
    else:
        # Display spinner while processing
        with st.spinner("Analyzing text..."):
            if model_type == "DistilBERT (Transformer)":
                prediction, confidence = predict_with_distilbert(text_input, model, tokenizer, max_length)
            else:
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
            
            if model_type == "DistilBERT (Transformer)":
                st.write("DistilBERT analyzes the context and semantic meaning of your text to identify political leanings based on language patterns it learned during training.")
            else:
                st.write("The traditional model identifies important words and phrases that are statistically associated with political leanings based on historical data.")

# Add footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>This model was trained on Reddit posts from political subreddits. It may not be accurate for all political content.</small>
</div>
""", unsafe_allow_html=True) 