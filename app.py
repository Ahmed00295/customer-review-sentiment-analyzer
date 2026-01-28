"""
Streamlit Dashboard - Customer Review Sentiment Analyzer
=========================================================

This is the main application interface that demonstrates
how all OOP classes work together:
- DataLoader: Handles data preprocessing
- BaseModel (subclasses): Train and predict
- Evaluator: Generate metrics and visualizations

The dashboard allows users to:
1. Upload their own CSV dataset
2. Select and train different ML models
3. Make predictions on custom text
4. View analytics and performance metrics
 
Author: Muhammad Ahmad
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from typing import Optional

# Import our OOP classes
from data_loader import DataLoader
from models import (
    BaseModel, SVMModel, RandomForestModel, KNNModel,
    NaiveBayesModel, LogisticRegressionModel, ModelFactory
)
from evaluator import Evaluator

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Sentiment Analyzer | AI-Powered Reviews Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# BACKGROUND IMAGE SETUP
# ============================================
def set_online_bg():
    """Sets a beautiful online abstract technology background."""
    bg_url = "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=2072"
    
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Make the sidebar transparent so the background shows through */
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.4) !important;
    }}
    </style>
    '''
    return page_bg_img

# Apply the background immediately
st.markdown(set_online_bg(), unsafe_allow_html=True)

# ============================================
# CUSTOM CSS STYLING - Modern Dark Theme
# ============================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Remove default padding/margins */
    .stApp {
        background: transparent;
        font-family: 'Inter', sans-serif;
        color: #ffffff; /* Default text color to white */
    }
    
    /* Ensure all widget labels are white and visible */
    label, .stMarkdown p, [data-testid="stWidgetLabel"] p {
        color: white !important;
    }
    
    /* Input field text visibility */
    .stSelectbox div[data-baseweb="select"] > div {
        color: white !important;
    }
    
    /* Hide top header bar and reduce whitespace */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        height: 0px !important;
        min-height: 0px !important;
    }
    
    /* Remove top padding from main content */
    .main .block-container {
        padding-top: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Main Header Styles - Enhanced */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="30" r="3" fill="rgba(255,255,255,0.1)"/><circle cx="40" cy="70" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="90" cy="80" r="2" fill="rgba(255,255,255,0.1)"/></svg>');
        opacity: 0.5;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Hero Banner Styles */
    .hero-banner {
        width: 100%;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    /* Card Styles - Glassmorphism */
    .glass-card {
        background: rgba(15, 15, 35, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Result Card Styles */
    .result-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .positive-result {
        border-color: #00ff88;
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 200, 100, 0.15) 100%);
    }
    
    .negative-result {
        border-color: #ff6b6b;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(200, 80, 80, 0.15) 100%);
    }
    
    .sentiment-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .sentiment-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: rgba(255,255,255,0.1);
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 15, 35, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: white;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Example Buttons - Special Styling */
    .example-btn {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%) !important;
    }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Text Input Styles */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        border-radius: 12px !important;
        color: #1a1a2e !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(0, 0, 0, 0.4) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* Info/Success Messages */
    .stSuccess, .stInfo {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 12px;
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.5);
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    /* Hide Streamlit Branding and Deploy Button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    .animate-float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables."""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = Evaluator()
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""

init_session_state()


# ============================================
# HELPER FUNCTIONS
# ============================================
def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for demonstration."""
    reviews = [
        # Positive Reviews
        ("This product exceeded all my expectations! Absolutely love it.", "positive"),
        ("Amazing quality and fast shipping. Highly recommend!", "positive"),
        ("Best purchase I've made this year. Works perfectly.", "positive"),
        ("Great value for money. Very satisfied with my order.", "positive"),
        ("Excellent customer service and fantastic product.", "positive"),
        ("Love this! It's exactly what I was looking for.", "positive"),
        ("Superb quality, will definitely buy again.", "positive"),
        ("Perfect fit and amazing design. 5 stars!", "positive"),
        ("Impressed with the build quality. Worth every penny.", "positive"),
        ("Outstanding product. My family loves it too!", "positive"),
        ("Wonderful experience from start to finish.", "positive"),
        ("This made my life so much easier. Thank you!", "positive"),
        ("Couldn't be happier with this purchase.", "positive"),
        ("Premium quality at an affordable price.", "positive"),
        ("Fast delivery and product works like a charm.", "positive"),
        
        # Negative Reviews
        ("Terrible quality. Broke after one day of use.", "negative"),
        ("Complete waste of money. Very disappointed.", "negative"),
        ("Product looks nothing like the pictures. Misleading.", "negative"),
        ("Horrible customer service. Never buying again.", "negative"),
        ("Item arrived damaged and took forever to ship.", "negative"),
        ("Does not work as advertised. Total scam.", "negative"),
        ("Cheaply made product. Falls apart easily.", "negative"),
        ("Not worth the price at all. Save your money.", "negative"),
        ("Worst purchase ever. Returning immediately.", "negative"),
        ("Product stopped working after a week. Junk.", "negative"),
        ("Very poor quality material. Not recommended.", "negative"),
        ("Shipping took 3 weeks and item was broken.", "negative"),
        ("False advertising. Product is completely different.", "negative"),
        ("Disappointed with this purchase. Low quality.", "negative"),
        ("Would give zero stars if I could. Awful.", "negative"),
        
        # More varied reviews
        ("Absolutely fantastic! Changed my daily routine for the better.", "positive"),
        ("The worst experience I've had shopping online.", "negative"),
        ("So happy with this product. My friends are jealous!", "positive"),
        ("Useless product. Doesn't do what it claims.", "negative"),
        ("This is a game changer! Highly recommend to everyone.", "positive"),
        ("Don't waste your time or money on this.", "negative"),
        ("Exceeded expectations in every way possible.", "positive"),
        ("Arrived broken and customer support was unhelpful.", "negative"),
        ("Will be ordering more for gifts. Excellent!", "positive"),
        ("Quality is terrible. Felt like a cheap knockoff.", "negative"),
    ]
    
    # Expand the dataset by repeating and slightly modifying
    expanded_reviews = []
    prefixes = ["", "Update: ", "After using: ", ""]
    suffixes = ["", " Overall satisfied.", " Would recommend.", " Happy customer."]
    neg_suffixes = ["", " Very upset.", " Not happy.", " Waste of money."]
    
    for review, sentiment in reviews:
        expanded_reviews.append((review, sentiment))
        # Add variations
        for prefix in prefixes[:2]:
            for suffix in (suffixes if sentiment == "positive" else neg_suffixes)[:2]:
                if prefix or suffix:
                    expanded_reviews.append((prefix + review + suffix, sentiment))
    
    df = pd.DataFrame(expanded_reviews, columns=['review_text', 'sentiment'])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def display_sentiment_result(sentiment: str, confidence: float):
    """Display the sentiment prediction with styling."""
    is_positive = str(sentiment).lower() in ['positive', '1', 'pos']
    
    emoji = "😊" if is_positive else "😔"
    label = "POSITIVE" if is_positive else "NEGATIVE"
    color = "#00ff88" if is_positive else "#ff6b6b"
    result_class = "positive-result" if is_positive else "negative-result"
    
    st.markdown(f"""
    <div class="result-card {result_class}">
        <div class="sentiment-emoji">{emoji}</div>
        <div class="sentiment-label" style="color: {color};">{label}</div>
        <div style="color: white; margin-top: 0.5rem;">
            Confidence: {confidence:.1%}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_metrics_cards(metrics: dict):
    """Display metrics in beautiful cards."""
    cols = st.columns(4)
    
    metric_data = [
        ("Accuracy", metrics.get('accuracy', 0), "🎯"),
        ("Precision", metrics.get('precision', 0), "🔍"),
        ("Recall", metrics.get('recall', 0), "📊"),
        ("F1 Score", metrics.get('f1_score', 0), "⚡")
    ]
    
    for col, (label, value, icon) in zip(cols, metric_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="metric-value">{value*100:.1f}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application function."""
    
    # ===== HEADER =====
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Customer Review Sentiment Analyzer</h1>
        <p>AI-Powered Sentiment Analysis using Machine Learning & NLP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero image is now used as app background
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("---")
        
        # Dataset Section
        st.markdown("#### 📁 Dataset")
        
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Dataset", "Upload CSV File"],
            help="Select whether to use our demo dataset or upload your own"
        )
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="CSV should have 'review_text' and 'sentiment' columns"
            )
            
            if uploaded_file is not None:
                try:
                    # Try reading with different encodings to handle common Excel/System exports
                    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        try:
                            # Reset pointer and try reading
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            break
                        except UnicodeDecodeError as e:
                            last_error = e
                            continue
                    
                    if df is not None:
                        st.session_state.uploaded_data = df
                        st.success(f"✅ Loaded {len(df)} reviews using {encoding} encoding")
                    else:
                        raise last_error if last_error else Exception("Failed to decode file with available encodings")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            
            # Show column mapping if data is loaded
            if data_source == "Upload CSV File" and hasattr(st.session_state, 'uploaded_data'):
                st.markdown("---")
                st.markdown("#### 🔗 Column Mapping")
                cols = st.session_state.uploaded_data.columns.tolist()
                
                # Default logic for review text
                rev_default = cols.index('review_text') if 'review_text' in cols else 0
                text_col = st.selectbox(
                    "Review Text Column:",
                    cols,
                    index=rev_default,
                    key="text_col_select"
                )
                
                # Default logic for sentiment
                sent_default = cols.index('sentiment') if 'sentiment' in cols else min(1, len(cols)-1)
                sent_col = st.selectbox(
                    "Sentiment Column:",
                    cols,
                    index=sent_default,
                    key="sent_col_select"
                )
                
                # Validation Warning
                if hasattr(st.session_state, 'uploaded_data'):
                    unique_count = st.session_state.uploaded_data[sent_col].nunique()
                    if unique_count > 10:
                        st.warning(f"⚠️ Warning: '{sent_col}' has {unique_count} unique values. Are you sure this is the sentiment label column?")
        
        st.markdown("---")
        
        # Model Selection Section
        st.markdown("#### 🤖 Model Selection")
        
        model_options = {
            'svm': '🎯 Support Vector Machine (SVM)',
            'random_forest': '🌲 Random Forest',
            'knn': '👥 K-Nearest Neighbors (KNN)',
            'naive_bayes': '📊 Naive Bayes',
            'logistic_regression': '📈 Logistic Regression'
        }
        
        selected_model = st.selectbox(
            "Select ML Model:",
            list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Choose the machine learning algorithm for sentiment classification"
        )
        
        st.markdown("---")
        
        # Training Section
        st.markdown("#### ⚡ Training")
        
        train_button = st.button("🚀 Train Model", use_container_width=True)
        
        if train_button:
            with st.spinner("Preparing data and training model..."):
                # Load data
                if data_source == "Use Sample Dataset":
                    df = create_sample_dataset()
                else:
                    if hasattr(st.session_state, 'uploaded_data'):
                        df = st.session_state.uploaded_data
                    else:
                        st.error("Please upload a CSV file first!")
                        return
                
                # Initialize DataLoader
                loader = DataLoader()
                loader.load_data_from_dataframe(df)
                loader.handle_missing_values()
                
                # Use specified columns if uploaded
                if data_source == "Upload CSV File":
                    t_col = st.session_state.text_col_select
                    s_col = st.session_state.sent_col_select
                else:
                    t_col = 'review_text'
                    s_col = 'sentiment'
                
                loader.preprocess_all_text(text_column=t_col)
                
                # Get train/test split
                X_train, X_test, y_train, y_test = loader.get_train_test_split(target_column=s_col)
                
                # Store in session state
                st.session_state.data_loader = loader
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Create and train model using Factory Pattern
                model = ModelFactory.create_model(selected_model)
                model.train(X_train, y_train)
                
                # Make predictions and calculate metrics
                y_pred = model.predict(X_test)
                
                evaluator = Evaluator()
                # Dynamically set labels based on unique values in target
                unique_labels = sorted(list(set(np.concatenate([y_train, y_test]))))
                evaluator.set_labels([str(l) for l in unique_labels])
                metrics = evaluator.calculate_metrics(y_test, y_pred, model.get_model_name())
                
                # Store results
                st.session_state.current_model = model
                st.session_state.evaluator = evaluator
                st.session_state.y_pred = y_pred
                st.session_state.metrics = metrics
                st.session_state.is_trained = True
                st.session_state.trained_models[selected_model] = {
                    'model': model,
                    'metrics': metrics
                }
                
            st.success(f"✅ {model.get_model_name()} trained successfully!")
        
        # Show training status
        if st.session_state.is_trained:
            st.markdown("---")
            st.markdown("#### 📊 Training Status")
            st.info(f"Active Model: {st.session_state.current_model.get_model_name()}")
            
            if st.session_state.metrics:
                st.metric(
                    "Current Accuracy",
                    f"{st.session_state.metrics['accuracy']*100:.1f}%"
                )
    
    # ===== MAIN CONTENT =====
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict Sentiment",
        "📊 Analytics Dashboard",
        "📈 Model Comparison",
        "ℹ️ About"
    ])
    
    # ===== TAB 1: PREDICTION =====
    with tab1:
        st.markdown('<h2 class="section-header">🔮 Sentiment Prediction</h2>', unsafe_allow_html=True)
        
        if not st.session_state.is_trained:
            st.warning("⚠️ Please train a model first using the sidebar controls.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: white; margin-bottom: 1rem;">📝 Enter Your Review</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Get example text from session state if available
                default_text = st.session_state.example_text if st.session_state.example_text else ""
                
                user_review = st.text_area(
                    "Type or paste a customer review below:",
                    value=default_text,
                    height=150,
                    placeholder="Example: This product is amazing! I love the quality and fast delivery...",
                    key="review_input_area"
                )
                
                # Clear example text after using it
                if st.session_state.example_text:
                    st.session_state.example_text = ""
                
                predict_button = st.button("🎯 Analyze Sentiment", use_container_width=True)
                
                if predict_button and user_review.strip():
                    with st.spinner("Analyzing sentiment..."):
                        # Transform the text using the fitted vectorizer
                        X_new = st.session_state.data_loader.transform_text(user_review)
                        
                        # Get prediction
                        prediction = st.session_state.current_model.predict(X_new)[0]
                        
                        # Get probability if available
                        try:
                            proba = st.session_state.current_model.predict_proba(X_new)
                            confidence = max(proba[0])
                        except:
                            confidence = 0.85  # Default confidence
                        
                        time.sleep(0.5)  # Small delay for effect
                    
                    with col2:
                        st.markdown("""
                        <div class="glass-card">
                            <h4 style="color: white; margin-bottom: 1rem;">🎯 Result</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        display_sentiment_result(prediction, confidence)
                
                elif predict_button and not user_review.strip():
                    st.error("Please enter a review to analyze.")
            
            # Quick Examples
            st.markdown("---")
            st.markdown("#### 💡 Try These Examples")
            st.markdown("<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>Click an example to auto-fill the review box above</p>", unsafe_allow_html=True)
            
            example_cols = st.columns(3)
            
            examples = [
                ("positive_ex", "✨ Positive", "This product is absolutely wonderful! Fast shipping and excellent quality. Highly recommend!", "#00ff88"),
                ("negative_ex", "😔 Negative", "Terrible quality and waste of money. Product broke after one day. Very disappointed.", "#ff6b6b"),
                ("neutral_ex", "🤔 Neutral", "The product is okay. Nothing special but does the job. Delivery was on time.", "#ffaa00")
            ]
            
            for col, (key, title, example_text, color) in zip(example_cols, examples):
                with col:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid {color}; border-radius: 12px; padding: 0.8rem; margin-bottom: 0.5rem;">
                        <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">{title}</p>
                        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0;">{example_text[:50]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Use {title}", key=key, use_container_width=True):
                        # Store the example in session state
                        st.session_state.example_text = example_text
                        st.rerun()
    
    # ===== TAB 2: ANALYTICS =====
    with tab2:
        st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        if not st.session_state.is_trained:
            st.warning("⚠️ Please train a model first to see analytics.")
        else:
            # Metrics Cards
            st.markdown("#### 📈 Performance Metrics")
            display_metrics_cards(st.session_state.metrics)
            
            st.markdown("---")
            
            # Charts Row
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Confusion Matrix")
                cm_fig = st.session_state.evaluator.plot_confusion_matrix(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    title="Confusion Matrix"
                )
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Metrics Overview")
                bar_fig = st.session_state.evaluator.plot_metrics_bar_chart(
                    st.session_state.metrics,
                    title="Model Performance"
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            
            st.markdown("---")
            
            # Additional Charts Row
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 🥧 Sentiment Distribution")
                dist_fig = st.session_state.evaluator.plot_sentiment_distribution(
                    st.session_state.y_test
                )
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col4:
                st.markdown("#### 🎯 Accuracy Gauge")
                gauge_fig = st.session_state.evaluator.plot_accuracy_gauge(
                    st.session_state.metrics['accuracy'],
                    title="Model Accuracy"
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Classification Report
            st.markdown("---")
            st.markdown("#### 📋 Detailed Classification Report")
            
            with st.expander("View Full Report"):
                report = st.session_state.evaluator.get_classification_report(
                    st.session_state.y_test,
                    st.session_state.y_pred
                )
                st.code(report, language='text')
    
    # ===== TAB 3: MODEL COMPARISON =====
    with tab3:
        st.markdown('<h2 class="section-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
        
        if len(st.session_state.trained_models) == 0:
            st.warning("⚠️ No models trained yet. Train multiple models to compare them.")
            st.info("💡 Tip: Train different models using the sidebar to see how they compare!")
        else:
            # Show comparison chart
            st.markdown("#### 🏆 Trained Models Performance")
            
            # Prepare metrics for comparison
            all_metrics = [data['metrics'] for data in st.session_state.trained_models.values()]
            
            comparison_fig = st.session_state.evaluator.plot_model_comparison(all_metrics)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Best model highlight
            if all_metrics:
                best = max(all_metrics, key=lambda x: x.get('f1_score', 0))
                st.success(f"🏆 Best Model: **{best.get('model_name')}** with F1 Score: {best.get('f1_score')*100:.1f}%")
            
            # Detailed comparison table
            st.markdown("---")
            st.markdown("#### 📋 Detailed Comparison")
            
            comparison_data = []
            for model_key, data in st.session_state.trained_models.items():
                m = data['metrics']
                comparison_data.append({
                    'Model': m.get('model_name', model_key),
                    'Accuracy': f"{m.get('accuracy', 0)*100:.2f}%",
                    'Precision': f"{m.get('precision', 0)*100:.2f}%",
                    'Recall': f"{m.get('recall', 0)*100:.2f}%",
                    'F1 Score': f"{m.get('f1_score', 0)*100:.2f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 4: ABOUT =====
    with tab4:
        st.markdown('<h2 class="section-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">🎯 Project Overview</h4>
                <p style="color: rgba(255,255,255,0.8);">
                    This Customer Review Sentiment Analyzer is an Object-Oriented 
                    Machine Learning system that demonstrates OOP principles while 
                    performing real sentiment analysis on customer reviews.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">🔧 OOP Principles Used</h4>
                <ul style="color: rgba(255,255,255,0.8);">
                    <li><b>Encapsulation:</b> Data hidden in classes with private attributes</li>
                    <li><b>Inheritance:</b> ML models inherit from BaseModel</li>
                    <li><b>Polymorphism:</b> Same interface for different algorithms</li>
                    <li><b>Abstraction:</b> Abstract base class defines contracts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">📦 Class Structure</h4>
                <ul style="color: rgba(255,255,255,0.8);">
                    <li><b>DataLoader:</b> Handles all data preprocessing</li>
                    <li><b>BaseModel:</b> Abstract class for ML models</li>
                    <li><b>SVMModel, RandomForestModel, etc.:</b> Concrete implementations</li>
                    <li><b>Evaluator:</b> Metrics and visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">👨‍💻 Author</h4>
                <p style="color: rgba(255,255,255,0.8);">
                    <b>Name:</b> Muhammad Ahmad<br>
                    <b>Course:</b> Object-Oriented Programming<br>
                    <b>Semester:</b> BS CS - 2nd Semester<br>
                    <b>Date:</b> January 2026
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tech Stack
        st.markdown("---")
        st.markdown("#### 🛠️ Technology Stack")
        
        tech_cols = st.columns(5)
        
        techs = [
            ("🐍", "Python 3.x"),
            ("📊", "Scikit-learn"),
            ("🎨", "Streamlit"),
            ("📈", "Plotly"),
            ("📝", "NLTK")
        ]
        
        for col, (icon, name) in zip(tech_cols, techs):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="color: white; margin-top: 0.5rem;">{name}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
