import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model_components():
    """
    Load model components with enhanced error handling and logging
    """
    try:
        model = joblib.load("best_xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        
        logger.info("Model components loaded successfully")
        return model, scaler, encoders, True
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        st.error("Model files are missing. Please check your directory.")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        st.error(f"An unexpected error occurred: {e}")
    
    return None, None, None, False

def safe_encode_categorical(input_df, encoders):
    """
    Safely encode categorical variables
    Handles unseen categories gracefully
    """
    categorical_cols = ['property_type', 'room_type', 'bed_type', 'location', 'total_sleeping_capacity']
    
    for col in categorical_cols:
        if col in encoders:
            # For total_sleeping_capacity, convert to string
            if col == 'total_sleeping_capacity':
                input_df[col] = input_df[col].astype(str)
            
            # Check if category exists in encoder
            existing_classes = set(encoders[col].classes_)
            input_value = input_df[col].iloc[0]
            
            if input_value not in existing_classes:
                logger.warning(f"Unseen category in {col}: {input_value}")
                # Default to first class or median
                input_df[col] = encoders[col].transform([encoders[col].classes_[0]])
            else:
                input_df[col] = encoders[col].transform(input_df[col])
    
    return input_df

def plot_feature_importance(model, features):
    """
    Create feature importance plot
    """
    plt.figure(figsize=(10, 6))
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(10))
    plt.title('Top 10 Features Influencing Price')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return plt