import numpy as np
import pandas as pd

def create_engineered_features(df):
    """
    Centralized feature engineering pipeline
    Matches exactly the process used during model training
    """
    # Ensure all required columns exist
    required_columns = [
        'price', 'accommodates', 'beds', 'number_of_reviews', 
        'zipcode', 'latitude', 'longitude'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default value if missing
    
    # Price per person (handling division by zero)
    df['price_per_person'] = df['price'] / (df['accommodates'] + 1)
    
    # Total sleeping capacity
    df['total_sleeping_capacity'] = df['beds'] * 2
    
    # Reviews activity flag
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    
    # Location feature
    df['location'] = df['zipcode'].astype(str)
    
    # Distance to center calculation (with error handling)
    try:
        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - df['latitude'].mean())**2 +
            (df['longitude'] - df['longitude'].mean())**2
        )
    except Exception:
        df['distance_to_center'] = 0.0
    
    return df

def preprocess_input(input_dict):
    """
    Convert input dictionary to preprocessed DataFrame
    Applies same transformations as training pipeline
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Apply feature engineering
    input_df = create_engineered_features(input_df)
    
    return input_df