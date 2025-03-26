import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from feature_engineering import preprocess_input
from model_utils import load_model_components, safe_encode_categorical, plot_feature_importance

# Configure constants
PROPERTY_TYPES = [
    "Apartment", "House", "Condominium", "Loft", "Townhouse", 
    "Hostel", "Guesthouse", "Bed & Breakfast", "Bungalow", "Villa"
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]
BED_TYPES = ["Real Bed", "Pull-out Sofa", "Futon", "Couch", "Airbed"]

def initialize_session_state():
    """Initialize session state variables if not already set."""
    if 'model_loaded' not in st.session_state:
        st.session_state.model, st.session_state.scaler, \
        st.session_state.encoders, st.session_state.model_loaded = load_model_components()

def create_prediction_form():
    """Create the main prediction input form."""
    with st.form("prediction_form"):
        # Location Details
        st.subheader("Location Details")
        zipcode = st.text_input("Zipcode", value="10001", help="Enter the postal code for your property location.")
        latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
        longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")

        st.markdown("---")

        # Property Details
        st.subheader("Property Details")
        property_type = st.selectbox("Property Type", options=PROPERTY_TYPES)
        room_type = st.selectbox("Room Type", options=ROOM_TYPES)
        bed_type = st.selectbox("Bed Type", options=BED_TYPES)

        st.markdown("---")

        # Accommodation Details
        st.subheader("Accommodation Details")
        accommodates = st.slider("Accommodates", min_value=1, max_value=16, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
        beds = st.number_input("Beds", min_value=0, max_value=16, value=1)

        st.markdown("---")

        # Availability & Reviews
        st.subheader("Availability & Reviews")
        availability_365 = st.slider("Days Available Per Year", min_value=0, max_value=365, value=300)
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
        review_scores_rating = st.slider("Overall Rating", min_value=0.0, max_value=100.0, value=90.0)
        review_scores_cleanliness = st.slider("Cleanliness Rating", min_value=0.0, max_value=10.0, value=9.0)
        review_scores_location = st.slider("Location Rating", min_value=0.0, max_value=10.0, value=9.0)
        review_scores_value = st.slider("Value Rating", min_value=0.0, max_value=10.0, value=9.0)

        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("Predict Price", use_container_width=True)

        return submitted, {
            "zipcode": zipcode,
            "latitude": latitude,
            "longitude": longitude,
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bathrooms": bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "bed_type": bed_type,
            "availability_365": availability_365,
            "number_of_reviews": number_of_reviews,
            "review_scores_rating": review_scores_rating,
            "review_scores_cleanliness": review_scores_cleanliness,
            "review_scores_location": review_scores_location,
            "review_scores_value": review_scores_value
        }

def preprocess_features(features_dict):
    """Preprocess and engineer features for prediction."""
    # Handle missing values with reasonable defaults
    features_dict['beds'] = features_dict.get('beds') or features_dict['accommodates']
    features_dict['bedrooms'] = features_dict.get('bedrooms') or max(1, features_dict['beds'] // 2)

    # Create engineered features
    features_dict['has_reviews'] = int(features_dict['number_of_reviews'] > 0)
    features_dict['total_sleeping_capacity'] = features_dict['beds'] * 2
    features_dict['location'] = features_dict['zipcode']
    features_dict['distance_to_center'] = 0.0
    features_dict['price_per_person'] = 1.0

    return features_dict

def main():
    # Configure the app
    st.set_page_config(
        page_title="Airbnb Price Predictor",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Main UI
    st.title("Airbnb Price Predictor")
    st.write("Enter your property details to get an estimated price.")

    # Create prediction form
    submitted, features_dict = create_prediction_form()

    # Handle prediction
    if submitted:
        if not st.session_state.model_loaded:
            st.error("❌ Model files not found. Please make sure the model files are in the app directory.")
        else:
            try:
                # Preprocess features
                processed_features = preprocess_features(features_dict)
                
                # Create input DataFrame
                input_df = preprocess_input(processed_features)
                
                # Safely encode categorical variables
                input_df = safe_encode_categorical(input_df, st.session_state.encoders)

                # Define required columns
                required_columns = [
                    'zipcode', 'latitude', 'longitude', 'property_type', 'room_type', 
                    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 
                    'availability_365', 'number_of_reviews', 'review_scores_rating', 
                    'review_scores_cleanliness', 'review_scores_location', 
                    'review_scores_value', 'price_per_person', 'total_sleeping_capacity', 
                    'has_reviews', 'location', 'distance_to_center'
                ]
                
                # Scale and predict
                input_scaled = st.session_state.scaler.transform(input_df[required_columns])
                predicted_price = st.session_state.model.predict(input_scaled)[0]
                
                # Display prediction
                st.success("Prediction Successful!")
                st.markdown(f"<h1>Predicted Price: ${predicted_price:.2f}</h1>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # Additional Information Section
    with st.expander("❓ How does the prediction work?"):
        st.write("""
        Our price prediction model uses **XGBoost**, a powerful machine learning algorithm 
        trained on thousands of Airbnb listings to identify patterns that influence price.

        The model considers factors such as:
        - **Location** (zipcode, latitude, longitude)
        - **Property characteristics** (type, size, accommodations)
        - **Booking history and availability**
        - **Review scores and guest satisfaction**

        This allows us to recommend an optimal price that maximizes your revenue 
        while remaining competitive in your local market.
        """)

    # Sidebar settings
    st.sidebar.title("Settings")
    debug_mode = st.sidebar.checkbox("Show Input Data", value=False)
    
    if debug_mode:
        st.json(features_dict)

    if st.sidebar.button("Reset Form"):
        st.rerun()

if __name__ == "__main__":
    main()