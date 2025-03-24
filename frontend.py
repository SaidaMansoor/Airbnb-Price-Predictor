import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configure the app
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)
         
# Property types from Airbnb data
PROPERTY_TYPES = [
    "Apartment", "House", "Condominium", "Loft", "Townhouse", 
    "Hostel", "Guesthouse", "Bed & Breakfast", "Bungalow", "Villa"
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]
BED_TYPES = ["Real Bed", "Pull-out Sofa", "Futon", "Couch", "Airbed"]

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("best_xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, scaler, encoders, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

model, scaler, encoders, model_loaded = load_model_components()

# Main UI
st.title(" Airbnb Price Predictor")
st.write("Enter your property details to get an estimated price.")

# Form layout
with st.form("prediction_form"):
    # Location Details
    st.subheader(" Location Details")
    zipcode = st.text_input("Zipcode", value="10001", help="Enter the postal code for your property location.")
    latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")

    st.markdown("---")

    # Property Details
    st.subheader(" Property Details")
    property_type = st.selectbox("Property Type", options=PROPERTY_TYPES)
    room_type = st.selectbox("Room Type", options=ROOM_TYPES)
    bed_type = st.selectbox("Bed Type", options=BED_TYPES)

    st.markdown("---")

    # Accommodation Details
    st.subheader(" Accommodation Details")
    accommodates = st.slider("Accommodates", min_value=1, max_value=16, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
    beds = st.number_input("Beds", min_value=0, max_value=16, value=1)

    st.markdown("---")

    # Availability & Reviews
    st.subheader(" Availability & Reviews")
    availability_365 = st.slider("Days Available Per Year", min_value=0, max_value=365, value=300)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
    review_scores_rating = st.slider("Overall Rating", min_value=0.0, max_value=100.0, value=90.0)
    review_scores_cleanliness = st.slider("Cleanliness Rating", min_value=0.0, max_value=10.0, value=9.0)
    review_scores_location = st.slider("Location Rating", min_value=0.0, max_value=10.0, value=9.0)
    review_scores_value = st.slider("Value Rating", min_value=0.0, max_value=10.0, value=9.0)

    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Handle prediction
if submitted:
    if not model_loaded:
        st.error("❌ Model files not found. Please make sure the model files are in the app directory.")
    else:
        try:
            with st.spinner(" Calculating optimal price..."):
                # Create features dictionary
                features_dict = {
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
                
                # Show request data if debug is enabled
                if st.sidebar.checkbox(" Show Request Data"):
                    st.json(features_dict)
                
                # Handle missing values with reasonable defaults
                features_dict['beds'] = features_dict.get('beds') or features_dict['accommodates']
                features_dict['bedrooms'] = features_dict.get('bedrooms') or max(1, features_dict['beds'] // 2)
                
                # Create engineered features
                features_dict['has_reviews'] = int(features_dict['number_of_reviews'] > 0)
                features_dict['total_sleeping_capacity'] = features_dict['beds'] * 2
                features_dict['location'] = features_dict['zipcode']  # Using zipcode as location
                features_dict['distance_to_center'] = 0.0  # Would calculate from lat/long in real implementation
                features_dict['price_per_person'] = 1.0  # Placeholder, will be replaced by prediction
                
                # Create DataFrame with all required columns
                required_columns = ['zipcode', 'latitude', 'longitude', 'property_type', 'room_type', 
                                'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 
                                'availability_365', 'number_of_reviews', 'review_scores_rating', 
                                'review_scores_cleanliness', 'review_scores_location', 
                                'review_scores_value', 'price_per_person', 'total_sleeping_capacity', 
                                'has_reviews', 'location', 'distance_to_center']
                
                input_df = pd.DataFrame([features_dict])
                
                # Encode categorical variables safely
                categorical_cols = ['property_type', 'room_type', 'bed_type', 'location', 'total_sleeping_capacity']
                for col in categorical_cols:
                    if col in encoders:
                        # Convert total_sleeping_capacity to string if it's in encoders (since it was originally numerical)
                        if col == 'total_sleeping_capacity':
                            input_df[col] = input_df[col].astype(str)
                        
                        # Handle unseen categories
                        if input_df[col].iloc[0] in encoders[col].classes_:
                            input_df[col] = encoders[col].transform(input_df[col])
                        else:
                            # For unseen categories, use the most common category in training
                            input_df[col] = 0  # Default to first class
                
                # Ensure all columns match the training data format
                for col in required_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0  # Default value for missing columns
                
                # Apply scaling
                input_scaled = scaler.transform(input_df[required_columns])
                
                # Make prediction
                predicted_price = model.predict(input_scaled)[0]
                
                # Round to 2 decimal places
                predicted_price = round(float(predicted_price), 2)
                
                # Display price prominently
                st.success(" Prediction Successful!")
                st.markdown(f"<h1 style='text-align: center;'> ${predicted_price:.2f}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-size: 18px;'>Suggested Price</p>", unsafe_allow_html=True)

            
        except Exception as e:
            st.error(f" Prediction Error: {str(e)}")

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

# Add settings in sidebar for app customization
st.sidebar.title("⚙️ Settings")