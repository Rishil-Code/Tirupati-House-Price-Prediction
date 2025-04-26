import joblib
import pandas as pd
import numpy as np

print("Debug: Script is running...")
print("Starting prediction script...")

# Load the saved model, scaler, and feature columns
try:
    model = joblib.load('models/house_price_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

print("Model, scaler, and encoder loaded successfully!")

# Define a sample input (Ensure these match actual training feature names)
sample_data = {
    'area': [5000],
    'bedRoom': [3],
    'bathroom': [2],
    'balcony': [1],
    'price_per_sqft': [1200],
    'property_type': ['Apartment'],
    'facing': ['North-East']
}

def predict_price(sample_input):
    try:
        # Create DataFrame with only the required features
        sample_input_df = pd.DataFrame([sample_input])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in sample_input_df.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Select only the features used during training
        sample_input_df = sample_input_df[feature_columns]
        
        # Encode categorical variables
        for col in sample_input_df.select_dtypes(include=['object']):
            sample_input_df[col] = pd.factorize(sample_input_df[col])[0]
        
        # Scale the features
        sample_input_scaled = scaler.transform(sample_input_df)
        
        # Make prediction
        prediction = model.predict(sample_input_scaled)[0]
        
        return {
            'price': prediction,
            'formatted_price': f'₹{prediction:,.2f}',
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }
df_input_scaled = scaler.transform(df_input_final)

print(f"Processed input shape: {df_input_scaled.shape}")

# Make prediction
try:
    prediction = model.predict(df_input_scaled)
    print(f"Predicted Price: {prediction[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
