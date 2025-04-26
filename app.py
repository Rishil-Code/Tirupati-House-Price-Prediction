from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Get the project root directory (works even after cloning)
PROJECT_ROOT = Path(__file__).resolve().parent

# Create data and models directories if they don't exist
for dir_name in ['data', 'models']:
    (PROJECT_ROOT / dir_name).mkdir(exist_ok=True)

app = Flask(__name__)

# Load the model and preprocessing objects
try:
    model = joblib.load(PROJECT_ROOT / 'models/house_price_model.joblib')
    scaler = joblib.load(PROJECT_ROOT / 'models/scaler.joblib')
    feature_columns = joblib.load(PROJECT_ROOT / 'models/feature_columns.joblib')
except FileNotFoundError:
    print("Model files not found. Please ensure the required model files are in the 'models' directory.")

# Create label encoders for categorical columns
categorical_columns = ['property_type', 'areaWithType', 'facing', 'agePossession', 'furnishDetails', 'features']
label_encoders = {}

# Read the dataset to get categorical values
try:
    df = pd.read_csv(PROJECT_ROOT / 'data/tirupati_houses.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'tirupati_houses.csv' is in the 'data' directory.")
    df = pd.DataFrame()
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    label_encoders[col] = le

# Load and save location settings
def load_location_settings():
    settings_file = PROJECT_ROOT / 'location_settings.json'
    if settings_file.exists():
        return json.loads(settings_file.read_text())
    return {}

def save_location_settings(settings):
    settings_file = PROJECT_ROOT / 'location_settings.json'
    settings_file.write_text(json.dumps(settings, indent=2))

@app.route('/')
def home():
    # Read the dataset to get location data
    try:
        df = pd.read_csv(PROJECT_ROOT / 'data/tirupati_houses.csv')
    except FileNotFoundError:
        print("Dataset not found. Please ensure 'tirupati_houses.csv' is in the 'data' directory.")
        df = pd.DataFrame()
    
    # Get location-specific details from dataset
    locations = []
    for _, row in df.iterrows():
        locations.append({
            'name': row['property_name'],
            'society': row['society'],
            'address': row['address'],
            'nearby': row['nearbyLocations'],
            'price': row['price_per_sqft']
        })
    
    return render_template('index.html', locations=locations)

@app.route('/api/location-settings/<address>', methods=['GET'])
def get_location_settings(address):
    settings = load_location_settings()
    return jsonify(settings.get(address, {
        'price_sqft': 3500,
        'price_bedroom': 100000,
        'price_bathroom': 75000,
        'price_additional': 50000,
        'price_balcony': 25000,
        'price_floor': 2
    }))

@app.route('/api/location-settings', methods=['POST'])
def update_location_settings():
    data = request.json
    settings = load_location_settings()
    settings[data['address']] = {
        'price_sqft': data['price_sqft'],
        'price_bedroom': data['price_bedroom'],
        'price_bathroom': data['price_bathroom'],
        'price_additional': data['price_additional'],
        'price_balcony': data['price_balcony'],
        'price_floor': data['price_floor']
    }
    save_location_settings(settings)
    return jsonify({'success': True})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from request
            data = request.get_json()
            
            # Create input data with default values
            input_data = {
                'property_type': 'Apartment',
                'price_per_sqft': float(data.get('price_per_sqft', 0)),
                'area': float(data.get('area', 0)),
                'areaWithType': 'Super built-up Area',
                'bedRoom': int(data.get('bedRoom', 0)),
                'bathroom': int(data.get('bathroom', 0)),
                'balcony': int(data.get('balcony', 0)),
                'additionalRoom': int(data.get('additionalRoom', 0)),
                'floorNum': int(data.get('floorNum', 0)),
                'facing': 'East',
                'agePossession': 'Ready to move',
                'furnishDetails': 'Semi-Furnished',
                'features': 'Basic Amenities',  # Using a value that exists in our dataset
                'rating': 4.0
            }
            
            # Create DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in categorical_columns:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            # Reorder columns to match training data
            df_input = df_input[feature_columns]
            
            # Scale numerical features
            df_input_scaled = scaler.transform(df_input)
            
            # Make prediction
            prediction = model.predict(df_input_scaled)[0]
            
            return jsonify({
                'price': float(prediction),
                'formatted_price': f'₹{float(prediction):,.2f}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid request method'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
