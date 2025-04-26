import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
import os

# -------------------- 📂 Load Dataset --------------------
print("\U0001f4c2 📂 Loading Tirupati dataset...\n")

try:
    df = pd.read_csv('data/tirupati_houses.csv')
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Display dataset info
print("\✅ Dataset Loaded Successfully!")
print(df.info())
print("\n📌 Columns in dataset:", df.columns)

# -------------------- 🏢 Data Preprocessing --------------------
print("\U0001f4c8 Data Preprocessing...\n")

# Fill missing values
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

# Select relevant features for prediction
feature_columns = [
    'property_type', 'price_per_sqft', 'area', 'areaWithType',
    'bedRoom', 'bathroom', 'balcony', 'additionalRoom', 'floorNum',
    'facing', 'agePossession', 'furnishDetails', 'features', 'rating'
]

# Split features and target
X = df[feature_columns]
y = df['price']

# Encode categorical variables
le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = le.fit_transform(X[col])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\u2705 Data Preprocessing Complete!")

# -------------------- 🏠 Train Model --------------------
print("🚀 Training Model...\n")
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

print("\u2705 Model Trained Successfully!")

# -------------------- 📊 Model Evaluation --------------------
# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = model.score(X_test_scaled, y_test)

print("\n📊 Model Evaluation:")
print(f"🔹 MAE: {mae}")
print(f"🔹 MSE: {mse}")
print(f"🔹 RMSE: {rmse}")
print(f"🔹 R-Squared: {r2}")

# -------------------- 💾 Save Model & Scaler --------------------
# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model, scaler, and feature list
joblib.dump(model, 'models/house_price_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_columns, 'models/feature_columns.joblib')

print("\n✅ Model and preprocessing objects saved successfully!")
print("\n🏠 Ready to predict house prices in Tirupati!")

# -------------------- 🚀 Predict New House Price --------------------
def predict_price(sample_input):
    # Convert the input to a DataFrame
    sample_input_df = pd.DataFrame([sample_input])

    # Load OneHotEncoder
    one_hot_encoder = joblib.load('./models/one_hot_encoder.pkl')

    # Get categorical and numerical columns from training
    categorical_columns = [col for col in X.columns if col in one_hot_encoder.feature_names_in_]
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Ensure all categorical columns exist in the input
    for col in categorical_columns:
        if col not in sample_input_df.columns:
            sample_input_df[col] = "unknown"

    # Select and encode categorical features
    sample_input_categorical = sample_input_df[categorical_columns]
    sample_input_categorical_encoded = one_hot_encoder.transform(sample_input_categorical)
    sample_input_categorical_df = pd.DataFrame(sample_input_categorical_encoded, 
                                               columns=one_hot_encoder.get_feature_names_out())

    # Ensure numerical columns exist in the input
    for col in numerical_columns:
        if col not in sample_input_df.columns:
            sample_input_df[col] = 0

    # Select numerical features
    sample_input_numerical = sample_input_df[numerical_columns]
    
    # Combine numerical and categorical features
    sample_input_final = pd.concat([sample_input_numerical, sample_input_categorical_df], axis=1)
    
    # Align with training feature order
    sample_input_final = sample_input_final.reindex(columns=X.columns, fill_value=0)
    
    # Standardize
    scaler = joblib.load('./models/scaler.pkl')
    sample_input_scaled = scaler.transform(sample_input_final)
    
    # Predict
    model = joblib.load('./models/house_price_model.pkl')
    predicted_price = model.predict(sample_input_scaled)[0]
    print("\n 🏡 Predicted House Price:", predicted_price)
    return predicted_price

# -------------------- 🏡 Dynamic House Price Prediction --------------------
def get_user_input():
    sample_input = {}

    try:
        # Get numerical inputs with validation
        while True:
            try:
                price = float(input("Enter price per sqft: "))
                if price <= 0:
                    raise ValueError("Price must be positive")
                sample_input["price_per_sqft"] = price
                break
            except ValueError:
                print("Please enter a valid positive number")
                
        while True:
            try:
                area = float(input("Enter area (sqft): "))
                if area <= 0:
                    raise ValueError("Area must be positive")
                sample_input["area"] = area
                break
            except ValueError:
                print("Please enter a valid positive number")

        while True:
            try:
                bedrooms = int(input("Enter number of bedrooms: "))
                if bedrooms <= 0:
                    raise ValueError("Number of bedrooms must be positive")
                sample_input["bedRoom"] = bedrooms
                break
            except ValueError:
                print("Please enter a valid positive integer")

        while True:
            try:
                bathrooms = int(input("Enter number of bathrooms: "))
                if bathrooms <= 0:
                    raise ValueError("Number of bathrooms must be positive")
                sample_input["bathroom"] = bathrooms
                break
            except ValueError:
                print("Please enter a valid positive integer")

        while True:
            try:
                floor = int(input("Enter floor number: "))
                if floor < 0:
                    raise ValueError("Floor number cannot be negative")
                sample_input["floorNum"] = floor
                break
            except ValueError:
                print("Please enter a valid non-negative integer")

        # Get categorical inputs with validation
        valid_property_types = ["Apartment", "Villa", "Independent House", "Builder Floor"]
        while True:
            prop_type = input(f"Enter property type {valid_property_types}: ").strip()
            if prop_type in valid_property_types:
                sample_input["property_type"] = prop_type
                break
            print(f"Please enter one of: {', '.join(valid_property_types)}")

        society = input("Enter society name (press Enter to skip): ").strip()
        sample_input["society"] = society if society else "Unknown"

        return sample_input

    except KeyboardInterrupt:
        print("\nInput cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error getting user input: {e}")
        sys.exit(1)

# Get user input dynamically
user_input = get_user_input()
predict_price(user_input)