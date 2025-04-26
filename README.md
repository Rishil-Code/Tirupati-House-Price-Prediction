# Tirupati House Price Prediction

A web application that predicts house prices in Tirupati based on various features like location, property type, area, and amenities.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tirupati-house-price-prediction.git
   cd tirupati-house-price-prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure required files are present:
   - `data/tirupati_houses.csv`: Dataset file
   - `models/house_price_model.joblib`: Trained model
   - `models/scaler.joblib`: Feature scaler
   - `models/feature_columns.joblib`: Feature columns

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and visit: http://localhost:5000

## Project Structure

```
├── app.py              # Flask application
├── data/               # Dataset directory
│   └── tirupati_houses.csv
├── models/             # Model files directory
│   ├── house_price_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates
└── requirements.txt    # Python dependencies
```

## Features

- Modern, responsive UI with dark mode support
- Real-time price predictions
- Location-based price adjustments
- Support for various property types and amenities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

A machine learning-based web application for predicting house prices in Tirupati, Andhra Pradesh. The application uses real estate data from various localities in Tirupati to provide accurate price predictions based on multiple parameters.

## Project Overview

This project implements a house price prediction system with the following key features:

1. **Property Type Selection**:
   - Apartments (1200-1500 sqft)
   - Villas (3000 sqft)
   - Duplexes (1750-2000 sqft)

2. **Location-based Pricing**:
   - 26 prime locations in Tirupati
   - Dynamic price adjustments based on locality
   - Nearby amenities consideration

3. **Property Features**:
   - Area (1000-6000 sqft)
   - Bedrooms (1-5 BHK)
   - Bathrooms (1-4)
   - Balconies (0-4)
   - Additional Rooms (0-3)
   - Floor Number (varies by property type)

4. **Price Adjustment Factors**:
   - Property type premium (Villa: +20%, Duplex: +10%)
   - Floor-based adjustments
   - Room configuration impact
   - Amenities consideration

## Project Structure

```
House-Price-Prediction/
├── app.py                 # Flask application server
├── main.py                # Model training script
├── predict.py             # Prediction logic
├── requirements.txt       # Project dependencies
├── data/
│   └── tirupati_houses.csv  # Dataset with property details
├── models/
│   └── model.pkl          # Trained ML model
└── templates/
    └── index.html         # Web interface
```

## Key Components

1. **Dataset (`tirupati_houses.csv`)**:
   - Comprehensive property database
   - Real market prices from Tirupati
   - Detailed property specifications

2. **Web Interface (`index.html`)**:
   - Modern, responsive design
   - Dynamic form validation
   - Real-time price updates
   - Property type-specific options

3. **Backend (`app.py`)**:
   - Flask-based server
   - RESTful API endpoints
   - Data preprocessing
   - Model integration

4. **Model Training (`main.py`)**:
   - Data preprocessing
   - Feature engineering
   - Model selection and training
   - Performance evaluation

## Technologies Used

- **Frontend**:
  - HTML5, CSS3, JavaScript
  - Tailwind CSS for styling
  - Dynamic form handling

- **Backend**:
  - Python 3.8+
  - Flask web framework
  - Pandas for data manipulation
  - Scikit-learn for ML models

- **Machine Learning**:
  - Linear Regression
  - Feature scaling
  - Cross-validation
  - Model persistence

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/House-Price-Prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the web interface at `http://localhost:5000`

## Features and Functionality

1. **Property Selection**:
   - Choose from 26 prime locations
   - Select property type (Apartment/Villa/Duplex)
   - Specify area and room configuration

2. **Price Prediction**:
   - Real-time price estimates
   - Factor-based adjustments
   - Market trend consideration

3. **User Interface**:
   - Intuitive form layout
   - Dynamic option updates
   - Responsive design
   - Light/Dark mode toggle

## Future Enhancements

## Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost

## Keywords
House Price Prediction, Machine Learning, Real Estate Analytics, Regression Models, Data Science, Market Valuation, Predictive Modeling, Web Deployment
