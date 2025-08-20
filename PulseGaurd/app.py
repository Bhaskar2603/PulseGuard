# app.py
import os
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and features
def load_model():
    model_path = os.path.join('models', 'pulseguard_xgb_model.pkl')
    features_path = os.path.join('models', 'pulseguard_features.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    return model, features

# Get current time features
def get_time_features():
    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()  # Monday=0, Sunday=6
    month = now.month
    
    # Check if current time is rush hour
    is_rush_hour = 1 if ((7 <= hour <= 10) or (16 <= hour <= 19)) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return hour, day_of_week, month, is_weekend, is_rush_hour

# Home route
@app.route('/')
def index():
    hour, day_of_week, month, is_weekend, is_rush_hour = get_time_features()
    
    # Default values for the form
    default_values = {
        'temperature': 28.5,
        'humidity': 45.0,
        'pm10': 120.0,
        'co2': 450.0,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour
    }
    
    return render_template('index.html', **default_values)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pm10 = float(request.form['pm10'])
        co2 = float(request.form['co2'])
        hour = int(request.form['hour'])
        day_of_week = int(request.form['day_of_week'])
        month = int(request.form['month'])
        is_weekend = int(request.form['is_weekend'])
        is_rush_hour = int(request.form['is_rush_hour'])
        
        # Load model
        model, features = load_model()
        
        # Make prediction
        input_data = np.array([[temperature, humidity, pm10, co2, hour, day_of_week, month, is_weekend, is_rush_hour]])
        prediction = model.predict(input_data)[0]
        
        # Determine risk level
        if prediction < 35:
            risk_level = "Low"
            risk_color = "success"
            risk_message = "Air quality is good. Enjoy your outdoor activities!"
        elif prediction < 75:
            risk_level = "Moderate"
            risk_color = "warning"
            risk_message = "Air quality is acceptable. However, there may be a risk for sensitive groups."
        elif prediction < 115:
            risk_level = "High"
            risk_color = "danger"
            risk_message = "Health warnings of emergency conditions. The entire population is more likely to be affected."
        else:
            risk_level = "Emergency"
            risk_color = "critical"
            risk_message = "Health alert: everyone may experience more serious health effects."
        
        # Return results
        return render_template('results.html', 
                              pm2_5=round(prediction, 2),
                              risk_level=risk_level,
                              risk_color=risk_color,
                              risk_message=risk_message,
                              temperature=temperature,
                              humidity=humidity,
                              pm10=pm10,
                              co2=co2)
                              
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)