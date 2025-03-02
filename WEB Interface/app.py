import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import csv

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for frontend-backend communication

# Function to load ECG data from CSV
def load_ecg_data():
    file_path = os.path.join(os.getcwd(), 'ecg_data.csv')  # Ensure correct file path
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]
            return data  # Returns a list of ECG records
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return []

# Load ECG data initially
ecg_data = load_ecg_data()
print("ECG Data Loaded:", ecg_data)  # Debugging to check if data is loaded

@app.route('/')
def index():
    """Render the ECG Monitoring Frontend."""
    return render_template('index.html')

@app.route('/ecg-data', methods=['GET'])
def get_ecg_data():
    """Fetch ECG data for frontend visualization."""
    try:
        return jsonify(ecg_data)  # Send ECG data as JSON
    except Exception as e:
        print(f"Error fetching ECG data: {e}")
        return jsonify({"error": "Unable to retrieve ECG data"}), 500

if __name__ == '__main__':
    app.run(debug=True)
