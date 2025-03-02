import numpy as np
import wfdb
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import serial
import time

# ✅ Load MIT-BIH Arrhythmia Dataset
def load_ecg_data(record_path, num_samples=1000):
    records = ['100', '101', '102', '103', '104']  # Sample records from MIT-BIH dataset
    signals, labels = [], []

    for rec in records:
        try:
            # ✅ Ensure correct file paths
            record_file = os.path.join(record_path, rec)
            record = wfdb.rdrecord(record_file)
            annotation = wfdb.rdann(record_file, 'atr')

            # ✅ Use only one lead (e.g., MLII)
            signal = record.p_signal[:, 0]  
            ann_labels = annotation.symbol
            ann_locations = annotation.sample  # ECG beat locations

            for i in range(len(ann_labels)):
                if ann_locations[i] + num_samples < len(signal):
                    segment = signal[ann_locations[i]: ann_locations[i] + num_samples]
                    signals.append(segment)
                    labels.append(ann_labels[i])  # Assign corresponding label

        except Exception as e:
            print(f"Error loading record {rec}: {e}")

    return np.array(signals), np.array(labels)

# ✅ Load dataset (Ensure correct absolute path)
record_path = "C:/Users/Siama Naseem/OneDrive/Desktop/Hackathon/ECG Raw Data"
X, y = load_ecg_data(record_path)

# ✅ Encode Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ✅ Normalize Data
X = (X - np.mean(X)) / np.std(X)

# ✅ Reshape for CNN (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ✅ Split into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ Build CNN Model
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(set(y_encoded)), activation='softmax')
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Initialize Serial Communication (Fix COM Port & Timeout)
try:
    ser = serial.Serial('COM3', 9600, timeout=2)  # ✅ Added timeout to avoid hanging
    time.sleep(2)  # Allow time for connection
    print("Serial connection established.")
except serial.SerialException:
    print("⚠️ Error: Could not open serial port. Check device connection.")
    ser = None  # Avoid breaking the script

# ✅ Real-Time ECG Signal Collection
def get_ecg_signal():
    if ser is None:
        print("⚠️ No serial connection available. Skipping ECG signal collection.")
        return None

    signal = []
    while len(signal) < 1000:  # Collect 1000 samples for prediction
        try:
            value = ser.readline().decode().strip()  # Read ECG values
            if value:
                signal.append(float(value))  
        except ValueError:
            continue
    return np.array(signal)

# ✅ Real-Time ECG Classification
if ser:
    while True:
        ecg_signal = get_ecg_signal()
        
        if ecg_signal is not None and len(ecg_signal) == 1000:  # Ensure full signal window is collected
            ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)  # Normalize
            ecg_signal = ecg_signal.reshape(1, len(ecg_signal), 1)  # Reshape for CNN

            prediction = model.predict(ecg_signal)
            predicted_class = np.argmax(prediction)

            detected_class = encoder.inverse_transform([predicted_class])[0]
            print(f"Detected Heartbeat Class: {detected_class}")

            if detected_class in ["V", "A", "F"]:  # Abnormal rhythms
                print("🚨 ALERT! Abnormal Heartbeat Detected 🚨")
