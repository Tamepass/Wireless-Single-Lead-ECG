import numpy as np
import wfdb
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_ecg_data(record_path, num_samples=1000):
    records = ['100', '101', '102', '103', '104']  # Sample records from MIT-BIH dataset
    signals, labels = [], []

    for rec in records:
        try:
            record_file = os.path.join(record_path, rec)
            record = wfdb.rdrecord(record_file)
            annotation = wfdb.rdann(record_file, 'atr')
            
            signal = record.p_signal[:, 0]  # Use only one lead
            ann_labels = annotation.symbol
            ann_locations = annotation.sample  # ECG beat locations

            for i in range(len(ann_labels)):
                if ann_locations[i] + num_samples < len(signal):
                    segment = signal[ann_locations[i]: ann_locations[i] + num_samples]
                    signals.append(segment)
                    labels.append(ann_labels[i])
        except Exception as e:
            print(f"Error loading record {rec}: {e}")

    return np.array(signals), np.array(labels)

# Load dataset
record_path = "C:/Users/Siama Naseem/OneDrive/Desktop/Hackathon/ECG Raw Data"
X, y = load_ecg_data(record_path)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalize data
X = (X - np.mean(X)) / np.std(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build CNN Model
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

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Function to generate synthetic ECG signal
def generate_synthetic_ecg(num_samples=1000):
    t = np.linspace(0, 1, num_samples)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.25 * np.random.randn(num_samples)
    return signal

# Simulated real-time ECG classification
for i in range(5):  # Generate 5 sample signals
    ecg_signal = generate_synthetic_ecg()
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)  # Normalize
    ecg_signal = ecg_signal.reshape(1, len(ecg_signal), 1)  # Reshape for CNN
    
    prediction = model.predict(ecg_signal)
    predicted_class = np.argmax(prediction)
    detected_class = encoder.inverse_transform([predicted_class])[0]
    print(f"Detected Heartbeat Class: {detected_class}")
    
    if i % 2 == 0:  # Reduce graph frequency by plotting every 2nd iteration
        plt.figure(figsize=(10, 4))
        plt.plot(ecg_signal[0], label=f'Predicted: {detected_class}')
        plt.title('Simulated ECG Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
