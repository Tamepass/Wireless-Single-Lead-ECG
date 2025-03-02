import wfdb
import numpy as np
import os
import glob
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ✅ Load MIT-BIH ECG Data
def load_mit_bih_data(data_dir):
    record_files = glob.glob(os.path.join(data_dir, "*.dat"))
    records = [os.path.splitext(os.path.basename(f))[0] for f in record_files]
    
    signals = []
    labels = []
    
    for record in records:
        try:
            # Read ECG signal
            signal, fields = wfdb.rdsamp(os.path.join(data_dir, record))
            annotation = wfdb.rdann(os.path.join(data_dir, record), 'atr')

            # ✅ Use only one lead (e.g., MLII)
            signals.append(signal[:, 0])

            # ✅ Assign a **single** label per ECG signal (first annotation)
            labels.append(annotation.symbol[0] if annotation.symbol else 'N')  # Default to 'N' (normal)
        
        except Exception as e:
            print(f"Error loading {record}: {e}")
            continue
    
    return signals, labels

# ✅ Preprocess ECG Data
def preprocess_data(signals, labels, max_length=1000):
    # Pad/truncate signals to fixed length
    X = np.array([np.pad(sig, (0, max(0, max_length - len(sig))), 'constant')[:max_length] for sig in signals])

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y = to_categorical(y)  # One-hot encode labels

    return X, y, label_encoder

# ✅ Load Dataset
data_dir = "C:/Users/Siama Naseem/OneDrive/Desktop/Hackathon/ECG Raw Data"
signals, labels = load_mit_bih_data(data_dir)

# ✅ Ensure Data is Loaded Properly
if not signals or not labels:
    raise ValueError("No ECG data found. Check the data directory!")

X, y, label_encoder = preprocess_data(signals, labels)

# ✅ Verify Shapes Before Splitting
print(f"Dataset Loaded: X shape {X.shape}, y shape {y.shape}")

# ✅ Split Dataset into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Reshape X for CNN (Add channel dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ✅ Build CNN Model
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# ✅ Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
