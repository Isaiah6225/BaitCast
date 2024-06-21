# phishing_detection/feature_extractor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """Load phishing data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the phishing data."""
    # Assuming the last column is the target and the rest are features
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def normalize_data(X, scaler=None):
    """Normalize the feature data."""
    if scaler is None:
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = scaler.transform(X)
    return X_normalized, scaler

def save_scaler(scaler, file_path):
    """Save the scaler to a file."""
    joblib.dump(scaler, file_path)

def load_scaler(file_path):
    """Load the scaler from a file."""
    return joblib.load(file_path)
