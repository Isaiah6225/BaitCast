# phishing_detection/feature_extractor.py

import pandas as pd

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
