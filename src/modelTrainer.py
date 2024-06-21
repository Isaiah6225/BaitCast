# phishing_detection/model_trainer.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y):
    """Train a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load the model from a file."""
    return joblib.load(file_path)
