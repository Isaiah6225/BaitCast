# phishing_detection/main.py

from featureExtractor import load_data, preprocess_data
from modelTrainer import train_model, save_model

def main():
    data = load_data('../data/phishing_data.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, '../phishing_model.pkl')

if __name__ == "__main__":
    main()
