import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from url_features import extract_url_features

def prepare_data_for_training():
    # Load datasets
    phishing = pd.read_csv("../datasets/malicious_phishing_urls.csv", header=None, names=["url"])
    phishing["label"] = 1

    safe = pd.read_csv("../datasets/benign_safe_urls.csv", header=None, names=["url"])
    safe["label"] = 0

    # Combine both
    data = pd.concat([phishing, safe], ignore_index=True)

    # Extract features
    feature_list = []
    labels = []

    for idx, row in data.iterrows():
        url = str(row["url"])  # FIXED: convert to string
        label = row["label"]

        features = extract_url_features(url)
        feature_list.append(list(features.values()))
        labels.append(label)

    return pd.DataFrame(feature_list), labels


def train_model():
    print("Preparing data...")
    X, y = prepare_data_for_training()

    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Model trained successfully!")

    # Save the model
    joblib.dump(model, "phishing_model.pkl")
    print("Model saved as phishing_model.pkl")


if __name__ == "__main__":
    train_model()
