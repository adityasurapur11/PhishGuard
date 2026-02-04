import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from urlnet_torch import URLNet
from urlnet_features import extract_features
from sklearn.model_selection import train_test_split

# ============================================================
#  CHECK DATASET FILES
# ============================================================
def load_datasets():
    print("Checking dataset folder...")

    phishing_path = "../datasets/phishing_urls.csv"
    safe_path     = "../datasets/safe_urls.csv"

    if not os.path.exists(phishing_path):
        print("❌ ERROR: phishing_urls.csv not found!")
        exit()

    if not os.path.exists(safe_path):
        print("❌ ERROR: safe_urls.csv not found!")
        exit()

    print("✔ Dataset found. Loading...")

    # CSV HAS NO HEADER → so force column index 0
    phish = pd.read_csv(phishing_path, header=None)[0].astype(str).tolist()
    safe  = pd.read_csv(safe_path,     header=None)[0].astype(str).tolist()

    print(f"Loaded phishing: {len(phish)}")
    print(f"Loaded safe: {len(safe)}")

    return phish, safe


# ============================================================
#  CONVERT URL → NUMERIC FEATURES (PyTorch FloatTensor)
# ============================================================
def encode_dataset(urls):
    print("Extracting URL features (this may take time)...")
    features = [extract_features(url) for url in urls]
    return torch.tensor(features, dtype=torch.float32)


# ============================================================
#  TRAIN MODEL
# ============================================================
def train_urlnet():

    # Load phishing + safe URLs
    phish, safe = load_datasets()
    urls = phish + safe
    labels = [1] * len(phish) + [0] * len(safe)

    # Convert to tensors
    X = encode_dataset(urls)
    y = torch.tensor(labels, dtype=torch.float32)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=64
    )

    # Model
    model = URLNet(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n========== TRAINING URLNET MODEL ==========\n")

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)

            preds = preds.view(-1)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/10, Loss = {total_loss:.4f}")

    # Accuracy test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            preds = (preds > 0.5).float()
            correct += (preds.view(-1) == batch_y).sum().item()
            total += len(batch_y)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f}")

    # Save model
    torch.save(model.state_dict(), "urlnet_model.pt")
    print("\n✔ Model saved as urlnet_model.pt")


# ============================================================
#  RUN TRAINING
# ============================================================
if __name__ == "__main__":
    train_urlnet()
