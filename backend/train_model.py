import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from urlnet_torch import URLNet
from urlnet_features import extract_features
from sklearn.model_selection import train_test_split
import numpy as np

def train_3class():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
    
    phishing_path = os.path.join(DATASETS_DIR, "phishing_urls.csv")
    safe_path     = os.path.join(DATASETS_DIR, "safe_urls.csv")
    suspicious_path = os.path.join(DATASETS_DIR, "suspicious_urls.csv")

    print(f"Loading massive dataset from {DATASETS_DIR}...")
    
    # Load
    phish = pd.read_csv(phishing_path, header=None)[0].astype(str).tolist()
    safe  = pd.read_csv(safe_path,     header=None)[0].astype(str).tolist()
    susp  = pd.read_csv(suspicious_path, header=None)[0].astype(str).tolist()

    urls = safe + phish + susp
    labels = [0]*len(safe) + [1]*len(phish) + [2]*len(susp)

    print(f"Total dataset: {len(urls)} URLs")

    # Encode
    print("Extracting features...")
    X_list = []
    total = len(urls)
    for i, u in enumerate(urls):
        X_list.append(extract_features(u))
        if (i+1) % 20000 == 0:
            print(f"   Processed {i+1}/{total}...")
            
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

    model = URLNet(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training for 100 epochs on {len(X_train)} samples...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Log every 10 epochs or the first few to show progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1}/100, Avg Loss: {total_loss/len(train_loader):.4f}")

    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct/len(y_test)
    print(f"Final Accuracy after 100 epochs: {accuracy:.4f}")
    
    model_save_path = os.path.join(BASE_DIR, "urlnet_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"✔ 100-Epoch Model saved to {model_save_path}")

if __name__ == "__main__":
    train_3class()
