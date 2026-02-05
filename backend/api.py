from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import requests
import re

from urlnet_features import extract_features
from urlnet_torch import URLNet

# ================= CONFIG =================
GOOGLE_API_KEY = "AIzaSyDQ5ovKP8zorJQg7NUBaxea_yqtC2n6CQU"
SAFE_BROWSING_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
MODEL_PATH = "urlnet_model.pt"

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= URL NORMALIZER =================
def normalize_url(url: str) -> str:
    """
    Converts defanged phishing URLs to real URLs safely
    Example:
    hxxps://google[.]com -> https://google.com
    """
    url = url.strip()
    url = url.replace("hxxps://", "https://")
    url = url.replace("hxxp://", "http://")
    url = url.replace("[.]", ".")
    return url

# ================= LOAD MODEL =================
INPUT_DIM = extract_features("https://example.com").shape[0]
model = URLNet(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ================= GOOGLE SAFE BROWSING =================
def check_google_safe_browsing(url: str) -> bool:
    payload = {
        "client": {
            "clientId": "phishing-detector",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE"
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    try:
        response = requests.post(
            f"{SAFE_BROWSING_URL}?key={GOOGLE_API_KEY}",
            json=payload,
            timeout=5
        )
        data = response.json()
        return "matches" in data
    except Exception:
        return False

# ================= API ENDPOINT =================
@app.get("/predict_url/")
def predict_url(url: str):

    # 1️⃣ Normalize URL (VERY IMPORTANT)
    url = normalize_url(url)

    # 2️⃣ Google Safe Browsing (GLOBAL REAL DATA)
    is_dangerous = check_google_safe_browsing(url)
    if is_dangerous:
        return {
            "prediction": "Phishing",
            "confidence": 1.0,
            "source": "Google Safe Browsing"
        }

    # 3️⃣ ML Feature Extraction
    try:
        features = extract_features(url)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    except Exception:
        return {
            "prediction": "Suspicious",
            "confidence": 0.75,
            "source": "URL Parsing Error"
        }

    # 4️⃣ ML Prediction
    with torch.no_grad():
        logit = model(features_tensor)
        probability = torch.sigmoid(logit).item()

    # 5️⃣ Decision Logic (REALISTIC)
    if probability >= 0.85:
        result = "Phishing"
    elif probability >= 0.60:
        result = "Suspicious"
    else:
        result = "Safe"

    return {
        "prediction": result,
        "confidence": round(probability, 2),
        "source": "ML Model"
    }
