from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from urlnet_torch import URLNet
from urlnet_features import extract_features
import numpy as np

app = FastAPI()

# CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = URLNet(input_dim=11)
model.load_state_dict(torch.load("urlnet_model.pt", map_location=torch.device("cpu")))
model.eval()

@app.get("/")
def root():
    return {"message": "Phishing Detector API is running!"}

@app.get("/predict_url/")
def predict_url(url: str):

    try:
        features = extract_features(url)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(x).item()

        prediction = "Phishing" if output >= 0.5 else "Safe"

        return {
            "prediction": prediction,
            "confidence": float(output)
        }

    except Exception as e:
        return {
            "error": str(e),
            "prediction": None,
            "confidence": None
        }
