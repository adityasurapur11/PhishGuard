from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import numpy as np
import requests
import re
import os
import pandas as pd
from urllib.parse import urlparse

from urlnet_features import extract_features
from urlnet_torch import URLNet

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
SAFE_CSV = os.path.join(DATASETS_DIR, "safe_urls.csv")
MODEL_PATH = os.path.join(BASE_DIR, "urlnet_model.pt")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ULTIMATE TRUST LIST (HARDCODED) =================
# This ensures that even if the 100k list has an issue, the giants never fail.
GLOBAL_TRUST_LIST = {
    'google.com', 'youtube.com', 'facebook.com', 'instagram.com', 'whatsapp.com',
    'twitter.com', 'x.com', 'linkedin.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'amazon.in', 'netflix.com', 'paypal.com', 'github.com',
    'stackoverflow.com', 'reddit.com', 'wikipedia.org', 'gmail.com', 'outlook.com',
    'yahoo.com', 'bing.com', 'adobe.com', 'zoom.us', 'spotify.com', 'medium.com'
}

# ================= LOAD SAFE DATABASE =================
print("⏳ Loading Safe URLs and preparing optimized lookup...")
SAFE_BASE_DOMAINS = set(GLOBAL_TRUST_LIST)
SAFE_FULL_DOMAINS = set()

try:
    if os.path.exists(SAFE_CSV):
        df = pd.read_csv(SAFE_CSV, header=None)
        
        def get_base_domain_robust(u):
            try:
                u_str = str(u).lower().strip()
                if not u_str.startswith("http"): u_str = "https://" + u_str
                dom = urlparse(u_str).netloc
                if dom.startswith("www."): dom = dom[4:]
                parts = dom.split('.')
                if len(parts) >= 2:
                    # Return base domain (e.g. google.com)
                    return ".".join(parts[-2:])
                return dom
            except: return ""

        base_doms = df[0].apply(get_base_domain_robust).dropna().unique()
        SAFE_BASE_DOMAINS.update(base_doms)
        print(f"✔ Loaded {len(SAFE_BASE_DOMAINS)} Trusted Base Domains")
except Exception as e:
    print(f"❌ Error loading safe DB: {e}")

# ================= HELPERS =================
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def get_domain_info(url: str):
    try:
        parsed = urlparse(url)
        full_domain = parsed.netloc.lower()
        if full_domain.startswith("www."): full_domain = full_domain[4:]
        
        parts = full_domain.split('.')
        base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else full_domain
        return full_domain, base_domain
    except: return "", ""

def check_phishing_rules(url: str, full_domain: str, base_domain: str):
    """Refined heuristic engine"""
    url_lower = url.lower()
    
    # Check if trusted
    is_trusted = base_domain in SAFE_BASE_DOMAINS or full_domain in SAFE_BASE_DOMAINS
    
    # 1. IP Address (Strict)
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", full_domain):
        return "IP Address instead of Domain"

    # 2. URL Shorteners (Strict)
    SHORTENERS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'rebrand.ly'}
    if any(s in full_domain for s in SHORTENERS):
        return "URL Shortener detected"

    # 3. Suspicious TLDs (Only for non-trusted)
    if not is_trusted:
        SUSPICIOUS_TLDS = {'.tk', '.ml', '.cf', '.ga', '.gq', '.zip', '.top'}
        if any(full_domain.endswith(t) for t in SUSPICIOUS_TLDS):
            return "Suspicious TLD detected"

    # 4. Brand Spoofing (Crucial: Detects google.com-verify.info)
    BRANDS = ['google', 'paypal', 'amazon', 'apple', 'microsoft', 'facebook', 'youtube', 'netflix']
    for brand in BRANDS:
        if brand in full_domain:
            # If the brand is present, the BASE domain MUST be the official one
            # e.g. 'google' must be on 'google.com' or 'google.co.in'
            official_bases = [f"{brand}.com", f"{brand}.in", f"{brand}.co.in", f"{brand}.net", f"{brand}.org"]
            if brand == 'youtube': official_bases.append('youtu.be')
            
            if base_domain not in official_bases:
                return f"Potential {brand} Brand Spoofing"

    # 5. Urgency Keywords (Only if not trusted)
    if not is_trusted:
        KEYWORDS = ['verify', 'urgent', 'suspend', 'update', 'confirm', 'secure', 'signin', 'identifier']
        if any(k in url_lower for k in KEYWORDS):
            # Also check if it looks like a login path but on a random domain
            if "/" in url_lower.split("://")[1]:
                return "Urgency keywords on untrusted domain"

    return None

# ================= LOAD MODEL =================
model = URLNet(input_dim=15)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

# ================= API ENDPOINT =================
@app.get("/predict_url/")
def predict_url(url: str):
    url = normalize_url(url)
    full_domain, base_domain = get_domain_info(url)

    # 1️⃣ INSTANT SAFE BYPASS (Database + Hardcoded List)
    if full_domain in SAFE_BASE_DOMAINS or base_domain in SAFE_BASE_DOMAINS:
        return {
            "prediction": "Safe",
            "confidence": 1.0,
            "source": "Verified Official Platform"
        }

    # 2️⃣ PHISHING RULES
    phish_reason = check_phishing_rules(url, full_domain, base_domain)
    if phish_reason:
        return {
            "prediction": "Phishing",
            "confidence": 1.0,
            "source": f"Heuristic: {phish_reason}"
        }

    # 3️⃣ AI PATTERN ANALYSIS (For unknown/suspicious links)
    try:
        features = extract_features(url)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(features_tensor)
            probs = F.softmax(logits, dim=1).numpy()[0]
            
            pred_class_idx = np.argmax(probs)
            mapping = {0: "Safe", 1: "Phishing", 2: "Suspicious"}
            result = mapping[pred_class_idx]
            confidence = float(probs[pred_class_idx])

            # Logic: If it's not a known brand, not in our 100k list, and model says safe,
            # we mark as 'Suspicious' because we can't verify its intent.
            if result == "Safe":
                result = "Suspicious"
                confidence = round(probs[0], 2)

            return {
                "prediction": result,
                "confidence": round(confidence, 2),
                "source": "AI Pattern Analysis"
            }
    except:
        return {"prediction": "Suspicious", "confidence": 0.5, "source": "Internal Fallback"}
