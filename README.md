# 🛡️ ShieldScan AI: Phishing URL Detector

ShieldScan AI is a high-performance, multi-layered phishing detection system that combines **Traditional Heuristics**, **Verified Domain Databases**, and **Deep Learning (URLNet)** to identify malicious links in real-time.

---

## 🚀 Key Features

- **Triple-Layer Detection Engine**:
  1. **Instant Trust Bypass**: Cross-references URLs against a database of 100k+ safe domains (Google, Amazon, etc.).
  2. **Heuristic Analysis**: Detects brand spoofing, suspicious TLDs (`.zip`, `.tk`), IP-based domains, and urgency keywords.
  3. **AI Pattern Analysis**: Utilizes a custom PyTorch-based Neural Network (URLNet) to analyze structural patterns in unknown URLs.
- **Modern UI**: A responsive, glassmorphism-inspired web interface for easy URL analysis.
- **High Performance**: Built with FastAPI for low-latency API responses.

---

## 🏗️ Project Architecture

```text
PhishingDetectorProject/
├── backend/                # FastAPI Application & AI Logic
│   ├── api.py              # Main API entry point
│   ├── urlnet_torch.py     # PyTorch Model Architecture
│   ├── urlnet_features.py  # Feature Engineering (15 unique features)
│   ├── train_urlnet.py     # Model Training Script
│   └── urlnet_model.pt     # Pre-trained Model Weights
├── frontend/               # User Interface
│   └── index.html          # Dashboard (Vanilla JS + CSS)
├── datasets/               # CSV Data for Training/Validation
│   ├── phishing_urls.csv
│   └── safe_urls.csv
└── requirements.txt        # Python Dependencies
```

---

## 🧠 AI Model: URLNet

The system uses a Deep Neural Network optimized for URL structural analysis. It extracts **15 key features** from every URL, including:
- URL, Domain, and Path lengths.
- Special character frequency (`-`, `.`, `@`, `?`, etc.).
- Protocol verification (HTTPS vs HTTP).
- Detection of URL shorteners (Bitly, TinyURL).
- Subdomain trickery and file extension analysis.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd PhishingDetectorProject
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3. Start the Backend Server
```bash
cd backend
python -m uvicorn api:app --reload
```
The API will be running at `http://127.0.0.1:8000`.

### 4. Launch the Frontend
Simply open `frontend/index.html` in any modern web browser.

---

## 📊 Detection Logic Flow

1. **Normalized Check**: The URL is cleaned and parsed.
2. **Safe List Check**: If the domain is a known giant (e.g., `google.com`), it is immediately marked **Safe**.
3. **Rule-Based Engine**:
   - Is it an IP address? 🚩
   - Is it a URL shortener? 🚩
   - Does it spoof a brand (e.g., `paypal-security.com`)? 🚩
4. **AI Analysis**: If the URL passes the above checks but is unknown, the PyTorch model calculates a confidence score based on structural patterns.

---

## 🛠️ Technologies Used

- **Frontend**: HTML5, CSS3 (Flexbox/Animations), JavaScript (Fetch API).
- **Backend**: Python 3, FastAPI, Uvicorn.
- **Machine Learning**: PyTorch, Scikit-learn, Pandas, NumPy.

---

## 🛡️ Safety Warning
*ShieldScan AI is a predictive tool. While highly accurate, users should always exercise caution when clicking on unknown links.*
