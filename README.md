# SentinX // Social Media Behavioral Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An end-to-end Behavioral Machine Learning & Explainable AI (XAI) application for detecting social media fraud, botnets, impersonators, and malicious campaigns based on **dynamic human vs automated behavioral signatures**.

---

## 🌟 Key Features

1. **28-Dimension Behavioral Feature Extraction**:
   - **Temporal Dynamics**: Inter-arrival variance, burstiness coefficient ($B \in [-1, 1]$), circadian Shannon entropy ($H_t$), and 24/7 night activity ratio.
   - **Interaction Asymmetry**: Follower/following ratio, mention frequency, outbound link density, and interaction reciprocity.
   - **Linguistic Volatility**: Pairwise Jaccard template repetition, lexical diversity (TTR), text entropy, and phishing/crypto scam triggers.
   - **Digital Footprint**: Account longevity, activity velocity (posts/day), and username digit entropy.

2. **Hybrid Machine Learning Architecture**:
   - **Supervised Classifier**: Tuned Random Forest ensemble with balanced class weights for 4-class discrimination (*Legitimate User*, *Automated Spambot*, *Phishing & Financial Scam*, *Coordinated Sybil*).
   - **Unsupervised Anomaly Detector**: Isolation Forest to detect zero-day inauthentic anomalies.
   - **Explainable AI (XAI)**: Mathematical feature attribution waterfall displaying exact positive/negative risk contributors with plain-English diagnostics.

3. **Interactive Web Application (Dashboard)**:
   - **Live Account Inspector**: Scan any profile or load preloaded benchmark cases (Verified Creator, Deal Spambot, Crypto Airdrop Scammer, Sybil Drone).
   - **Radial Animated Risk Meter**: 0–100% risk score with color-coded risk tiers (Safe, Suspicious, Critical).
   - **Behavioral Radar Matrix**: 6-axis polar comparison against human baseline.
   - **24-Hour Circadian Clock**: Hourly activity breakdown showing day vs night posting distribution.
   - **Interactive "What-If" Sandbox**: Sliders for burstiness, link density, urgency lures, and account age with instant real-time risk recalculation.
   - **Model Metrics & Confusion Matrix**: Visualized test split performance and feature importances.

---

## 🚀 How to Run Locally

### Option A: Standard Python (Any OS)
```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
python run.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

---

### Option B: Docker (Zero Setup Required)
```bash
docker compose up
```
Open **[http://localhost:8080](http://localhost:8080)**.

---

## 🌐 Deploy Online (Make it Work Remotely for Everyone)

You can host this project online completely free so anyone with a smartphone, tablet, or PC can access it:

### 1. Free Cloud Hosting on Render (Recommended)
1. Push this project to your GitHub repository.
2. Sign in to **[Render.com](https://render.com)**.
3. Click **New +** $\rightarrow$ **Web Service**.
4. Connect your GitHub repository.
5. Set:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
6. Click **Deploy Web Service**. You will receive a public URL (e.g., `https://sentinx-fraud.onrender.com`) accessible from any device worldwide!

### 2. Free Cloud Hosting on Hugging Face Spaces
1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** $\rightarrow$ **Create new Space**.
2. Select **Docker** or **Gradio/FastAPI**.
3. Push or upload this repo, and your app will run instantly on Hugging Face's global cloud.

---

## 📁 Project Architecture

```
social-fraud-detection/
├── backend/
│   ├── app/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── synthetic_generator.py   # Benchmark data generator & case studies
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   └── extractor.py             # 28 behavioral metric extraction
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── fraud_model.py           # Hybrid RF + Isolation Forest + XAI
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── predict.py               # REST API endpoints
│   │   ├── __init__.py
│   │   └── main.py                      # FastAPI app entry point
│   └── static/
│       └── index.html                   # Interactive dark-mode dashboard
├── run.py                               # 1-click launcher script
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Container definition
├── docker-compose.yml                   # Docker compose config
├── render.yaml                          # 1-click cloud deploy config
└── README.md                            # Documentation
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
