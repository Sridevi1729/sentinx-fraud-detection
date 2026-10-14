import os
import pickle
import instaloader
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes.predict import router as predict_router
from .models.fraud_model import get_model

app = FastAPI(
    title="Social Media Behavioral Fraud Detection System",
    description="Multi-dimensional behavioral machine learning system for detecting social media bots, phishing, and fraud.",
    version="1.0.0"
)

# Enable CORS for local development or external UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(predict_router)

# Mount static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- INSTAGRAM DETECTION LOGIC ---

# Safely locate and load the model from the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INSTA_MODEL_PATH = os.path.join(BASE_DIR, "instagram_model.pkl")

instagram_model = None
try:
    with open(INSTA_MODEL_PATH, "rb") as file:
        instagram_model = pickle.load(file)
    print("Instagram model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load Instagram model: {e}")


def get_instagram_features(username: str):
    """Scrapes public Instagram profile metadata and formats the 11 Kaggle features."""
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        
        full_name = profile.full_name or ""
        features = [
            1 if profile.profile_pic_url else 0,
            sum(c.isdigit() for c in username) / len(username) if username else 0,
            len(full_name.split()) if full_name else 0,
            sum(c.isdigit() for c in full_name) / len(full_name) if full_name else 0,
            1 if full_name == username else 0,
            len(profile.biography) if profile.biography else 0,
            1 if profile.external_url else 0,
            1 if profile.is_private else 0,
            profile.mediacount,
            profile.followers,
            profile.followees
        ]
        return features
    except Exception:
        return None


@app.get("/detect/instagram/{username}")
def detect_instagram(username: str):
    """API endpoint to analyze an Instagram profile."""
    if not instagram_model:
        return {"error": "Instagram model is not loaded on the server."}
        
    features = get_instagram_features(username)
    if not features:
        return {"error": "Could not fetch profile data. The account may not exist or is fully restricted."}
        
    prediction = instagram_model.predict([features])[0]
    probability = instagram_model.predict_proba([features])[0].max() * 100
    
    result = "Fake Account" if prediction == 1 else "Genuine Account"
    return {
        "username": username,
        "platform": "Instagram",
        "prediction": result,
        "confidence": f"{probability:.2f}%"
    }

# --- END INSTAGRAM LOGIC ---


@app.on_event("startup")
def startup_event():
    """Ensure the machine learning model is trained and initialized upon startup."""
    print("Initializing Behavioral Fraud Detection Machine Learning Pipeline...")
    model = get_model()
    print(f"Model ready! Test Accuracy: {model.evaluation_metrics.get('accuracy', 'N/A')}")


@app.get("/")
def serve_index():
    """Serve the interactive web application dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Social Media Behavioral Fraud Detection API is running. Access /docs for API documentation."}


@app.get("/api/health")
def healthcheck():
    return {"status": "healthy", "service": "social-fraud-detection"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)