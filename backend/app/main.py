import os
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
