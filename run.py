import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure backend directory is in sys.path
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # Support cloud host environment variable PORT (e.g. Render, Railway, Heroku)
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 65)
    print(" SentinX // Social Media Behavioral Fraud Detection System")
    print("=" * 65)
    print(f" Server running at: http://{host}:{port}")
    print(f" Local Access:     http://localhost:{port}")
    print(f" API Documentation: http://localhost:{port}/docs")
    print("=" * 65)
    
    uvicorn.run("app.main:app", host=host, port=port, reload=False, app_dir=backend_dir)
