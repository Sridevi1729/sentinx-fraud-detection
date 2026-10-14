from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np

from ..features.extractor import (
    BehavioralFeatureExtractor,
    AccountBehaviorInput,
    ExtractedFeatures
)
from ..models.fraud_model import get_model, CLASSES
from ..data.synthetic_generator import get_sample_profiles, generate_synthetic_feature_dataset

router = APIRouter(prefix="/api", tags=["fraud-detection"])


class SimulationInput(BaseModel):
    burstiness_coefficient: float = Field(ge=-1.0, le=1.0, default=0.0)
    circadian_entropy_norm: float = Field(ge=0.0, le=1.0, default=0.7)
    night_activity_ratio: float = Field(ge=0.0, le=1.0, default=0.1)
    url_density: float = Field(ge=0.0, le=1.0, default=0.1)
    urgency_scam_score: float = Field(ge=0.0, le=1.0, default=0.0)
    follower_ratio: float = Field(ge=0.0, le=1.0, default=0.5)
    following_to_follower_ratio: float = Field(ge=0.0, le=50.0, default=1.0)
    repetition_similarity: float = Field(ge=0.0, le=1.0, default=0.05)
    retweet_ratio: float = Field(ge=0.0, le=1.0, default=0.2)
    account_age_days: int = Field(ge=1, le=5000, default=365)
    activity_velocity: float = Field(ge=0.0, le=100.0, default=2.0)
    username_digit_count: int = Field(ge=0, le=15, default=0)
    has_default_avatar: int = Field(ge=0, le=1, default=0)
    has_empty_bio: int = Field(ge=0, le=1, default=0)
    is_verified: int = Field(ge=0, le=1, default=0)


@router.post("/predict")
def predict_account_risk(account: AccountBehaviorInput) -> Dict[str, Any]:
    """Extract behavioral features from raw account activity and return fraud risk score and XAI explanations."""
    try:
        model = get_model()
        features = BehavioralFeatureExtractor.extract_from_account(account)
        result = model.predict(features)
        result["account_summary"] = {
            "username": account.username,
            "account_age_days": account.account_age_days,
            "followers": account.followers_count,
            "following": account.following_count,
            "total_posts": account.total_posts_count,
            "is_verified": account.is_verified,
            "recent_posts_analyzed": len(account.recent_posts)
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.post("/simulate")
def simulate_behavior(sim: SimulationInput) -> Dict[str, Any]:
    """Real-time parameter simulation endpoint for what-if behavioral tuning."""
    try:
        model = get_model()
        
        # Build features from slider values with sane correlated defaults for unexposed variables
        features = ExtractedFeatures(
            burstiness_coefficient=sim.burstiness_coefficient,
            circadian_entropy=sim.circadian_entropy_norm * 4.585,
            circadian_entropy_norm=sim.circadian_entropy_norm,
            night_activity_ratio=sim.night_activity_ratio,
            inter_arrival_mean_seconds=max(60.0, 86400.0 / max(0.5, sim.activity_velocity)),
            inter_arrival_cv=1.0 + abs(sim.burstiness_coefficient) * 0.8,
            peak_posts_per_hour=max(1, int(sim.activity_velocity * 0.4)),
            follower_ratio=sim.follower_ratio,
            following_to_follower_ratio=sim.following_to_follower_ratio,
            mentions_per_post=0.5 + sim.url_density * 0.8,
            hashtag_density=0.8 + sim.url_density * 2.0,
            url_density=sim.url_density,
            retweet_ratio=sim.retweet_ratio,
            reciprocal_rate_est=max(0.05, 1.0 - (sim.following_to_follower_ratio / 10.0)),
            repetition_similarity=sim.repetition_similarity,
            lexical_diversity=max(0.15, 1.0 - sim.repetition_similarity * 0.8),
            shannon_text_entropy=4.2 - sim.repetition_similarity * 1.5,
            urgency_scam_score=sim.urgency_scam_score,
            avg_post_length=110.0,
            all_caps_ratio=sim.urgency_scam_score * 0.4,
            account_age_days=sim.account_age_days,
            activity_velocity=sim.activity_velocity,
            username_digit_count=sim.username_digit_count,
            username_entropy=2.8 + (sim.username_digit_count * 0.15),
            has_default_avatar=sim.has_default_avatar,
            has_empty_bio=sim.has_empty_bio,
            is_verified=sim.is_verified
        )

        return model.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/samples")
def get_benchmark_samples() -> List[Dict[str, Any]]:
    """Return realistic test profiles for 1-click inspection."""
    return get_sample_profiles()


@router.get("/metrics")
def get_model_evaluation_metrics() -> Dict[str, Any]:
    """Return model test accuracy, F1, confusion matrix, and feature importances."""
    model = get_model()
    return model.evaluation_metrics


@router.get("/batch-scan")
def get_sample_batch_scan() -> Dict[str, Any]:
    """Return batch analysis of 24 representative social accounts."""
    model = get_model()
    
    # Generate a small sample set with known labels
    X, y = generate_synthetic_feature_dataset(n_samples=24, random_seed=99)
    results = []

    fraud_counts = {"Legitimate": 0, "Suspicious": 0, "High-Risk Fraud": 0}

    for i in range(len(y)):
        vec = X[i]
        true_class_idx = int(y[i])
        probs = model.supervised_model.predict_proba([vec])[0]
        pred_class_idx = int(np.argmax(probs))
        
        fraud_prob = 1.0 - probs[0]
        risk_score = round(float(fraud_prob * 100), 1)

        if risk_score < 30.0:
            tier = "Legitimate"
        elif risk_score < 70.0:
            tier = "Suspicious"
        else:
            tier = "High-Risk Fraud"

        fraud_counts[tier] += 1

        results.append({
            "id": f"ACC-{1000 + i}",
            "handle": f"@user_{1000 + i}",
            "true_label": CLASSES[true_class_idx]["label"],
            "predicted_label": CLASSES[pred_class_idx]["label"],
            "risk_score": risk_score,
            "risk_tier": tier,
            "url_density": round(float(vec[9]), 2),
            "burstiness": round(float(vec[0]), 2),
            "scam_score": round(float(vec[15]), 2),
            "age_days": int(vec[18])
        })

    return {
        "total_analyzed": len(results),
        "summary": fraud_counts,
        "accounts": results
    }
