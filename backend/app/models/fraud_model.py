import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from ..features.extractor import ExtractedFeatures
from ..data.synthetic_generator import generate_synthetic_feature_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")

CLASSES = {
    0: {
        "label": "Legitimate User",
        "category": "Organic",
        "color": "emerald",
        "risk_tier": "Low Risk",
        "desc": "Natural human circadian activity, balanced interaction reciprocity, and varied organic speech."
    },
    1: {
        "label": "Automated Spambot",
        "category": "Automated",
        "color": "amber",
        "risk_tier": "High Risk",
        "desc": "Scripted posting intervals, repetitive template structures, and aggressive link distribution."
    },
    2: {
        "label": "Phishing & Financial Scam",
        "category": "Malicious",
        "color": "rose",
        "risk_tier": "Critical Risk",
        "desc": "Fresh account age, high-urgency keywords, credential/crypto lures, and high uppercase ratios."
    },
    3: {
        "label": "Coordinated Sybil Account",
        "category": "Inauthentic",
        "color": "purple",
        "risk_tier": "High Risk",
        "desc": "Coordinated retweet/share amplification network with minimal original discourse."
    }
}


class FraudDetectionModel:
    def __init__(self):
        self.supervised_model: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.feature_names = ExtractedFeatures.feature_names()
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.evaluation_metrics: Dict[str, Any] = {}

    def train_and_evaluate(self, n_samples: int = 1600, force: bool = False) -> Dict[str, Any]:
        """Train ensemble model and compute test evaluation metrics."""
        if not force and os.path.exists(MODEL_PATH):
            try:
                saved_data = joblib.load(MODEL_PATH)
                self.supervised_model = saved_data["supervised_model"]
                self.anomaly_detector = saved_data["anomaly_detector"]
                self.baseline_stats = saved_data["baseline_stats"]
                self.evaluation_metrics = saved_data["evaluation_metrics"]
                return self.evaluation_metrics
            except Exception:
                pass  # Fall back to training

        X, y = generate_synthetic_feature_dataset(n_samples=n_samples, random_seed=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # 1. Supervised Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
        rf.fit(X_train, y_train)

        # 2. Unsupervised Anomaly Detector (Isolation Forest) on legitimate baseline
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        iso.fit(X_train[y_train == 0])

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        conf_mat = confusion_matrix(y_test, y_pred).tolist()

        # Compute baselines for feature explanation
        baseline_stats = {}
        for idx, feat_name in enumerate(self.feature_names):
            legit_vals = X_train[y_train == 0, idx]
            baseline_stats[feat_name] = {
                "mean": float(np.mean(legit_vals)),
                "std": float(np.std(legit_vals) + 1e-6),
                "importance": float(rf.feature_importances_[idx])
            }

        self.supervised_model = rf
        self.anomaly_detector = iso
        self.baseline_stats = baseline_stats
        self.evaluation_metrics = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "confusion_matrix": conf_mat,
            "classes": [CLASSES[i]["label"] for i in range(4)],
            "feature_importances": [
                {"name": name, "importance": round(float(rf.feature_importances_[i]), 4)}
                for i, name in enumerate(self.feature_names)
            ]
        }

        # Persist
        joblib.dump({
            "supervised_model": self.supervised_model,
            "anomaly_detector": self.anomaly_detector,
            "baseline_stats": self.baseline_stats,
            "evaluation_metrics": self.evaluation_metrics
        }, MODEL_PATH)

        return self.evaluation_metrics

    def predict(self, features: ExtractedFeatures) -> Dict[str, Any]:
        """Perform full behavioral inference, risk scoring, explainability, and radar generation."""
        if self.supervised_model is None:
            self.train_and_evaluate()

        vec = np.array([features.to_feature_vector()], dtype=np.float32)
        probabilities = self.supervised_model.predict_proba(vec)[0]  # [p0, p1, p2, p3]
        predicted_class_idx = int(np.argmax(probabilities))

        # Hybrid Fraud Probability: (1 - P(Legitimate)) * 100
        # If anomaly detector flags anomaly, add safety margin
        p_legit = probabilities[0]
        base_fraud_prob = (1.0 - p_legit)

        # Anomaly score from Isolation Forest: decision_function negative = outlier
        iso_score = float(self.anomaly_detector.decision_function(vec)[0])
        anomaly_flag = iso_score < -0.05
        if anomaly_flag and base_fraud_prob < 0.65:
            base_fraud_prob = min(0.95, base_fraud_prob + 0.15)

        risk_score = round(float(base_fraud_prob * 100), 1)

        # Determine Risk Tier
        if risk_score < 30.0:
            risk_level = "Legitimate (Safe)"
            risk_badge = "emerald"
        elif risk_score < 70.0:
            risk_level = "Suspicious (Moderate Risk)"
            risk_badge = "amber"
        else:
            risk_level = "High-Risk Fraud (Malicious)"
            risk_badge = "rose"

        class_info = CLASSES[predicted_class_idx]

        # Explainability: Feature Attribution & Key Risk Drivers
        explanations = self._calculate_feature_attributions(vec[0], features, predicted_class_idx)

        # 6-Axis Behavioral Radar Profile
        radar_metrics = self._calculate_radar_metrics(features)

        # Circadian hourly activity distribution mockup based on features
        circadian_curve = self._simulate_circadian_curve(features)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "predicted_class": class_info["label"],
            "category": class_info["category"],
            "class_description": class_info["desc"],
            "probabilities": {
                CLASSES[i]["label"]: round(float(probabilities[i] * 100), 1)
                for i in range(4)
            },
            "is_anomaly": anomaly_flag,
            "anomaly_score": round(iso_score, 3),
            "radar_metrics": radar_metrics,
            "circadian_curve": circadian_curve,
            "explanations": explanations,
            "raw_features": features.model_dump()
        }

    def _calculate_feature_attributions(
        self, vec: np.ndarray, features: ExtractedFeatures, pred_class: int
    ) -> List[Dict[str, Any]]:
        """Compute relative impact of features deviating from human baseline."""
        drivers = []
        human_readable_names = {
            "burstiness_coefficient": "Temporal Burstiness & Periodicity",
            "circadian_entropy_norm": "Circadian 24/7 Flat Activity",
            "night_activity_ratio": "Night-time Action Ratio",
            "inter_arrival_cv": "Inter-arrival Interval Variance",
            "peak_posts_per_hour": "Peak Hourly Posting Velocity",
            "follower_ratio": "Follower / Following Asymmetry",
            "following_to_follower_ratio": "Following Aggression Ratio",
            "mentions_per_post": "Mention & Unsolicited Tag Frequency",
            "hashtag_density": "Hashtag Keyword Stuffing",
            "url_density": "Outbound URL / Link Density",
            "retweet_ratio": "Retweet Amplification Ratio",
            "reciprocal_rate_est": "Interaction Reciprocity Rate",
            "repetition_similarity": "Pairwise Post Template Repetition",
            "lexical_diversity": "Vocabulary Diversity (TTR)",
            "shannon_text_entropy": "Text Complexity & Entropy",
            "urgency_scam_score": "Financial / Phishing Scam Keyword Triggers",
            "avg_post_length": "Average Character Count",
            "all_caps_ratio": "Uppercase Urgency / Shouting Ratio",
            "account_age_days": "Account Longevity (Days Active)",
            "activity_velocity": "Lifetime Activity Rate (Posts/Day)",
            "username_digit_count": "Numeric Digits in Handle",
            "username_entropy": "Handle Randomness Entropy",
            "has_default_avatar": "Default Placeholder Avatar",
            "has_empty_bio": "Unfilled Profile Bio",
            "is_verified": "Platform Verification Badge"
        }

        for idx, feat_name in enumerate(self.feature_names):
            val = float(vec[idx])
            stat = self.baseline_stats.get(feat_name, {"mean": 0.0, "std": 1.0, "importance": 0.04})
            mean = stat["mean"]
            std = stat["std"]
            importance = stat["importance"]

            z_score = (val - mean) / std

            # Invert polarity for protective features
            if feat_name in ["account_age_days", "reciprocal_rate_est", "lexical_diversity", "is_verified"]:
                contribution = -z_score * importance * 35.0
            elif feat_name == "burstiness_coefficient":
                # Both extreme negative (rigid timer) and extreme positive (campaign burst) are fraudulent
                contribution = (abs(val) - abs(mean)) * importance * 45.0
            else:
                contribution = z_score * importance * 35.0

            # Generate natural language diagnostic
            diagnostic = ""
            if feat_name == "burstiness_coefficient" and val < -0.4:
                diagnostic = f"Rigid clockwork timing (B = {val}) indicates automated scheduled cron job."
            elif feat_name == "burstiness_coefficient" and val > 0.4:
                diagnostic = f"Bursty flood waves (B = {val}) typical of mass synchronized push campaigns."
            elif feat_name == "urgency_scam_score" and val > 0.3:
                diagnostic = f"High density of scam/giveaway trigger keywords ({round(val * 100)}% hits)."
            elif feat_name == "url_density" and val > 0.5:
                diagnostic = f"Extreme link density ({round(val * 100)}% of posts contain links) indicates traffic driving."
            elif feat_name == "retweet_ratio" and val > 0.65:
                diagnostic = f"Retweet ratio ({round(val * 100)}%) resembles an amplification puppet / sybil node."
            elif feat_name == "account_age_days" and val < 30:
                diagnostic = f"Fresh account ({int(val)} days old) lacking behavioral longevity history."
            elif feat_name == "circadian_entropy_norm" and val > 0.88:
                diagnostic = "Active continuously across 24 hours without human sleep troughs."
            elif feat_name == "repetition_similarity" and val > 0.35:
                diagnostic = f"High text similarity ({round(val * 100)}%) across successive posts."
            elif feat_name == "account_age_days" and val > 500:
                diagnostic = f"Established account age ({int(val)} days) strongly reduces fraud probability."
            elif feat_name == "reciprocal_rate_est" and val > 0.5:
                diagnostic = "Balanced mutual follow relationships reflect genuine human social networking."

            drivers.append({
                "feature": feat_name,
                "label": human_readable_names.get(feat_name, feat_name),
                "value": round(val, 3),
                "contribution": round(contribution, 2),
                "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                "diagnostic": diagnostic
            })

        # Sort by absolute impact
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return drivers[:8]  # Top 8 most impactful drivers

    def _calculate_radar_metrics(self, features: ExtractedFeatures) -> Dict[str, float]:
        """Map 25 features to 6 intuitive high-level behavioral dimensions on a 0-100 scale."""
        # 1. Temporal Automation (0 = organic sleep rhythms, 100 = robotic continuous timer)
        b_risk = abs(features.burstiness_coefficient) * 60.0
        c_risk = max(0.0, (features.circadian_entropy_norm - 0.6) * 150.0)
        temporal = np.clip(b_risk * 0.5 + c_risk * 0.5 + features.night_activity_ratio * 100.0 * 0.3, 5.0, 98.0)

        # 2. Content Redundancy (0 = diverse human discourse, 100 = duplicate templated spam)
        rep_risk = features.repetition_similarity * 100.0
        lex_risk = (1.0 - features.lexical_diversity) * 100.0
        redundancy = np.clip(rep_risk * 0.6 + lex_risk * 0.4, 5.0, 95.0)

        # 3. Scam & Urgency Triggers (0 = zero scam keywords, 100 = aggressive lures)
        scam_urgency = np.clip(features.urgency_scam_score * 120.0 + features.all_caps_ratio * 80.0, 0.0, 100.0)

        # 4. Outbound Spreading & Velocity (0 = casual chatter, 100 = link flood)
        spreading = np.clip(features.url_density * 60.0 + (features.peak_posts_per_hour / 15.0) * 40.0, 5.0, 98.0)

        # 5. Network Asymmetry (0 = mutual friend circle, 100 = mass following / low reciprocity)
        network = np.clip((1.0 - features.reciprocal_rate_est) * 70.0 + min(30.0, features.following_to_follower_ratio * 5.0), 5.0, 95.0)

        # 6. Profile Anonymity (0 = verified established persona, 100 = disposable egg account)
        age_factor = max(0.0, 1.0 - (features.account_age_days / 365.0)) * 40.0
        avatar_factor = features.has_default_avatar * 25.0
        bio_factor = features.has_empty_bio * 15.0
        digit_factor = min(20.0, features.username_digit_count * 4.0)
        verified_discount = -25.0 if features.is_verified else 0.0
        anonymity = np.clip(age_factor + avatar_factor + bio_factor + digit_factor + verified_discount, 5.0, 95.0)

        return {
            "Temporal Automation": round(float(temporal), 1),
            "Content Redundancy": round(float(redundancy), 1),
            "Scam / Urgency Triggers": round(float(scam_urgency), 1),
            "Outbound Link Velocity": round(float(spreading), 1),
            "Network Asymmetry": round(float(network), 1),
            "Profile Anonymity": round(float(anonymity), 1)
        }

    def _simulate_circadian_curve(self, features: ExtractedFeatures) -> List[int]:
        """Generate a representative 24-hour hourly activity count curve for visualization."""
        curve = []
        is_bot = features.circadian_entropy_norm > 0.85
        total_sample_posts = max(24, int(features.activity_velocity * 3))

        for hour in range(24):
            if is_bot:
                # Flat distribution across 24 hours
                base = total_sample_posts / 24.0
                noise = np.random.uniform(-0.15, 0.15) * base
                val = max(1, int(base + noise))
            else:
                # Human circadian pattern: low at night (0-6), peak around 12-14 and 19-22
                if 1 <= hour <= 5:
                    weight = 0.08
                elif 6 <= hour <= 9:
                    weight = 0.6
                elif 10 <= hour <= 17:
                    weight = 1.0
                elif 18 <= hour <= 22:
                    weight = 1.2
                else:
                    weight = 0.3
                val = max(0, int(weight * (total_sample_posts / 18.0) + np.random.normal(0, 0.5)))
            curve.append(val)
        return curve


_model_instance: Optional[FraudDetectionModel] = None

def get_model() -> FraudDetectionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = FraudDetectionModel()
        _model_instance.train_and_evaluate()
    return _model_instance
