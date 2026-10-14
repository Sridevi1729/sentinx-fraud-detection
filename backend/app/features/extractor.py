import re
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PostData(BaseModel):
    text: str
    timestamp: datetime
    is_retweet: bool = False
    likes_count: int = 0
    retweets_count: int = 0


class AccountBehaviorInput(BaseModel):
    username: str
    account_age_days: int = Field(ge=1, description="Account age in days")
    followers_count: int = Field(ge=0, default=0)
    following_count: int = Field(ge=0, default=0)
    total_posts_count: int = Field(ge=0, default=0)
    has_default_avatar: bool = False
    has_empty_bio: bool = False
    is_verified: bool = False
    bio_text: str = ""
    recent_posts: List[PostData] = []


class ExtractedFeatures(BaseModel):
    # Temporal Dynamics
    burstiness_coefficient: float
    circadian_entropy: float
    circadian_entropy_norm: float
    night_activity_ratio: float
    inter_arrival_mean_seconds: float
    inter_arrival_cv: float
    peak_posts_per_hour: int

    # Interaction & Network
    follower_ratio: float
    following_to_follower_ratio: float
    mentions_per_post: float
    hashtag_density: float
    url_density: float
    retweet_ratio: float
    reciprocal_rate_est: float

    # Content & Linguistic
    repetition_similarity: float
    lexical_diversity: float
    shannon_text_entropy: float
    urgency_scam_score: float
    avg_post_length: float
    all_caps_ratio: float

    # Digital Footprint & Metadata
    account_age_days: int
    activity_velocity: float  # posts per day
    username_digit_count: int
    username_entropy: float
    has_default_avatar: int
    has_empty_bio: int
    is_verified: int

    def to_feature_vector(self) -> List[float]:
        """Return ordered vector matching the training feature columns."""
        return [
            self.burstiness_coefficient,
            self.circadian_entropy_norm,
            self.night_activity_ratio,
            self.inter_arrival_cv,
            self.peak_posts_per_hour,
            self.follower_ratio,
            self.following_to_follower_ratio,
            self.mentions_per_post,
            self.hashtag_density,
            self.url_density,
            self.retweet_ratio,
            self.reciprocal_rate_est,
            self.repetition_similarity,
            self.lexical_diversity,
            self.shannon_text_entropy,
            self.urgency_scam_score,
            self.avg_post_length,
            self.all_caps_ratio,
            float(self.account_age_days),
            self.activity_velocity,
            float(self.username_digit_count),
            self.username_entropy,
            float(self.has_default_avatar),
            float(self.has_empty_bio),
            float(self.is_verified),
        ]

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "burstiness_coefficient",
            "circadian_entropy_norm",
            "night_activity_ratio",
            "inter_arrival_cv",
            "peak_posts_per_hour",
            "follower_ratio",
            "following_to_follower_ratio",
            "mentions_per_post",
            "hashtag_density",
            "url_density",
            "retweet_ratio",
            "reciprocal_rate_est",
            "repetition_similarity",
            "lexical_diversity",
            "shannon_text_entropy",
            "urgency_scam_score",
            "avg_post_length",
            "all_caps_ratio",
            "account_age_days",
            "activity_velocity",
            "username_digit_count",
            "username_entropy",
            "has_default_avatar",
            "has_empty_bio",
            "is_verified",
        ]


class BehavioralFeatureExtractor:
    SCAM_KEYWORDS = {
        "crypto", "giveaway", "airdrop", "free", "claim", "dm", "whatsapp",
        "telegram", "invest", "profit", "guaranteed", "urgent", "winner",
        "congratulations", "bonus", "discount", "limited", "exclusive",
        "passive income", "double your", "seed phrase", "binance", "metamask",
        "btc", "eth", "solana", "forex", "trading bot"
    }

    URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    MENTION_REGEX = re.compile(r"@\w+")
    HASHTAG_REGEX = re.compile(r"#\w+")

    @classmethod
    def calculate_shannon_entropy(cls, text: str) -> float:
        """Calculate Shannon entropy for a given text string."""
        if not text:
            return 0.0
        counts = {}
        for char in text:
            counts[char] = counts.get(char, 0) + 1
        n = len(text)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        return round(entropy, 4)

    @classmethod
    def calculate_jaccard_similarity(cls, tokens1: set, tokens2: set) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def extract_from_account(cls, account: AccountBehaviorInput) -> ExtractedFeatures:
        posts = account.recent_posts
        num_posts = len(posts)

        # 1. Temporal Dynamics
        if num_posts >= 2:
            # Sort by timestamp
            sorted_posts = sorted(posts, key=lambda p: p.timestamp)
            deltas = [
                (sorted_posts[i + 1].timestamp - sorted_posts[i].timestamp).total_seconds()
                for i in range(num_posts - 1)
            ]
            # Ensure non-negative deltas
            deltas = [max(0.1, d) for d in deltas]

            mean_delta = sum(deltas) / len(deltas)
            var_delta = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
            std_delta = math.sqrt(var_delta)

            # Burstiness B = (sigma - mu) / (sigma + mu) in [-1, 1]
            if (std_delta + mean_delta) > 0:
                burstiness = (std_delta - mean_delta) / (std_delta + mean_delta)
            else:
                burstiness = 0.0

            # Inter-arrival CV
            inter_arrival_cv = (std_delta / mean_delta) if mean_delta > 0 else 0.0

            # Circadian distribution (24 hourly bins)
            hour_counts = [0] * 24
            night_posts = 0
            hour_dict: Dict[str, int] = {}

            for p in sorted_posts:
                h = p.timestamp.hour
                hour_counts[h] += 1
                if 0 <= h < 6:
                    night_posts += 1
                # hourly peak key YYYY-MM-DD-HH
                hour_key = p.timestamp.strftime("%Y-%m-%d-%H")
                hour_dict[hour_key] = hour_dict.get(hour_key, 0) + 1

            # Shannon entropy over 24 bins
            circadian_entropy = 0.0
            for count in hour_counts:
                if count > 0:
                    p_i = count / num_posts
                    circadian_entropy -= p_i * math.log2(p_i)

            max_circadian_entropy = math.log2(24)  # ~4.585
            circadian_entropy_norm = circadian_entropy / max_circadian_entropy
            night_activity_ratio = night_posts / num_posts
            peak_posts_per_hour = max(hour_dict.values()) if hour_dict else 1
            inter_arrival_mean = mean_delta
        else:
            # Default / single post fallback
            burstiness = 0.0
            circadian_entropy = 0.0
            circadian_entropy_norm = 0.5
            night_activity_ratio = 0.1
            inter_arrival_mean = 86400.0  # 1 day
            inter_arrival_cv = 0.0
            peak_posts_per_hour = num_posts

        # 2. Interaction & Network Metrics
        total_connections = account.followers_count + account.following_count + 1
        follower_ratio = account.followers_count / total_connections
        following_to_follower_ratio = account.following_count / (account.followers_count + 1)

        total_mentions = 0
        total_hashtags = 0
        total_urls = 0
        total_retweets = 0
        total_chars = 0
        total_uppercase = 0
        scam_hits = 0
        all_tokens = []
        token_sets = []

        for p in posts:
            text = p.text
            total_chars += len(text)
            total_uppercase += sum(1 for c in text if c.isupper())
            
            mentions = cls.MENTION_REGEX.findall(text)
            hashtags = cls.HASHTAG_REGEX.findall(text)
            urls = cls.URL_REGEX.findall(text)

            total_mentions += len(mentions)
            total_hashtags += len(hashtags)
            total_urls += 1 if len(urls) > 0 else 0
            if p.is_retweet:
                total_retweets += 1

            # Tokenize for linguistic metrics
            clean_text = cls.URL_REGEX.sub("", text.lower())
            tokens = [w for w in re.findall(r"\b\w+\b", clean_text) if len(w) > 1]
            all_tokens.extend(tokens)
            token_sets.append(set(tokens))

            # Scam score hits
            for kw in cls.SCAM_KEYWORDS:
                if kw in clean_text:
                    scam_hits += 1

        mentions_per_post = (total_mentions / num_posts) if num_posts > 0 else 0.0
        hashtag_density = (total_hashtags / num_posts) if num_posts > 0 else 0.0
        url_density = (total_urls / num_posts) if num_posts > 0 else 0.0
        retweet_ratio = (total_retweets / num_posts) if num_posts > 0 else 0.0

        # Estimated reciprocal rate based on follower balance & organic interactions
        reciprocal_rate_est = min(1.0, 2.0 * min(account.followers_count, account.following_count) / max(1, account.followers_count + account.following_count))

        # 3. Content & Linguistic Volatility
        if num_posts >= 2:
            similarities = []
            sample_size = min(10, num_posts)
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    sim = cls.calculate_jaccard_similarity(token_sets[i], token_sets[j])
                    similarities.append(sim)
            repetition_similarity = (sum(similarities) / len(similarities)) if similarities else 0.0
        else:
            repetition_similarity = 0.0

        unique_tokens = len(set(all_tokens))
        total_token_count = len(all_tokens)
        lexical_diversity = (unique_tokens / total_token_count) if total_token_count > 0 else 1.0

        combined_text = " ".join([p.text for p in posts]) + " " + account.bio_text
        shannon_text_entropy = cls.calculate_shannon_entropy(combined_text)

        urgency_scam_score = min(1.0, scam_hits / max(1, num_posts))
        avg_post_length = (total_chars / num_posts) if num_posts > 0 else 0.0
        all_caps_ratio = (total_uppercase / total_chars) if total_chars > 0 else 0.0

        # 4. Digital Footprint & Metadata
        effective_age = max(1, account.account_age_days)
        total_posts = max(account.total_posts_count, num_posts)
        activity_velocity = total_posts / effective_age

        digits_in_username = sum(1 for c in account.username if c.isdigit())
        username_entropy = cls.calculate_shannon_entropy(account.username)

        return ExtractedFeatures(
            burstiness_coefficient=round(burstiness, 4),
            circadian_entropy=round(circadian_entropy, 4),
            circadian_entropy_norm=round(circadian_entropy_norm, 4),
            night_activity_ratio=round(night_activity_ratio, 4),
            inter_arrival_mean_seconds=round(inter_arrival_mean, 2),
            inter_arrival_cv=round(inter_arrival_cv, 4),
            peak_posts_per_hour=peak_posts_per_hour,
            follower_ratio=round(follower_ratio, 4),
            following_to_follower_ratio=round(following_to_follower_ratio, 4),
            mentions_per_post=round(mentions_per_post, 4),
            hashtag_density=round(hashtag_density, 4),
            url_density=round(url_density, 4),
            retweet_ratio=round(retweet_ratio, 4),
            reciprocal_rate_est=round(reciprocal_rate_est, 4),
            repetition_similarity=round(repetition_similarity, 4),
            lexical_diversity=round(lexical_diversity, 4),
            shannon_text_entropy=round(shannon_text_entropy, 4),
            urgency_scam_score=round(urgency_scam_score, 4),
            avg_post_length=round(avg_post_length, 2),
            all_caps_ratio=round(all_caps_ratio, 4),
            account_age_days=account.account_age_days,
            activity_velocity=round(activity_velocity, 4),
            username_digit_count=digits_in_username,
            username_entropy=round(username_entropy, 4),
            has_default_avatar=1 if account.has_default_avatar else 0,
            has_empty_bio=1 if account.has_empty_bio else 0,
            is_verified=1 if account.is_verified else 0,
        )
