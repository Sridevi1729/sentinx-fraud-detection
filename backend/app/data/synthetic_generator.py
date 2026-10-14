import random
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any
from ..features.extractor import AccountBehaviorInput, PostData, ExtractedFeatures


def generate_synthetic_feature_dataset(n_samples: int = 1200, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic feature matrix X and multiclass labels y.
    Classes:
      0: Legitimate User
      1: Automated Spambot
      2: Phishing / Financial Scam
      3: Coordinated Sybil / Astroturfer
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    n_per_class = n_samples // 4
    features = []
    labels = []

    # 1. Legitimate Users (Class 0)
    for _ in range(n_per_class):
        burstiness = np.clip(np.random.normal(loc=-0.05, scale=0.18), -0.6, 0.4)
        circadian_norm = np.clip(np.random.normal(loc=0.72, scale=0.1), 0.45, 0.92)  # clear sleep cycles
        night_ratio = np.clip(np.random.beta(a=1.5, b=8.0), 0.0, 0.25)
        inter_arrival_cv = np.clip(np.random.normal(loc=1.1, scale=0.3), 0.5, 2.5)
        peak_posts_per_hour = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.12, 0.06, 0.02])

        follower_ratio = np.clip(np.random.beta(a=3.0, b=3.0), 0.2, 0.8)
        following_to_follower = np.clip(np.random.exponential(scale=1.2), 0.1, 5.0)
        mentions_per_post = np.clip(np.random.exponential(scale=0.4), 0.0, 1.5)
        hashtag_density = np.clip(np.random.exponential(scale=0.6), 0.0, 2.0)
        url_density = np.clip(np.random.beta(a=1.2, b=4.0), 0.0, 0.4)
        retweet_ratio = np.clip(np.random.beta(a=2.0, b=4.0), 0.05, 0.5)
        reciprocal_rate = np.clip(np.random.normal(loc=0.6, scale=0.15), 0.25, 0.95)

        repetition_similarity = np.clip(np.random.beta(a=1.0, b=6.0), 0.0, 0.25)
        lexical_diversity = np.clip(np.random.normal(loc=0.78, scale=0.08), 0.55, 0.95)
        shannon_text_entropy = np.clip(np.random.normal(loc=4.3, scale=0.3), 3.5, 4.9)
        urgency_scam_score = np.clip(np.random.beta(a=0.5, b=10.0), 0.0, 0.08)
        avg_post_length = np.clip(np.random.normal(loc=110.0, scale=35.0), 30.0, 260.0)
        all_caps_ratio = np.clip(np.random.beta(a=1.0, b=12.0), 0.0, 0.15)

        account_age_days = float(np.random.randint(150, 4000))
        activity_velocity = np.clip(np.random.gamma(shape=1.5, scale=0.8), 0.05, 6.0)
        username_digit_count = float(np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05]))
        username_entropy = np.clip(np.random.normal(loc=2.8, scale=0.4), 1.8, 3.8)
        has_default_avatar = float(np.random.choice([0, 1], p=[0.94, 0.06]))
        has_empty_bio = float(np.random.choice([0, 1], p=[0.85, 0.15]))
        is_verified = float(np.random.choice([0, 1], p=[0.92, 0.08]))

        features.append([
            burstiness, circadian_norm, night_ratio, inter_arrival_cv, peak_posts_per_hour,
            follower_ratio, following_to_follower, mentions_per_post, hashtag_density, url_density,
            retweet_ratio, reciprocal_rate, repetition_similarity, lexical_diversity, shannon_text_entropy,
            urgency_scam_score, avg_post_length, all_caps_ratio, account_age_days, activity_velocity,
            username_digit_count, username_entropy, has_default_avatar, has_empty_bio, is_verified
        ])
        labels.append(0)

    # 2. Automated Spambots (Class 1) - Periodic timers, massive velocity, link spam
    for _ in range(n_per_class):
        # Strict timer periodicity means negative burstiness or extreme burstiness in bursts
        burstiness = np.clip(np.random.choice([
            np.random.normal(loc=-0.75, scale=0.15),  # regular cron
            np.random.normal(loc=0.85, scale=0.1)     # massive wave burst
        ]), -1.0, 1.0)
        circadian_norm = np.clip(np.random.normal(loc=0.96, scale=0.04), 0.88, 1.0)  # active 24/7 flat
        night_ratio = np.clip(np.random.normal(loc=0.25, scale=0.06), 0.18, 0.40)
        inter_arrival_cv = np.clip(np.random.normal(loc=0.25, scale=0.15), 0.05, 0.7)  # low variance
        peak_posts_per_hour = np.random.choice([15, 25, 45, 80], p=[0.3, 0.4, 0.2, 0.1])

        follower_ratio = np.clip(np.random.beta(a=1.0, b=5.0), 0.01, 0.25)
        following_to_follower = np.clip(np.random.normal(loc=8.0, scale=4.0), 2.5, 30.0)
        mentions_per_post = np.clip(np.random.normal(loc=1.8, scale=0.6), 0.4, 4.0)
        hashtag_density = np.clip(np.random.normal(loc=3.8, scale=1.2), 1.5, 8.0)
        url_density = np.clip(np.random.beta(a=5.0, b=1.5), 0.65, 1.0)
        retweet_ratio = np.clip(np.random.beta(a=3.0, b=2.0), 0.35, 0.95)
        reciprocal_rate = np.clip(np.random.normal(loc=0.12, scale=0.08), 0.01, 0.3)

        repetition_similarity = np.clip(np.random.normal(loc=0.65, scale=0.15), 0.35, 0.95)
        lexical_diversity = np.clip(np.random.normal(loc=0.38, scale=0.1), 0.15, 0.55)
        shannon_text_entropy = np.clip(np.random.normal(loc=3.2, scale=0.4), 2.1, 3.9)
        urgency_scam_score = np.clip(np.random.beta(a=1.5, b=4.0), 0.1, 0.6)
        avg_post_length = np.clip(np.random.normal(loc=75.0, scale=25.0), 25.0, 160.0)
        all_caps_ratio = np.clip(np.random.beta(a=1.5, b=6.0), 0.05, 0.35)

        account_age_days = float(np.random.randint(5, 300))
        activity_velocity = np.clip(np.random.normal(loc=25.0, scale=10.0), 8.0, 75.0)
        username_digit_count = float(np.random.choice([4, 5, 6, 8], p=[0.3, 0.4, 0.2, 0.1]))
        username_entropy = np.clip(np.random.normal(loc=3.6, scale=0.3), 3.0, 4.2)
        has_default_avatar = float(np.random.choice([0, 1], p=[0.45, 0.55]))
        has_empty_bio = float(np.random.choice([0, 1], p=[0.4, 0.6]))
        is_verified = 0.0

        features.append([
            burstiness, circadian_norm, night_ratio, inter_arrival_cv, peak_posts_per_hour,
            follower_ratio, following_to_follower, mentions_per_post, hashtag_density, url_density,
            retweet_ratio, reciprocal_rate, repetition_similarity, lexical_diversity, shannon_text_entropy,
            urgency_scam_score, avg_post_length, all_caps_ratio, account_age_days, activity_velocity,
            username_digit_count, username_entropy, has_default_avatar, has_empty_bio, is_verified
        ])
        labels.append(1)

    # 3. Phishing / Financial Scam Accounts (Class 2) - High urgency, impersonation, new age, high caps
    for _ in range(n_per_class):
        burstiness = np.clip(np.random.normal(loc=0.55, scale=0.2), 0.1, 0.95)
        circadian_norm = np.clip(np.random.normal(loc=0.82, scale=0.1), 0.6, 0.98)
        night_ratio = np.clip(np.random.normal(loc=0.18, scale=0.08), 0.05, 0.38)
        inter_arrival_cv = np.clip(np.random.normal(loc=1.4, scale=0.4), 0.6, 2.5)
        peak_posts_per_hour = np.random.choice([6, 12, 18, 25], p=[0.4, 0.35, 0.15, 0.1])

        follower_ratio = np.clip(np.random.beta(a=1.5, b=4.0), 0.05, 0.4)
        following_to_follower = np.clip(np.random.normal(loc=5.5, scale=3.0), 1.5, 18.0)
        mentions_per_post = np.clip(np.random.normal(loc=1.2, scale=0.5), 0.2, 3.0)
        hashtag_density = np.clip(np.random.normal(loc=2.2, scale=0.8), 0.8, 4.5)
        url_density = np.clip(np.random.beta(a=4.0, b=1.5), 0.6, 0.98)
        retweet_ratio = np.clip(np.random.beta(a=1.5, b=3.5), 0.05, 0.45)
        reciprocal_rate = np.clip(np.random.normal(loc=0.15, scale=0.08), 0.02, 0.35)

        repetition_similarity = np.clip(np.random.normal(loc=0.58, scale=0.15), 0.25, 0.88)
        lexical_diversity = np.clip(np.random.normal(loc=0.45, scale=0.1), 0.2, 0.65)
        shannon_text_entropy = np.clip(np.random.normal(loc=3.4, scale=0.35), 2.5, 4.1)
        # Strong indicator: high urgency scam words (giveaway, dm, free crypto)
        urgency_scam_score = np.clip(np.random.beta(a=5.0, b=2.0), 0.55, 1.0)
        avg_post_length = np.clip(np.random.normal(loc=130.0, scale=35.0), 60.0, 240.0)
        all_caps_ratio = np.clip(np.random.beta(a=3.0, b=5.0), 0.18, 0.65)

        account_age_days = float(np.random.randint(1, 45))  # freshly created
        activity_velocity = np.clip(np.random.normal(loc=14.0, scale=6.0), 4.0, 35.0)
        username_digit_count = float(np.random.choice([2, 3, 4, 6], p=[0.2, 0.4, 0.3, 0.1]))
        username_entropy = np.clip(np.random.normal(loc=3.3, scale=0.35), 2.5, 4.0)
        has_default_avatar = float(np.random.choice([0, 1], p=[0.75, 0.25]))  # often steals legit avatar
        has_empty_bio = float(np.random.choice([0, 1], p=[0.8, 0.2]))
        is_verified = 0.0

        features.append([
            burstiness, circadian_norm, night_ratio, inter_arrival_cv, peak_posts_per_hour,
            follower_ratio, following_to_follower, mentions_per_post, hashtag_density, url_density,
            retweet_ratio, reciprocal_rate, repetition_similarity, lexical_diversity, shannon_text_entropy,
            urgency_scam_score, avg_post_length, all_caps_ratio, account_age_days, activity_velocity,
            username_digit_count, username_entropy, has_default_avatar, has_empty_bio, is_verified
        ])
        labels.append(2)

    # 4. Coordinated Inauthentic Sybil Accounts (Class 3) - Retweet amplifies, coordinated bursts
    for _ in range(n_per_class):
        burstiness = np.clip(np.random.normal(loc=0.72, scale=0.15), 0.35, 0.98)
        circadian_norm = np.clip(np.random.normal(loc=0.88, scale=0.08), 0.72, 0.98)
        night_ratio = np.clip(np.random.normal(loc=0.22, scale=0.07), 0.1, 0.38)
        inter_arrival_cv = np.clip(np.random.normal(loc=1.6, scale=0.5), 0.8, 3.2)
        peak_posts_per_hour = np.random.choice([10, 20, 35], p=[0.5, 0.35, 0.15])

        follower_ratio = np.clip(np.random.beta(a=2.0, b=4.0), 0.1, 0.45)
        following_to_follower = np.clip(np.random.normal(loc=4.0, scale=2.0), 1.5, 10.0)
        mentions_per_post = np.clip(np.random.normal(loc=0.9, scale=0.4), 0.1, 2.0)
        hashtag_density = np.clip(np.random.normal(loc=2.8, scale=1.0), 1.0, 5.5)
        url_density = np.clip(np.random.beta(a=2.0, b=3.0), 0.15, 0.6)
        # Signature: extremely high retweet ratio (amplification drone)
        retweet_ratio = np.clip(np.random.beta(a=6.0, b=1.5), 0.75, 1.0)
        reciprocal_rate = np.clip(np.random.normal(loc=0.2, scale=0.08), 0.05, 0.4)

        repetition_similarity = np.clip(np.random.normal(loc=0.45, scale=0.15), 0.2, 0.8)
        lexical_diversity = np.clip(np.random.normal(loc=0.55, scale=0.12), 0.3, 0.75)
        shannon_text_entropy = np.clip(np.random.normal(loc=3.8, scale=0.3), 3.0, 4.4)
        urgency_scam_score = np.clip(np.random.beta(a=1.0, b=5.0), 0.02, 0.25)
        avg_post_length = np.clip(np.random.normal(loc=95.0, scale=25.0), 40.0, 180.0)
        all_caps_ratio = np.clip(np.random.beta(a=1.5, b=8.0), 0.02, 0.2)

        account_age_days = float(np.random.randint(60, 900))  # aged sleeper accounts
        activity_velocity = np.clip(np.random.normal(loc=12.0, scale=5.0), 3.0, 25.0)
        username_digit_count = float(np.random.choice([1, 2, 4, 5], p=[0.2, 0.3, 0.3, 0.2]))
        username_entropy = np.clip(np.random.normal(loc=3.1, scale=0.3), 2.4, 3.8)
        has_default_avatar = float(np.random.choice([0, 1], p=[0.7, 0.3]))
        has_empty_bio = float(np.random.choice([0, 1], p=[0.6, 0.4]))
        is_verified = 0.0

        features.append([
            burstiness, circadian_norm, night_ratio, inter_arrival_cv, peak_posts_per_hour,
            follower_ratio, following_to_follower, mentions_per_post, hashtag_density, url_density,
            retweet_ratio, reciprocal_rate, repetition_similarity, lexical_diversity, shannon_text_entropy,
            urgency_scam_score, avg_post_length, all_caps_ratio, account_age_days, activity_velocity,
            username_digit_count, username_entropy, has_default_avatar, has_empty_bio, is_verified
        ])
        labels.append(3)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def get_sample_profiles() -> List[Dict[str, Any]]:
    """Return preloaded representative profile case studies for quick 1-click testing in the UI."""
    now = datetime.now(timezone.utc)

    # 1. Genuine Human Tech Creator
    legit_posts = [
        PostData(
            text="Really enjoying the new Python 3.13 features today! JIT compiler is super promising. What are your thoughts?",
            timestamp=now - timedelta(hours=3, minutes=12),
            likes_count=42,
            retweets_count=7
        ),
        PostData(
            text="Working on our open-source ML benchmark suite. Pull requests and feedback are welcome at https://github.com/example/benchmarks",
            timestamp=now - timedelta(hours=7, minutes=45),
            likes_count=89,
            retweets_count=19
        ),
        PostData(
            text="Coffee break after debugging an async deadlock for 2 hours. Life is good again ☕",
            timestamp=now - timedelta(hours=28, minutes=10),
            likes_count=134,
            retweets_count=3
        ),
        PostData(
            text="Great conversation on tech architectures with @sarah_dev. Always learn so much from peer reviews.",
            timestamp=now - timedelta(hours=52, minutes=5),
            likes_count=56,
            retweets_count=4
        )
    ]
    legit_profile = AccountBehaviorInput(
        username="alex_engineer",
        account_age_days=1420,
        followers_count=3450,
        following_count=890,
        total_posts_count=2140,
        has_default_avatar=False,
        has_empty_bio=False,
        is_verified=True,
        bio_text="Senior ML Engineer & Open Source contributor. Writing about distributed systems and Python.",
        recent_posts=legit_posts
    )

    # 2. Automated High-Frequency Spambot
    bot_posts = []
    for i in range(12):
        bot_posts.append(
            PostData(
                text=f"🔥 Top trending discounts online! Click the link now before it expires! http://tinyurl.com/deal{i} #deals #sale #discount #shopping",
                timestamp=now - timedelta(minutes=15 * i),  # exactly 15 min cron interval
                likes_count=0,
                retweets_count=1,
                is_retweet=False
            )
        )
    spambot_profile = AccountBehaviorInput(
        username="dealz_flash_984712",
        account_age_days=18,
        followers_count=42,
        following_count=2150,
        total_posts_count=8420,
        has_default_avatar=True,
        has_empty_bio=True,
        is_verified=False,
        bio_text="",
        recent_posts=bot_posts
    )

    # 3. Phishing / Crypto Scammer
    phish_posts = [
        PostData(
            text="🚨 OFFICIAL AIRDROP ANNOUNCEMENT! Claim your 5,000 FREE SOL tokens immediately! LIMITED TO FIRST 500 USERS! Connect wallet here: http://sol-claim-airdrop.xyz 🚀💰",
            timestamp=now - timedelta(minutes=22),
            likes_count=14,
            retweets_count=8
        ),
        PostData(
            text="CONGRATULATIONS TO ALL WINNERS! 🎁 DM US NOW ON TELEGRAM @crypto_bonus_support to receive your guaranteed bitcoin prize! HURRY!",
            timestamp=now - timedelta(minutes=45),
            likes_count=9,
            retweets_count=5
        ),
        PostData(
            text="Binance exclusive bonus event! Double your deposit within 24h guaranteed! WhatsApp support available 24/7! Click: http://claim-crypto-bonus.net",
            timestamp=now - timedelta(hours=2, minutes=10),
            likes_count=21,
            retweets_count=12
        )
    ]
    phish_profile = AccountBehaviorInput(
        username="solana_official_airdrop99",
        account_age_days=4,  # created 4 days ago
        followers_count=180,
        following_count=1420,
        total_posts_count=320,
        has_default_avatar=False,
        has_empty_bio=False,
        is_verified=False,
        bio_text="Official Solana Airdrop Distribution Team. Support & claim verification desk.",
        recent_posts=phish_posts
    )

    # 4. Coordinated Inauthentic Sybil Drone
    sybil_posts = []
    for i in range(8):
        sybil_posts.append(
            PostData(
                text="RT @mega_influencer: Check out this breaking news! Unbelievable revelations today! #TrendAlert #Viral",
                timestamp=now - timedelta(hours=i * 2 + (i % 2)),
                is_retweet=True,
                likes_count=1,
                retweets_count=0
            )
        )
    sybil_profile = AccountBehaviorInput(
        username="user_echo_38120",
        account_age_days=310,
        followers_count=110,
        following_count=1890,
        total_posts_count=4910,
        has_default_avatar=True,
        has_empty_bio=False,
        is_verified=False,
        bio_text="News enthusiast and re-sharer.",
        recent_posts=sybil_posts
    )

    return [
        {
            "id": "legitimate_creator",
            "name": "Alex (Verified Tech Creator)",
            "category": "Legitimate User",
            "expected_risk": "Low (Safe)",
            "description": "Organic circadian posting pattern, balanced follower ratio, high lexical diversity, and no scam triggers.",
            "data": legit_profile.model_dump(mode="json")
        },
        {
            "id": "automated_spambot",
            "name": "DealsBot 984712 (Scraper/Spammer)",
            "category": "Automated Spambot",
            "expected_risk": "High (Fraudulent)",
            "description": "Rigid 15-minute cron periodicity, 100% URL density, repetitive template spam, and skewed followings.",
            "data": spambot_profile.model_dump(mode="json")
        },
        {
            "id": "crypto_phishing",
            "name": "Crypto Airdrop Scammer",
            "category": "Phishing & Financial Fraud",
            "expected_risk": "Critical (Fraudulent)",
            "description": "Created 4 days ago, high urgency keywords, uppercase shouting, suspicious phishing domains, and numeric handle.",
            "data": phish_profile.model_dump(mode="json")
        },
        {
            "id": "sybil_amplifier",
            "name": "Coordinated Sybil Retweet Drone",
            "category": "Coordinated Inauthentic Behavior",
            "expected_risk": "High (Suspicious)",
            "description": "Aged dormant account with 100% retweet amplification, low reciprocity, and minimal original content.",
            "data": sybil_profile.model_dump(mode="json")
        }
    ]
