"""
Retention Messaging Engine.
Template-based personalized retention message generation.
No LLM API required — uses rich template library with dynamic personalization.
"""

import logging
import random
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_retention_templates() -> dict[str, list[dict[str, str]]]:
    """
    Returns a comprehensive dict of segment → list of message templates.
    Each template has: subject, body, offer_type, urgency.
    Placeholders: {name}, {tenure_days}, {plan_name}, {favorite_feature},
                  {discount_pct}, {discount_months}, {churn_risk_label},
                  {days_inactive}, {clv_tier}
    """
    return {
        "high_clv_persuadable": [
            {
                "subject": "We've missed you, {name} — here's a special thank-you",
                "body": (
                    "Hi {name},\n\n"
                    "We noticed you haven't been on {plan_name} lately — "
                    "and we'd love to welcome you back.\n\n"
                    "As one of our most valued members (you've been with us {tenure_days} days!), "
                    "we'd like to offer you {discount_pct}% off your next {discount_months} months "
                    "— absolutely free to apply, no commitment needed.\n\n"
                    "Your favorite feature, {favorite_feature}, has gotten even better with new updates. "
                    "Come discover what's new.\n\n"
                    "Claim your offer here: [OFFER_LINK]\n\n"
                    "With appreciation,\nThe {plan_name} Team"
                ),
                "offer_type": "discount",
                "urgency": "high",
            },
            {
                "subject": "Exclusive loyalty reward inside — {name}, this is for you",
                "body": (
                    "Dear {name},\n\n"
                    "Your {tenure_days}-day journey with us means everything. "
                    "To show how much we value you, we're unlocking a premium upgrade — "
                    "{discount_pct}% off for the next {discount_months} months, "
                    "plus early access to our newest features.\n\n"
                    "You love {favorite_feature}, and we've added three new capabilities "
                    "we think you'll enjoy even more.\n\n"
                    "Your personalized offer expires in 7 days. Don't miss it.\n\n"
                    "Redeem now: [OFFER_LINK]\n\n"
                    "Warm regards,\nYour {plan_name} Team"
                ),
                "offer_type": "upgrade",
                "urgency": "high",
            },
        ],
        "medium_clv_persuadable": [
            {
                "subject": "{name}, we want you to stay — special offer inside",
                "body": (
                    "Hi {name},\n\n"
                    "We've noticed you've been less active recently "
                    "({days_inactive} days since your last session). "
                    "We'd love to know if there's anything we can do better.\n\n"
                    "In the meantime, enjoy {discount_pct}% off your next {discount_months} months "
                    "as our way of saying thank you for being with us.\n\n"
                    "Log in today and rediscover {favorite_feature}.\n\n"
                    "Claim your discount: [OFFER_LINK]\n\n"
                    "Best,\nThe {plan_name} Team"
                ),
                "offer_type": "discount",
                "urgency": "medium",
            },
            {
                "subject": "A gift for you, {name} — come back and see what's new",
                "body": (
                    "Hello {name},\n\n"
                    "It's been a while since we've seen you! We've been busy adding "
                    "new features to {plan_name} — and we think you'll love them.\n\n"
                    "As a thank-you for your loyalty ({tenure_days} days and counting), "
                    "here's {discount_pct}% off for {discount_months} months. "
                    "Automatically applied — no code needed.\n\n"
                    "Explore what's new: [OFFER_LINK]\n\n"
                    "Cheers,\nThe {plan_name} Team"
                ),
                "offer_type": "discount",
                "urgency": "medium",
            },
        ],
        "low_clv_persuadable": [
            {
                "subject": "Thinking of leaving? Here's {discount_pct}% off to stay",
                "body": (
                    "Hi {name},\n\n"
                    "We saw that you might be considering leaving {plan_name}. "
                    "Before you go, we'd like to offer you {discount_pct}% off for {discount_months} months "
                    "to give us another chance.\n\n"
                    "Our team is constantly improving, and we'd love to have you experience "
                    "the latest updates to {favorite_feature}.\n\n"
                    "Accept offer: [OFFER_LINK]\n\n"
                    "Hope to see you soon,\nThe {plan_name} Team"
                ),
                "offer_type": "discount",
                "urgency": "low",
            },
        ],
        "sure_thing": [
            {
                "subject": "You're amazing, {name} — thank you for your loyalty",
                "body": (
                    "Hi {name},\n\n"
                    "We just wanted to take a moment to say THANK YOU. "
                    "Your {tenure_days}-day membership means the world to us, "
                    "and your engagement with {favorite_feature} inspires our whole team.\n\n"
                    "As a loyal {plan_name} member, you now have early access to our "
                    "newest features before anyone else. Explore them here: [FEATURE_LINK]\n\n"
                    "With gratitude,\nThe {plan_name} Team"
                ),
                "offer_type": "loyalty_reward",
                "urgency": "low",
            },
        ],
        "lost_cause": [
            {
                "subject": "Before you go, {name} — a final gift from us",
                "body": (
                    "Hi {name},\n\n"
                    "We're sorry to see you go. If you've decided to cancel {plan_name}, "
                    "we completely understand.\n\n"
                    "If there's anything we could have done better, we'd genuinely love to hear it: "
                    "[FEEDBACK_LINK]\n\n"
                    "As a parting gift, your account data and playlists will remain saved for "
                    "90 days should you decide to return.\n\n"
                    "Thank you for being with us,\nThe {plan_name} Team"
                ),
                "offer_type": "winback_prep",
                "urgency": "low",
            },
        ],
        "sleeping_dog": [
            {
                "subject": "No action needed, {name} — just a heads-up",
                "body": (
                    "Hi {name},\n\n"
                    "Your {plan_name} subscription is active and everything is in order. "
                    "When you're ready to explore new features, we'll be here.\n\n"
                    "Best,\nThe {plan_name} Team"
                ),
                "offer_type": "none",
                "urgency": "none",
            },
        ],
        "onboarding_at_risk": [
            {
                "subject": "Getting started with {plan_name}? We're here to help, {name}",
                "body": (
                    "Hi {name},\n\n"
                    "Welcome to {plan_name}! We noticed you haven't fully explored everything yet — "
                    "and we'd love to help you get the most out of your membership.\n\n"
                    "Here are 3 things to try today:\n"
                    "1. Explore {favorite_feature} — most members say it's their favorite\n"
                    "2. Set up your personal preferences for better recommendations\n"
                    "3. Invite a friend and get 1 month free!\n\n"
                    "Start exploring: [ONBOARDING_LINK]\n\n"
                    "We're rooting for you,\nThe {plan_name} Team"
                ),
                "offer_type": "referral",
                "urgency": "medium",
            },
        ],
    }


def generate_retention_message(
    user_segment: str,
    clv_tier: str,
    churn_risk: float,
    favorite_feature: str = "personalized recommendations",
    tenure_days: int = 90,
    plan_name: str = "Premium",
    user_name: str = "Valued Member",
    days_inactive: int = 0,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate a personalized retention message for a user.

    Args:
        user_segment: 'Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs'
        clv_tier: 'High', 'Medium', 'Low'
        churn_risk: float 0-1
        favorite_feature: user's most-used feature
        tenure_days: days since subscription start
        plan_name: subscription plan name
        user_name: user's display name
        days_inactive: days since last login
        seed: random seed for template selection

    Returns:
        dict with keys: subject, body, offer_type, discount_pct, urgency, segment_key
    """
    templates = get_retention_templates()

    # Map segment + CLV to template key
    segment_key_map = {
        ("Persuadables", "High"): "high_clv_persuadable",
        ("Persuadables", "Medium"): "medium_clv_persuadable",
        ("Persuadables", "Low"): "low_clv_persuadable",
        ("Sure Things", "High"): "sure_thing",
        ("Sure Things", "Medium"): "sure_thing",
        ("Sure Things", "Low"): "sure_thing",
        ("Lost Causes", "High"): "medium_clv_persuadable",
        ("Lost Causes", "Medium"): "lost_cause",
        ("Lost Causes", "Low"): "lost_cause",
        ("Sleeping Dogs", "High"): "sleeping_dog",
        ("Sleeping Dogs", "Medium"): "sleeping_dog",
        ("Sleeping Dogs", "Low"): "sleeping_dog",
    }

    # Lifecycle-based override for new users
    if tenure_days < 30:
        segment_key = "onboarding_at_risk"
    else:
        segment_key = segment_map = segment_key_map.get(
            (user_segment, clv_tier), "medium_clv_persuadable"
        )

    template_list = templates.get(segment_key, templates["medium_clv_persuadable"])

    rng = random.Random(seed)
    template = rng.choice(template_list)

    # Determine discount based on CLV and churn risk
    discount_map = {
        "High": {"high": 30, "medium": 20, "low": 15},
        "Medium": {"high": 20, "medium": 15, "low": 10},
        "Low": {"high": 15, "medium": 10, "low": 5},
    }
    risk_level = "high" if churn_risk > 0.6 else "medium" if churn_risk > 0.35 else "low"
    discount_pct = discount_map.get(clv_tier, {}).get(risk_level, 10)
    discount_months = 3 if clv_tier == "High" else 2 if clv_tier == "Medium" else 1

    # Fill template placeholders
    placeholders = {
        "name": user_name,
        "tenure_days": str(tenure_days),
        "plan_name": plan_name,
        "favorite_feature": favorite_feature,
        "discount_pct": str(discount_pct),
        "discount_months": str(discount_months),
        "churn_risk_label": f"{churn_risk:.0%}",
        "days_inactive": str(days_inactive),
        "clv_tier": clv_tier,
    }

    subject = template["subject"]
    body = template["body"]
    for key, value in placeholders.items():
        subject = subject.replace(f"{{{key}}}", value)
        body = body.replace(f"{{{key}}}", value)

    return {
        "subject": subject,
        "body": body,
        "offer_type": template["offer_type"],
        "discount_pct": discount_pct,
        "discount_months": discount_months,
        "urgency": template["urgency"],
        "segment_key": segment_key,
        "user_segment": user_segment,
        "clv_tier": clv_tier,
        "churn_risk": round(churn_risk, 3),
    }


def personalize_offer(
    clv_tier: str,
    uplift_segment: str,
    churn_probability: float,
) -> dict[str, Any]:
    """
    Determine the optimal intervention type based on CLV, uplift segment, and churn risk.

    Decision matrix:
    - High CLV + Persuadable → premium_offer (large discount + upgrade)
    - High CLV + Sure Thing  → loyalty_reward (no cost, just appreciation)
    - High CLV + Lost Cause  → win_back_aggressive (maximum effort)
    - Medium CLV + Persuadable → standard_offer (moderate discount)
    - Low CLV + Sure Thing  → no_action (they'll stay, save the cost)
    - Any + Sleeping Dog    → no_action (intervention hurts)
    """
    action_matrix: dict[tuple[str, str], dict] = {
        ("High", "Persuadables"): {
            "action": "premium_offer",
            "discount_pct": 30,
            "priority": "urgent",
            "channel": "email + sms",
            "estimated_save_prob": 0.65,
        },
        ("High", "Sure Things"): {
            "action": "loyalty_reward",
            "discount_pct": 0,
            "priority": "medium",
            "channel": "email",
            "estimated_save_prob": 0.85,
        },
        ("High", "Lost Causes"): {
            "action": "win_back_aggressive",
            "discount_pct": 40,
            "priority": "urgent",
            "channel": "email + sms + call",
            "estimated_save_prob": 0.25,
        },
        ("High", "Sleeping Dogs"): {
            "action": "no_action",
            "discount_pct": 0,
            "priority": "none",
            "channel": "none",
            "estimated_save_prob": 0.90,
        },
        ("Medium", "Persuadables"): {
            "action": "standard_offer",
            "discount_pct": 20,
            "priority": "high",
            "channel": "email",
            "estimated_save_prob": 0.50,
        },
        ("Medium", "Sure Things"): {
            "action": "light_touch",
            "discount_pct": 5,
            "priority": "low",
            "channel": "email",
            "estimated_save_prob": 0.80,
        },
        ("Medium", "Lost Causes"): {
            "action": "minimal_offer",
            "discount_pct": 10,
            "priority": "low",
            "channel": "email",
            "estimated_save_prob": 0.15,
        },
        ("Medium", "Sleeping Dogs"): {
            "action": "no_action",
            "discount_pct": 0,
            "priority": "none",
            "channel": "none",
            "estimated_save_prob": 0.75,
        },
        ("Low", "Persuadables"): {
            "action": "basic_offer",
            "discount_pct": 10,
            "priority": "medium",
            "channel": "email",
            "estimated_save_prob": 0.35,
        },
        ("Low", "Sure Things"): {
            "action": "no_action",
            "discount_pct": 0,
            "priority": "none",
            "channel": "none",
            "estimated_save_prob": 0.80,
        },
        ("Low", "Lost Causes"): {
            "action": "no_action",
            "discount_pct": 0,
            "priority": "none",
            "channel": "none",
            "estimated_save_prob": 0.05,
        },
        ("Low", "Sleeping Dogs"): {
            "action": "no_action",
            "discount_pct": 0,
            "priority": "none",
            "channel": "none",
            "estimated_save_prob": 0.70,
        },
    }

    key = (clv_tier, uplift_segment)
    offer = action_matrix.get(key, {
        "action": "standard_offer",
        "discount_pct": 15,
        "priority": "medium",
        "channel": "email",
        "estimated_save_prob": 0.40,
    }).copy()

    offer["clv_tier"] = clv_tier
    offer["uplift_segment"] = uplift_segment
    offer["churn_probability"] = round(churn_probability, 3)
    return offer


def compute_campaign_roi(
    users_df: pd.DataFrame,
    uplift_scores: pd.Series,
    intervention_cost_per_user: float = 5.0,
    avg_monthly_revenue: float = 15.0,
    months_retained: int = 6,
    treatment_budget_pct: float = 0.20,
) -> dict[str, Any]:
    """
    Compute expected ROI of retention campaign.

    ROI = (retained_revenue - intervention_costs) / intervention_costs

    Args:
        users_df: must contain 'churn_flag', optionally 'clv', 'uplift_segment'
        uplift_scores: per-user uplift scores
        intervention_cost_per_user: cost to deliver one retention message
        avg_monthly_revenue: average monthly revenue per user
        months_retained: how many months a retained user stays
        treatment_budget_pct: fraction of at-risk users to target
    """
    n_users = len(users_df)
    churn_rate = float(users_df["churn_flag"].mean()) if "churn_flag" in users_df.columns else 0.25
    n_at_risk = int(n_users * churn_rate)

    # Sort by uplift score, take top treatment_budget_pct as targets
    sorted_uplift = uplift_scores.sort_values(ascending=False)
    n_targeted = max(1, int(n_at_risk * treatment_budget_pct))
    target_idx = sorted_uplift.index[:n_targeted]

    target_uplift = uplift_scores.loc[target_idx]
    avg_uplift = float(target_uplift.mean())

    # Expected saves = n_targeted * avg_uplift (incremental response)
    expected_saves = max(0, n_targeted * avg_uplift)

    # Revenue retained
    retained_revenue = expected_saves * avg_monthly_revenue * months_retained

    # Intervention costs
    total_cost = n_targeted * intervention_cost_per_user

    net_benefit = retained_revenue - total_cost
    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0.0

    # By segment breakdown
    segment_breakdown: dict[str, dict] = {}
    if "uplift_segment" in users_df.columns:
        for seg in users_df["uplift_segment"].unique():
            seg_mask = users_df["uplift_segment"] == seg
            seg_n = int(seg_mask.sum())
            seg_uplift_vals = uplift_scores[users_df.index[seg_mask]]
            seg_avg_uplift = float(seg_uplift_vals.mean()) if len(seg_uplift_vals) > 0 else 0.0
            segment_breakdown[str(seg)] = {
                "n_users": seg_n,
                "avg_uplift": round(seg_avg_uplift, 4),
                "expected_saves": round(seg_n * max(0, seg_avg_uplift), 1),
                "intervention_cost": round(seg_n * intervention_cost_per_user, 2),
            }

    result = {
        "n_total_users": n_users,
        "n_at_risk": n_at_risk,
        "n_targeted": n_targeted,
        "churn_rate": round(churn_rate, 4),
        "avg_uplift_targeted": round(avg_uplift, 4),
        "expected_saves": round(expected_saves, 1),
        "retained_revenue_usd": round(retained_revenue, 2),
        "intervention_cost_usd": round(total_cost, 2),
        "net_benefit_usd": round(net_benefit, 2),
        "roi_pct": round(roi, 2),
        "break_even_save_rate": round(total_cost / (avg_monthly_revenue * months_retained * n_targeted)
                                       if n_targeted > 0 else 0, 4),
        "segment_breakdown": segment_breakdown,
    }

    logger.info(
        f"Campaign ROI: {roi:.1f}%, net_benefit=${net_benefit:,.2f}, "
        f"expected_saves={expected_saves:.0f} users"
    )
    return result
