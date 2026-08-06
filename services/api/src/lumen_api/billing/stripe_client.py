"""Stripe Checkout — the one place this codebase talks to Stripe.

Stripe is a downstream consumer of `usage_records` (ADR-0004 §4), never the
source of truth for what an org is entitled to: nothing in the app reads a
plan's entitlements from Stripe, only from `subscriptions`/`plans`. A Stripe
outage delays the ability to *upgrade*, it does not change what's enforced.

Deliberately not built in this pass: the webhook that syncs a subscription's
status back from Stripe (`subscription.updated`/`deleted`). That is a
genuinely separate piece — its own endpoint, signature verification, and a
publicly reachable URL to receive it, none of which can be verified against
this local setup. Checkout creation (the forward path) is real; the backward
sync is the next thing to build once there is a URL to point Stripe at.
"""

from __future__ import annotations

import uuid

from lumen_api.settings import get_settings


def checkout_url_for_plan_change(org_id: uuid.UUID, plan_code: str, price_cents: int) -> str | None:
    """A real Stripe Checkout URL, or None when Stripe isn't configured or the
    plan is free (nothing to check out for)."""
    settings = get_settings()
    if not settings.has_stripe or price_cents <= 0:
        return None

    import stripe

    stripe.api_key = settings.stripe_secret_key.get_secret_value()
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": price_cents,
                    "recurring": {"interval": "month"},
                    "product_data": {"name": f"Lumen — {plan_code.title()}"},
                },
                "quantity": 1,
            }
        ],
        success_url=f"{settings.app_base_url}/app?checkout=success",
        cancel_url=f"{settings.app_base_url}/app?checkout=cancelled",
        client_reference_id=str(org_id),
        metadata={"org_id": str(org_id), "plan_code": plan_code},
    )
    return session.url
