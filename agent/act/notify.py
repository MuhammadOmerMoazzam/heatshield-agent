"""Notify action: posts a message to the configured Slack webhook."""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()


def notify_slack(message: str, webhook_url: str | None = None) -> str | None:
    """POST `message` to the Slack webhook. Returns "slack" (for
    Decision.notified_channel) only once Slack actually accepted it --
    raises on a non-2xx response rather than reporting success anyway, so
    the audit trail never records a notification that didn't go out.
    Returns None if no webhook is configured -- a dev/test environment
    shouldn't crash the loop over a missing SLACK_WEBHOOK_URL.
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return None
    resp = httpx.post(url, json={"text": message}, timeout=10.0)
    resp.raise_for_status()
    return "slack"
