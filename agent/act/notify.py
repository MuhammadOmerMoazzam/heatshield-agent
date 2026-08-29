"""Notify action: posts a message to the configured Slack webhook."""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()


def notify_slack(message: str, webhook_url: str | None = None) -> str | None:
    """POST `message` to the Slack webhook. Returns "slack" (for
    Decision.notified_channel) on success, or None if no webhook is
    configured -- a dev/test environment shouldn't crash the loop over a
    missing SLACK_WEBHOOK_URL.
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return None
    httpx.post(url, json={"text": message}, timeout=10.0)
    return "slack"
