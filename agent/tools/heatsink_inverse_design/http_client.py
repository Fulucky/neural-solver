"""HTTP helpers shared by heatsink inverse-design tools."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


API_BASE_ENV = "HEATSINK_INFERENCE_API_URL"
DEFAULT_API_BASE = "http://127.0.0.1:8000"


def api_base(api_base_url: str | None = None) -> str:
    return (api_base_url or os.getenv(API_BASE_ENV) or DEFAULT_API_BASE).rstrip("/")


def post_json(
    path: str,
    payload: dict[str, Any],
    api_base_url: str | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    url = f"{api_base(api_base_url)}{path}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed: {exc.code} {exc.reason}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach heatsink inference API at {url}: {exc}") from exc
