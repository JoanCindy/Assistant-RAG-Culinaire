from __future__ import annotations

import os

import httpx
from openai import OpenAI


def make_openai_client() -> OpenAI:
    ignore_proxy = os.getenv("OPENAI_IGNORE_PROXY", "1") == "1"
    if ignore_proxy:
        return OpenAI(http_client=httpx.Client(trust_env=False))
    return OpenAI()
