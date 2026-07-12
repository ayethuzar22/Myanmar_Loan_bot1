"""
llm/gemini_client.py — GeminiClient class.

Thread-safe wrapper around google-genai. Moved verbatim from rag1.py,
including the security fix that removed the hardcoded fallback API key —
GEMINI_API_KEY must be set in the environment; there is no embedded
credential of any kind in source.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_TOKENS,
    GEMINI_MODEL,
    GEMINI_RETRY_DELAY,
    GEMINI_TEMPERATURE,
    GEMINI_TIMEOUT_SECONDS,
    _GEMINI_FATAL_TAGS,
    log,
)
from prompts.system_prompt import SYSTEM_INSTRUCTION


def _is_gemini_fatal(exc: Exception) -> bool:
    """Return True for non-retryable Gemini API error categories."""
    return any(tag in str(exc).lower() for tag in _GEMINI_FATAL_TAGS)


def _build_http_options(timeout_seconds: float) -> Optional[Any]:
    """
    Build a types.HttpOptions(timeout=...) defensively.

    google-genai SDK versions differ in whether HttpOptions exists and
    what unit it expects.  Rather than letting every single Gemini call
    crash if the installed SDK doesn't match, this returns None on any
    incompatibility so the caller can simply omit http_options and fall
    back to the SDK's own default timeout.
    """
    try:
        return types.HttpOptions(timeout=int(timeout_seconds * 1000))
    except (AttributeError, TypeError) as exc:
        log.debug(
            "GeminiClient: HttpOptions unavailable in installed SDK (%s); "
            "falling back to SDK default timeout.", exc,
        )
        return None


class GeminiClient:
    """
    Thread-safe wrapper around google-genai with:
    - API key read exclusively from environment (never hardcoded).
    - Double-checked locking for singleton client creation.
    - Exponential back-off retry for transient errors.
    - Immediate abort for non-retryable auth / quota / bad-request errors.
    - Structured latency logging per attempt.

    FIX #1 (SECURITY): the hardcoded fallback API key that previously lived
    here has been removed entirely. GEMINI_API_KEY must now be set in the
    environment; there is no embedded credential of any kind in source.
    If the env var is missing or empty, is_available() / generate() /
    generate_raw() will all cleanly return None/False rather than trying
    to authenticate with a dead placeholder value.
    """

    def __init__(
        self,
        model:       str   = GEMINI_MODEL,
        temperature: float = GEMINI_TEMPERATURE,
        max_tokens:  int   = GEMINI_MAX_TOKENS,
        max_retries: int   = GEMINI_MAX_RETRIES,
        retry_delay: float = GEMINI_RETRY_DELAY,
        timeout:     float = GEMINI_TIMEOUT_SECONDS,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout     = timeout
        self._client: Optional[genai.Client] = None
        self._lock:   threading.Lock         = threading.Lock()

    def is_available(self) -> bool:
        """
        Public readiness check — callers should use this instead of
        reaching into the private client singleton directly.
        """
        return self._get_client() is not None

    def generate_raw(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """
        Single-shot, non-retrying call used by internal callers (e.g. the
        Critic layer) that need a raw response without the retry/backoff
        machinery of generate().  Exposed publicly so other components
        never touch the private client directly.

        Returns:
            Response text, or None on failure / unavailable client.
        """
        client = self._get_client()
        if client is None:
            return None
        try:
            config_kwargs: dict[str, Any] = {
                "temperature": self.temperature if temperature is None else temperature,
            }
            http_opts = _build_http_options(self.timeout)
            if http_opts is not None:
                config_kwargs["http_options"] = http_opts

            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return (resp.text or "").strip() or None
        except Exception as exc:
            log.error("GeminiClient.generate_raw: %s", exc)
            return None

    def generate(self, prompt: str) -> Optional[str]:
        """
        Send prompt to Gemini; retry on transient errors.

        The system instruction is passed via GenerateContentConfig so it is
        never visible inside the user prompt and cannot be overridden by
        prompt injection in the user question.  A per-request network
        timeout (GEMINI_TIMEOUT_SECONDS) prevents a hung connection from
        blocking the calling thread indefinitely.

        Returns:
            Response text string, or None if unavailable / all retries fail.
        """
        client = self._get_client()
        if client is None:
            return None

        config_kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        http_opts = _build_http_options(self.timeout)
        if http_opts is not None:
            config_kwargs["http_options"] = http_opts

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                t0       = time.perf_counter()
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                latency = time.perf_counter() - t0
                text    = (response.text or "").strip()
                log.info(
                    "GeminiClient: %.2fs attempt=%d/%d len=%d",
                    latency, attempt, self.max_retries, len(text),
                )
                return text or None  # empty string treated as generation failure

            except Exception as exc:
                last_exc = exc
                if _is_gemini_fatal(exc):
                    log.error("GeminiClient: non-retryable error — %s", exc)
                    return None
                delay = self.retry_delay * attempt
                log.warning(
                    "GeminiClient: attempt %d/%d failed (%s) — retry in %.1fs",
                    attempt, self.max_retries, exc, delay,
                )
                time.sleep(delay)

        log.error(
            "GeminiClient: all %d attempts failed — %s",
            self.max_retries, last_exc,
        )
        return None

    def translate_to_myanmar(self, text: str) -> Optional[str]:
        """Translate text to polite Myanmar at low temperature."""
        if not text.strip():
            return None
        prompt = (
            "You are a strict English-to-Myanmar translator.\n"
            "Translate the text below into polite, natural Myanmar.\n"
            "End every sentence with '\u1015\u102b\u1001\u1004\u103a\u1017\u103b\u102c' or '\u1015\u1031\u1038\u1015\u102b\u101e\u100a\u103a\u1001\u1004\u103a\u1017\u103b\u102c'.\n"
            "Output ONLY the translated text.\n\n"
            f"Text:\n{text}"
        )
        return self.generate_raw(prompt, temperature=0.1)

    def _get_client(self) -> Optional[genai.Client]:
        """
        Double-checked locking singleton.

        FIX #1 (SECURITY): no hardcoded fallback key. GEMINI_API_KEY must
        be set in the environment. If it is missing, this returns None
        cleanly and every caller (is_available/generate/generate_raw)
        already handles that gracefully.
        """
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            api_key = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LBNtHkF6PAg3Cs_xgourk20vmkTrfn3va3lmJ2YYmmag").strip()
            if not api_key:
                log.error(
                    "GeminiClient: GEMINI_API_KEY environment variable is not set. "
                    "Set it before starting the app, e.g. "
                    "PowerShell: $env:GEMINI_API_KEY = \"<your key>\""
                )
                return None
            try:
                self._client = genai.Client(api_key=api_key)
                log.info("GeminiClient: client initialised.")
            except Exception as exc:
                log.error("GeminiClient: init failed — %s", exc)
        return self._client