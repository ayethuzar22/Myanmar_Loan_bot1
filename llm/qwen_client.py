"""
llm/qwen_client.py — Local Ollama-backed Qwen client.

Matches the GeminiClient interface exactly (is_available / generate /
generate_raw) so RAGPipeline and AutonomousLearningFilter work with either
backend interchangeably.

CHANGE FROM PREVIOUS VERSION
-----------------------------
No more `transformers` / `torch` / model downloads / GPU VRAM detection.
This now sends HTTP requests to a locally-running Ollama server
(http://localhost:11434 by default), which is already serving the
`qwen2.5:1.5b` model you pulled with `ollama pull qwen2.5:1.5b`.

Requirements:
    - Ollama must be running locally (`ollama serve`, or it's already
      running as a background service after install).
    - The model must be pulled once: `ollama pull qwen2.5:1.5b`
    - `pip install requests` (only new dependency — no torch/transformers)

REQUIRED CONFIG (add to config.py)
-------------------------------------
    OLLAMA_BASE_URL — defaults to "http://localhost:11434" if not set
    OLLAMA_MODEL     — defaults to "qwen2.5:1.5b" if not set
    QWEN_MAX_NEW_TOKENS — reused from before (maps to Ollama's num_predict)
    QWEN_TEMPERATURE    — reused from before
    OLLAMA_TIMEOUT       — optional, defaults to 120s (CPU generation on a
                            1.5B model is slower than a cloud API, so this
                            is more generous than a typical HTTP timeout)
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import config as _config
from config import QWEN_MAX_NEW_TOKENS, QWEN_TEMPERATURE, log
from prompts.system_prompt import SYSTEM_INSTRUCTION

# Optional config — fall back to sane defaults if you haven't added these
# to config.py yet, so this module never hard-fails on a missing constant.
OLLAMA_BASE_URL = getattr(_config, "OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = getattr(_config, "OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT  = getattr(_config, "OLLAMA_TIMEOUT", 120)


class QwenClient:
    """
    Thread-safe wrapper around a local Ollama server. Mirrors GeminiClient's
    public surface:
        is_available() -> bool
        generate(prompt) -> Optional[str]
        generate_raw(prompt, temperature=None) -> Optional[str]

    The `requests.Session` is built lazily on first use and connectivity is
    verified with a cheap GET before any generation call, so importing or
    constructing QwenClient never fails just because Ollama isn't running
    yet — that failure is deferred to first real use and reported the same
    way a Gemini/cloud outage would be (caller falls back to local KB /
    no-info path).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
        temperature: float = QWEN_TEMPERATURE,
        timeout: float = OLLAMA_TIMEOUT,
    ) -> None:
        self._model_name = model_name or OLLAMA_MODEL
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._session: Optional[Any] = None
        self._lock: threading.Lock = threading.Lock()
        self._init_failed: bool = False  # sticky — don't retry a hard failure every call

    # ── Public interface (mirrors GeminiClient) ─────────────────────────────

    def is_available(self) -> bool:
        return self._ensure_session()

    def generate(self, prompt: str) -> Optional[str]:
        return self._run_generation(prompt, self.temperature)

    def generate_raw(self, prompt: str, temperature: Optional[float] = None) -> Optional[str]:
        return self._run_generation(
            prompt, self.temperature if temperature is None else temperature
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_generation(self, prompt: str, temperature: float) -> Optional[str]:
        if not self._ensure_session():
            return None
        try:
            resp = self._session.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "system": SYSTEM_INSTRUCTION,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": self.max_new_tokens,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            return text or None

        except Exception as exc:
            log.error("QwenClient: Ollama generation failed — %s", exc)
            return None

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if self._init_failed:
            return False
        with self._lock:
            if self._session is not None:
                return True
            if self._init_failed:
                return False
            try:
                self._init_session()
                return True
            except Exception as exc:
                log.error(
                    "QwenClient: Ollama connection failed — %s. Is `ollama "
                    "serve` running, and did you `ollama pull %s`? Falling "
                    "back as unavailable; caller should treat this like "
                    "Gemini being down (local KB fallback / no-info path).",
                    exc, self._model_name,
                )
                self._init_failed = True
                return False

    def _init_session(self) -> None:
        import requests  # local import: don't require `requests` at module import time

        session = requests.Session()
        # Cheap connectivity check — Ollama's root path returns a plain
        # "Ollama is running" string when the server is up. This fails
        # fast (connection refused) if Ollama isn't started, instead of
        # waiting for the much slower /api/generate call to time out.
        probe = session.get(self._base_url, timeout=5)
        probe.raise_for_status()

        self._session = session
        log.info(
            "QwenClient: Ollama session ready — model='%s', base_url='%s'.",
            self._model_name, self._base_url,
        )