
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
OLLAMA_TIMEOUT  = getattr(_config, "OLLAMA_TIMEOUT", 300)


class QwenClient:

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

    # def _run_generation(self, prompt: str, temperature: float) -> Optional[str]:
    #     if not self._ensure_session():
    #         return None
    #     try:
    #         full_prompt = f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    #         resp = self._session.post(
    #             f"{self._base_url}/api/generate",
    #             json={
    #                 "model": self._model_name,
    #                 "prompt": prompt,
    #                 "system": SYSTEM_INSTRUCTION,
    #                 "stream": False,
    #                 "options": {
    #                     "temperature": temperature,
    #                     "num_predict": self.max_new_tokens,
    #                 },
    #             },
    #             timeout=self.timeout,
    #         )
    #         resp.raise_for_status()
    #         data = resp.json()
    #         text = (data.get("response") or "").strip()
    #         return text or None
    #
    #     except Exception as exc:
    #         log.error("QwenClient: Ollama generation failed — %s", exc)
    #         return None

    def _run_generation(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        if not self._ensure_session():
            return None
        try:
            # Qwen အတွက် Format ကို သေချာသတ်မှတ်ပေးပါ
            formatted_prompt = (
                f"<|im_start|>system\n"
                f"You are a helpful customer service assistant for Wonderami Smart Loan. "
                f"Answer the user's question clearly, politely, and accurately using ONLY the provided context.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            resp = self._session.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": formatted_prompt,
                    "raw": True,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Hallucination နည်းအောင် 0.1 ထားပါ
                        "top_p": 0.8,
                        "stop": ["<|im_start|>", "<|im_end|>"], # Prompt ပြန်မထွက်အောင် ပိတ်ပါ
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("response") or "").strip()
        except Exception as exc:
            log.error("QwenClient failed: %s", exc)
            return None

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