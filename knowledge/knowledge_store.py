"""
knowledge/knowledge_store.py — KnowledgeStore class.

Thread-safe loader, validator, and in-process cache for loan.json.
Moved verbatim from rag1.py.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

from config import RAW_JSON_PATH, log
from models.loan_document import LoanDocument
from utils.text_utils import clean_text, detect_language


class KnowledgeStore:
    """
    Thread-safe loader, validator, and in-process cache for loan.json.

    Only records where active==true and all REQUIRED_FIELDS are non-empty
    are kept.  A threading.RLock guards all mutable state so concurrent
    Django WSGI workers never corrupt the in-memory document list.

    The parallel ``_cleaned_questions`` list enables O(n) exact matching
    without re-cleaning on every query call.
    """

    REQUIRED_FIELDS: tuple[str, ...] = (
        "id", "category", "topic", "language", "question", "answer",
    )

    def __init__(self, json_path: str = RAW_JSON_PATH) -> None:
        self.json_path: str                 = json_path
        self._documents: list[LoanDocument] = []
        self._cleaned_q: list[str]          = []
        self._loaded: bool                  = False
        self._lock: threading.RLock         = threading.RLock()

    # ── Public interface ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Parse loan.json and populate internal caches.  Idempotent."""
        with self._lock:
            if self._loaded:
                return
            self._load_unlocked()

    def reload(self) -> None:
        """Force a fresh load from disk (e.g. after append_and_save)."""
        with self._lock:
            self._documents = []
            self._cleaned_q = []
            self._loaded    = False
            self._load_unlocked()

    @property
    def documents(self) -> list[LoanDocument]:
        """Lazy-load on first access; return cached list thereafter."""
        if not self._loaded:
            self.load()
        return self._documents

    def find_exact(self, cleaned_query: str) -> Optional[LoanDocument]:
        """
        Return the first document whose cleaned question equals cleaned_query,
        or None.  Uses the pre-built parallel list — no per-call re-cleaning.
        """
        if not self._loaded:
            self.load()
        try:
            return self._documents[self._cleaned_q.index(cleaned_query)]
        except ValueError:
            return None

    def append_and_save(
        self,
        question: str,
        answer: str,
        category: str = "self_learned",
    ) -> bool:
        """
        Atomically append a new entry to loan.json.

        Returns True if the entry was written, False on duplicate.
        Uses an atomic os.replace() for crash-safe writes on POSIX systems.
        Holds self._lock for the entire read-modify-write cycle.
        """
        with self._lock:
            cleaned_q = clean_text(question)
            if cleaned_q in self._cleaned_q:
                log.info("KnowledgeStore.append_and_save: duplicate — skipping.")
                return False

            new_id = max((d.id for d in self._documents), default=0) + 1
            new_entry: dict[str, Any] = {
                "id":             new_id,
                "category":       category,
                "topic":          "Self-Learned",
                "language":       detect_language(question),
                "question":       question.strip(),
                "aliases":        [],
                "keywords":       [],
                "answer":         answer.strip(),
                "related_topics": [],
                "source":         "autonomous_learning",
                "active":         True,
                "last_updated":   time.strftime("%Y-%m-%d"),
            }

            try:
                with open(self.json_path, "r", encoding="utf-8-sig") as fh:
                    database: list[dict[str, Any]] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                database = []

            database.append(new_entry)

            tmp = self.json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(database, fh, ensure_ascii=False, indent=4)
            os.replace(tmp, self.json_path)  # atomic on POSIX

            log.info("KnowledgeStore: entry id=%d saved.", new_id)
            # Invalidate cache — next .documents access triggers reload
            self._documents = []
            self._cleaned_q = []
            self._loaded    = False
            return True

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_unlocked(self) -> None:
        """Must be called with self._lock held."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"loan.json not found at: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8-sig") as fh:
            try:
                raw: list[Any] = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {self.json_path}: {exc}"
                ) from exc

        docs:    list[LoanDocument] = []
        cleaned: list[str]         = []
        n_inactive = n_invalid = 0

        for item in raw:
            if not isinstance(item, dict):
                n_invalid += 1
                continue
            if not item.get("active", True):
                n_inactive += 1
                continue
            if not all(item.get(f) for f in self.REQUIRED_FIELDS):
                log.warning(
                    "KnowledgeStore: skipping id=%s — missing required fields.",
                    item.get("id", "?"),
                )
                n_invalid += 1
                continue
            try:
                doc = LoanDocument(
                    id=int(item["id"]),
                    category=str(item["category"]).strip(),
                    topic=str(item["topic"]).strip(),
                    language=str(item.get("language", "my")).strip(),
                    question=str(item["question"]).strip(),
                    aliases=tuple(str(a) for a in item.get("aliases", [])),
                    keywords=tuple(str(k) for k in item.get("keywords", [])),
                    answer=str(item["answer"]).strip(),
                    related_topics=tuple(
                        str(r) for r in item.get("related_topics", [])
                    ),
                    source=str(item.get("source", "loan.json")),
                )
            except (TypeError, ValueError) as exc:
                log.warning(
                    "KnowledgeStore: skipping id=%s — %s",
                    item.get("id", "?"), exc,
                )
                n_invalid += 1
                continue

            docs.append(doc)
            cleaned.append(clean_text(doc.question))

        self._documents = docs
        self._cleaned_q = cleaned
        self._loaded    = True
        log.info(
            "KnowledgeStore: loaded=%d inactive=%d invalid=%d path=%s",
            len(docs), n_inactive, n_invalid, self.json_path,
        )