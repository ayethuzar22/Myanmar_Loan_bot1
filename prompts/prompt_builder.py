"""
prompts/prompt_builder.py — PromptBuilder class.

Assembles safe, context-bounded prompts for the Gemini API. Moved
verbatim from rag1.py. Kept in the prompts/ package (rather than rag/)
since its sole responsibility is prompt text assembly.
"""
from __future__ import annotations

from typing import Optional

from config import HISTORY_WINDOW
from models.chat_turn import ChatTurn
from models.retrieval_result import RetrievalResult


class PromptBuilder:
    """
    Assembles safe, context-bounded prompts for the Gemini API.

    Rules:
    - Only top-K retrieved snippets are included (never the full KB).
    - User input is placed inside a clearly delimited [USER QUESTION] block
      so the system instruction injection-warning applies to all content
      appearing after that delimiter.
    - History is truncated to HISTORY_WINDOW turns to bound token usage.
    """
    @staticmethod
    def build(
        user_question: str,
        results: list[RetrievalResult],
        chat_history: Optional[list[ChatTurn]] = None,
    ) -> str:
        """
        Build the full Gemini prompt string.

        Structure:
            [CONVERSATION HISTORY]          optional
            [RETRIEVED KNOWLEDGE BASE CONTEXT]
            --- Context N ---
            ...
            [USER QUESTION]
            <sanitised question>
        """
        parts: list[str] = []

        if chat_history:
            parts.append("[CONVERSATION HISTORY]")
            for turn in chat_history[-HISTORY_WINDOW:]:
                prefix = "User" if turn.role == "user" else "Assistant"
                parts.append(f"{prefix}: {turn.content}")
            parts.append("")

        if results:
            parts.append("[RETRIEVED KNOWLEDGE BASE CONTEXT]")
            parts.append(
                "Use ONLY the information below to answer. "
                "Do NOT invent policies, numbers, or facts."
            )
            parts.append("")
            for r in results:
                doc = r.document
                parts.append(f"--- Context {r.rank} (score={r.score:.3f}) ---")
                parts.append(f"Category : {doc.category}")
                parts.append(f"Topic    : {doc.topic}")
                parts.append(f"Question : {doc.question}")
                parts.append(f"Answer   : {doc.answer}")
                parts.append("")
        else:
            parts.append("[NO RELEVANT CONTEXT FOUND]")
            parts.append(
                "No matching knowledge base entries found. "
                "Inform the user politely that you cannot find the information."
            )
            parts.append("")

        parts.append(f"[USER QUESTION]\n{user_question}")
        return "\n".join(parts)