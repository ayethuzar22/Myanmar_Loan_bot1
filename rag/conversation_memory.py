from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from rag.state_machine import LoanStage

@dataclass
class ChatMessage:

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda:
        datetime.now().isoformat()
    )

class ConversationMemory:


    def __init__(self):

        self.messages: List[ChatMessage] = []

        self.entities: Dict[str, Any] = {}
        self.stage: "LoanStage" = LoanStage.START

        # ── Added for rag/dialogue_manager.py ──────────────────────────
        # Optional, additive-only fields. Any existing code that never
        # references these two attributes behaves exactly as before.
        # They track a clarifying question the dialogue manager itself
        # asked, so the NEXT short reply ("Yes"/"Farmer"/etc.) can be
        # resolved against it instead of being treated as a fresh,
        # under-specified query.
        self.pending_clarify_query: Optional[str] = None
        self.pending_clarify_type: Optional[str] = None

    def add_user_message(self, text:str):

        self.messages.append(
            ChatMessage(
                role="user",
                content=text
            )
        )

    def add_bot_message(self,text:str):

        self.messages.append(
            ChatMessage(
                role="assistant",
                content=text
            )
        )

    def update_entities(self,new_entities:dict):

        self.entities.update(new_entities)

    def get_recent_history(self,limit=5):

        return [
            {
                "role":m.role,
                "content":m.content
            }
            for m in self.messages[-limit:]
        ]
    def get_context_text(self):

        return "\n".join(
            [
                f"{m.role}: {m.content}"
                for m in self.messages[-10:]
            ]
        )

    def clear(self):
        self.messages.clear()
        self.entities.clear()
        self.pending_clarify_query = None
        self.pending_clarify_type = None