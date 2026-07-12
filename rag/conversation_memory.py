"""
Conversation memory for loan chatbot.

Stores:
- chat history
- extracted entities
- current conversation context
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
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