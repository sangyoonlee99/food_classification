# common/message_contract.py
from typing import List, Dict, Any
from pydantic import BaseModel


class MessageCard(BaseModel):
    title: str
    summary: str
    highlights: List[str]
    tips: List[str]
    meta: Dict[str, Any] = {}
