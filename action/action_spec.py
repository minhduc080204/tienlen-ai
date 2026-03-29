# action/action_spec.py
from dataclasses import dataclass
from typing import List
from core.move_type import MoveType

@dataclass(frozen=True)
class ActionSpec:
    move_type: MoveType
    length: int               # số lá dùng
    # ranks: List[int]          # rank_value (0–12)
