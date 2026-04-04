# core/card.py
from dataclasses import dataclass

# rank: 3 → 15 (2 = 15)
# suit: 1=♠, 2=♣, 3=♦, 4=♥
# id  = rank * 10 + suit

RANK_INT_TO_STR = {r: str(r) for r in range(3, 15)}
RANK_INT_TO_STR[15] = "2"

SUIT_INT_TO_STR = {1: "♠", 2: "♣", 3: "♦", 4: "♥"}

# Rank string → int (dùng cho backward compat khi parse)
RANK_STR_TO_INT = {v: k for k, v in RANK_INT_TO_STR.items()}
RANK_STR_TO_INT.update({"J": 11, "Q": 12, "K": 13, "A": 14})

SUIT_STR_TO_INT = {v: k for k, v in SUIT_INT_TO_STR.items()}


@dataclass(frozen=True)
class Card:
    rank: int  # 3 → 15  (2 = 15)
    suit: int  # 1=♠, 2=♣, 3=♦, 4=♥

    @property
    def id(self) -> int:
        """ID duy nhất: rank*10 + suit. Vd: 3♠=31, 2♥=154"""
        return self.rank * 10 + self.suit

    # ── backward-compat aliases ──────────────────────────────────
    @property
    def rank_value(self) -> int:
        return self.rank

    @property
    def suit_value(self) -> int:
        return self.suit

    @property
    def card_id(self) -> int:
        """Alias của id (dùng trong hand/discard encoder)"""
        return self.id

    # ── helpers ──────────────────────────────────────────────────
    def __str__(self) -> str:
        return f"{RANK_INT_TO_STR.get(self.rank, '?')}{SUIT_INT_TO_STR.get(self.suit, '?')}"

    # ── factory methods ──────────────────────────────────────────
    @classmethod
    def from_id(cls, card_id: int) -> "Card":
        """Parse từ id = rank*10+suit"""
        suit = card_id % 10
        rank = card_id // 10
        return cls(rank=rank, suit=suit)

    @classmethod
    def from_old_ints(cls, rank_int: int, suit_int: int) -> "Card":
        """
        Backward compat: convert từ format cũ (rank 0-12, suit 0-3).
        3→3, 4→4, ..., A→14, 2→15
        suit: 0→1 (♠), 1→2 (♣), 2→3 (♦), 3→4 (♥)
        """
        OLD_RANK_MAP = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        rank = OLD_RANK_MAP[rank_int]
        suit = suit_int + 1
        return cls(rank=rank, suit=suit)
