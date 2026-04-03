# core/card.py
from dataclasses import dataclass

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['♠', '♣', '♦', '♥']

RANK_TO_VALUE = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_VALUE = {s: i for i, s in enumerate(SUITS)}

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    @classmethod
    def from_ints(cls, rank_int: int, suit_int: int) -> "Card":
        """
        Khởi tạo Card trực tiếp từ int (rank: 0-12, suit: 0-3)
        Ví dụ: Card.from_ints(0, 0) -> Card(rank='3', suit='♠')
        """
        return cls(rank=RANKS[rank_int], suit=SUITS[suit_int])

    @property
    def rank_value(self) -> int:
        return RANK_TO_VALUE[self.rank]

    @property
    def suit_value(self) -> int:
        return SUIT_TO_VALUE[self.suit]

    @property
    def card_id(self) -> int:
        """
        ID từ 0 → 51
        """
        return self.rank_value * 4 + self.suit_value

    def __str__(self):
        return f"{self.rank}{self.suit}"
