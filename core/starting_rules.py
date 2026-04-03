# core/starting_rules.py
from core.card import Card


def find_starting_player(hands: list[list[Card]]) -> int:
    """
    Luật:
    - Có 3♠ (rank=3, suit=1, id=31) → người đó đi trước
    - Không có 3♠ → người có lá nhỏ nhất (rank thấp → suit thấp) đi trước
    """

    # 1️⃣ tìm 3♠ (smallest card in Tiến Lên)
    for pid, hand in enumerate(hands):
        for card in hand:
            if card.rank == 3 and card.suit == 1:  # 3♠ = id 31
                return pid

    # 2️⃣ không có 3♠ → tìm lá nhỏ nhất
    min_card: Card | None = None
    start_player = 0

    for pid, hand in enumerate(hands):
        for card in hand:
            if min_card is None:
                min_card = card
                start_player = pid
            else:
                if (
                    card.rank < min_card.rank or
                    (card.rank == min_card.rank and card.suit < min_card.suit)
                ):
                    min_card = card
                    start_player = pid

    return start_player
