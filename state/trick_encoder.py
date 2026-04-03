# state/trick_encoder.py
import numpy as np
from core.card import Card
from core.rules import detect_move_type
from core.move_type import MoveType

# 7 type bits + 1 no-trick + rank/suit info + extras = 18 dims
TRICK_VECTOR_SIZE = 18


def encode_trick(cards: list[Card] | None) -> np.ndarray:
    """
    Encode current trick on the table.

    Layout:
    [0]     no trick (free to play anything)
    [1-6]   one-hot move type: SINGLE, PAIR, TRIPLE, STRAIGHT, FOUR_OF_KIND, DOUBLE_STRAIGHT
    [7]     main rank / 15.0
    [8-11]  one-hot highest suit (1..4 → idx 0..3)
    [12]    trick length / 13.0
    [13]    is_two (heo) flag
    [14]    is beateable by higher same-type (rank < 15) — not a two
    [15]    relative strength (rank / 15.0 duplicated for emphasis)
    [16]    is special (FOUR_OF_KIND or DOUBLE_STRAIGHT)
    [17]    padding
    """

    vec = np.zeros(TRICK_VECTOR_SIZE, dtype=np.float32)

    # no trick
    if not cards:
        vec[0] = 1.0
        return vec

    move_type = detect_move_type(cards)

    type_map = {
        MoveType.SINGLE:          0,
        MoveType.PAIR:            1,
        MoveType.TRIPLE:          2,
        MoveType.STRAIGHT:        3,
        MoveType.FOUR_OF_KIND:    4,
        MoveType.DOUBLE_STRAIGHT: 5,
    }

    if move_type in type_map:
        vec[1 + type_map[move_type]] = 1.0

    # Main rank (max rank in trick) — normalized by 15 (max rank = 2 = 15)
    max_rank = max(c.rank for c in cards)
    vec[7] = max_rank / 15.0

    # Highest card suit (suit 1..4 → index 0..3)
    highest = max(cards, key=lambda c: (c.rank, c.suit))
    vec[8 + (highest.suit - 1)] = 1.0   # vec[8]..vec[11]

    # Trick length
    vec[12] = len(cards) / 13.0

    # Is two (heo)
    vec[13] = 1.0 if max_rank == 15 else 0.0

    # Is beatable by normal same-type card (not a two)
    vec[14] = 0.0 if max_rank == 15 else 1.0

    # Relative strength (emphasis feature)
    vec[15] = max_rank / 15.0

    # Is special combo (四条 or 双顺)
    vec[16] = 1.0 if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT] else 0.0

    return vec
