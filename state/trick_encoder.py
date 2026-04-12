# state/trick_encoder.py
import numpy as np
from core.card import Card
from core.rules import detect_move_type
from core.move_type import MoveType

# 52 binary + 18 feature bits = 70 dims
TRICK_VECTOR_SIZE = 70


def _card_to_index(card: Card) -> int:
    """rank 3..14 -> 0..47, rank 15 -> 48..51"""
    if card.rank == 15:
        return 48 + (card.suit - 1)
    return (card.rank - 3) * 4 + (card.suit - 1)


def encode_trick(cards: list[Card] | None) -> np.ndarray:
    """
    Encode current trick on the table.

    Layout:
    [0-51]  binary: which cards are currently on table
    [52]    no trick (free to play anything)
    [53-58] one-hot move type: SINGLE, PAIR, TRIPLE, STRAIGHT, FOUR_OF_KIND, DOUBLE_STRAIGHT
    [59]    main rank / 15.0
    [60-63] one-hot highest suit (1..4 -> idx 0..3)
    [64]    trick length / 13.0
    [65]    is_two (heo) flag
    [66]    is beateable by higher same-type (rank < 15)
    [67]    relative strength (rank / 15.0)
    [68]    is special (FOUR_OF_KIND or DOUBLE_STRAIGHT)
    [69]    reserved
    """

    vec = np.zeros(TRICK_VECTOR_SIZE, dtype=np.float32)

    # 1. Binary bits (0-51)
    if cards:
        for card in cards:
            vec[_card_to_index(card)] = 1.0

    offset = 52

    # 2. No trick flag
    if not cards:
        vec[offset] = 1.0
        return vec

    # 3. Move Type Features
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
        vec[offset + 1 + type_map[move_type]] = 1.0

    # 4. Rank and Suit
    max_rank = max(c.rank for c in cards)
    vec[offset + 7] = max_rank / 15.0

    highest = max(cards, key=lambda c: (c.rank, c.suit))
    vec[offset + 8 + (highest.suit - 1)] = 1.0

    # 5. Metadata
    vec[offset + 12] = len(cards) / 13.0
    vec[offset + 13] = 1.0 if max_rank == 15 else 0.0
    vec[offset + 14] = 0.0 if max_rank == 15 else 1.0
    vec[offset + 15] = max_rank / 15.0
    vec[offset + 16] = 1.0 if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT] else 0.0

    return vec
