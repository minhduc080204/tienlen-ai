# state/hand_encoder.py
import numpy as np
from collections import Counter
from core.card import Card

# 52 binary (one card per slot) + 13 rank-group stats = 65 dims
HAND_BINARY_SIZE = 52
HAND_STATS_SIZE  = 13
HAND_VECTOR_SIZE = HAND_BINARY_SIZE + HAND_STATS_SIZE  # 65


# Map card.id → index 0..51
# Ranks 3..14 × suits 1..4 = 48 cards, rank 15 × suits 1..4 = 4 cards → 52 total
def _card_to_index(card: Card) -> int:
    """
    rank 3..14 → index (rank-3)*4 + (suit-1)   [ 0..47 ]
    rank 15    → index 48 + (suit-1)             [ 48..51 ]
    """
    if card.rank == 15:
        return 48 + (card.suit - 1)
    return (card.rank - 3) * 4 + (card.suit - 1)


def encode_hand(hand: list[Card]) -> np.ndarray:
    vec = np.zeros(HAND_VECTOR_SIZE, dtype=np.float32)

    rank_counter = Counter()

    for card in hand:
        idx = _card_to_index(card)
        vec[idx] = 1.0
        rank_counter[card.rank] += 1

    # ---- Rank-group stats (13 dims) ----
    # Slot 0: số lá / 13
    vec[52] = len(hand) / 13.0

    # Slot 1-4: số heo (rank=15) 0..4 → /4
    num_twos = rank_counter.get(15, 0)
    vec[53] = num_twos / 4.0

    # Slot 5: có tứ quý không
    vec[54] = 1.0 if any(v == 4 for v in rank_counter.values()) else 0.0

    # Slot 6: số đôi (không tính heo)
    num_pairs = sum(1 for r, v in rank_counter.items() if v >= 2 and r != 15)
    vec[55] = min(num_pairs, 6) / 6.0

    # Slot 7: số bộ ba
    num_triples = sum(1 for v in rank_counter.values() if v >= 3)
    vec[56] = min(num_triples, 4) / 4.0

    # Slot 8: bài mạnh (rank >= 13 = K,A,2) / 4
    high_cards = sum(1 for c in hand if c.rank >= 13)
    vec[57] = min(high_cards, 4) / 4.0

    # Slot 9: bài trung (rank 9-12 = 9,10,J,Q) / 4
    mid_cards = sum(1 for c in hand if 9 <= c.rank <= 12)
    vec[58] = min(mid_cards, 4) / 4.0

    # Slot 10: bài nhỏ (rank 3-8) / 6
    low_cards = sum(1 for c in hand if c.rank <= 8)
    vec[59] = min(low_cards, 6) / 6.0

    # Slot 11: có thể đánh sảnh ≥3 không
    unique_ranks = sorted(r for r in rank_counter.keys() if r != 15)
    has_straight = False
    for i in range(len(unique_ranks) - 2):
        if unique_ranks[i] + 1 == unique_ranks[i+1] and unique_ranks[i+1] + 1 == unique_ranks[i+2]:
            has_straight = True
            break
    vec[60] = 1.0 if has_straight else 0.0

    # Slot 12: rank trung bình / 15
    if hand:
        avg_rank = sum(c.rank for c in hand) / len(hand)
        vec[61] = avg_rank / 15.0
    else:
        vec[61] = 0.0

    # Slot 13 (offset 64): padding
    # (vec[62] và vec[63] để 0)

    return vec
