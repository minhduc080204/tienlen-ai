# state/discard_encoder.py
import numpy as np
from collections import Counter
from core.card import Card

# 52 binary card slots + 18 strategic flags = 70 dims
DISCARD_BINARY_SIZE   = 52
DISCARD_STATS_SIZE    = 18
DISCARD_VECTOR_SIZE   = DISCARD_BINARY_SIZE + DISCARD_STATS_SIZE  # 70


def _card_to_index(card: Card) -> int:
    """Same mapping as hand_encoder"""
    if card.rank == 15:
        return 48 + (card.suit - 1)
    return (card.rank - 3) * 4 + (card.suit - 1)


def encode_discard_pile(discard_pile: list[Card]) -> np.ndarray:
    """
    Encode public memory of all played cards.

    Layout:
    [0-51]  binary: lá nào đã ra
    [52-55] 4 heo đã ra (rank=15, suit 1-4)
    [56]    tứ quý đã xuất hiện
    [57]    đôi thông ≥3 đã xuất hiện
    [58]    tỉ lệ bài đã đánh (len/52)
    [59]    còn heo chưa ra (normalized)
    [60]    bài nhỏ (rank 3-6) đã ra / 16
    [61]    bài trung dưới (rank 7-9) đã ra / 12
    [62]    bài trung trên (rank 10-12) đã ra / 12
    [63]    bài lớn (rank 13-14) đã ra / 8
    [64]    rank trung bình bài đã ra / 15
    [65]    số lần tứ quý đã chặt heo (heuristic: nếu twos ra sớm)
    [66]    bài nhỏ còn nhiều không (proxy: discard ratio of low cards)
    [67]    tổng rank bài đã ra / (52*15)  [normalized cumulative power played]
    [68-69] reserved (zero)
    """

    vec = np.zeros(DISCARD_VECTOR_SIZE, dtype=np.float32)

    if not discard_pile:
        return vec

    rank_counter = Counter(c.rank for c in discard_pile)

    # ========================
    # [0-51] Binary card slots
    # ========================
    for card in discard_pile:
        idx = _card_to_index(card)
        vec[idx] = 1.0

    offset = 52

    # ========================
    # [52-55] Heo đã ra (suit 1..4)
    # ========================
    twos = [c for c in discard_pile if c.rank == 15]
    for c in twos:
        vec[offset + (c.suit - 1)] = 1.0   # offset+0..offset+3

    # ========================
    # [56] Tứ quý đã xuất hiện
    # ========================
    vec[offset + 4] = 1.0 if any(v == 4 for v in rank_counter.values()) else 0.0

    # ========================
    # [57] Đôi thông ≥3 đã xuất hiện
    # ========================
    pairs = [r for r, v in rank_counter.items() if v >= 2 and r != 15]
    pairs.sort()
    found_ds = False
    for i in range(len(pairs) - 2):
        if pairs[i] + 1 == pairs[i + 1] and pairs[i + 1] + 1 == pairs[i + 2]:
            found_ds = True
            break
    vec[offset + 5] = 1.0 if found_ds else 0.0

    # ========================
    # [58] Tỉ lệ bài đã đánh
    # ========================
    vec[offset + 6] = len(discard_pile) / 52.0

    # ========================
    # [59] Còn heo chưa ra
    # ========================
    vec[offset + 7] = max(0, 4 - len(twos)) / 4.0

    # ========================
    # [60-63] Rank group counters
    # ========================
    low_played  = sum(1 for c in discard_pile if 3  <= c.rank <= 6)
    mid1_played = sum(1 for c in discard_pile if 7  <= c.rank <= 9)
    mid2_played = sum(1 for c in discard_pile if 10 <= c.rank <= 12)
    high_played = sum(1 for c in discard_pile if 13 <= c.rank <= 14)

    vec[offset + 8]  = low_played  / 16.0   # max 4 ranks × 4 suits
    vec[offset + 9]  = mid1_played / 12.0   # max 3 ranks × 4 suits
    vec[offset + 10] = mid2_played / 12.0
    vec[offset + 11] = high_played / 8.0    # max 2 ranks × 4 suits

    # ========================
    # [64] Rank trung bình bài đã ra / 15
    # ========================
    if discard_pile:
        avg_rank = sum(c.rank for c in discard_pile) / len(discard_pile)
        vec[offset + 12] = avg_rank / 15.0

    # ========================
    # [65] Bài nhỏ còn nhiều không (low cards discard ratio)
    # ========================
    max_low = 16  # 4 ranks × 4 suits
    vec[offset + 13] = low_played / max_low

    # ========================
    # [66] Tổng power bài đã ra (normalized)
    # ========================
    total_power = sum(c.rank for c in discard_pile)
    vec[offset + 14] = total_power / (52 * 15.0)

    # [67-69] reserved
    # vec[offset+15..17] stays 0

    return vec
