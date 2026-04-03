# state/opponent_encoder.py
import numpy as np
from collections import Counter
from core.card import Card

# 30 dims total:
# [0-2]   card_count/13 per opponent (up to 3)
# [3-5]   flag: has_very_few (<=3) per opponent
# [6-8]   danger_score per opponent (1 - count/13)
# [9]     min opponent cards / 13
# [10]    max opponent cards / 13
# [11]    any opponent <= 2 cards (urgent)
# [12]    any opponent <= 1 card (critical)
# [13]    num opponents with <=3 cards / 3
# [14]    estimated twos remaining / 4
# [15]    estimated four-of-kind remaining (binary)
# [16]    estimated double-straight potential (binary)
# [17-19] relative rank pressure per opponent (ước tính sức mạnh)
# [20-22] pass flag estimate (đối thủ nhiều bài → likely holding big cards)
# [23]    opponent avg cards / 13
# [24-29] reserved / zero-padded

OPPONENT_VECTOR_SIZE = 30


def encode_opponents(
    opponent_counts: list[int],
    player_id: int,
    num_players: int,
    discard_pile: list[Card] | None = None,
) -> np.ndarray:
    """
    Encode rich opponent information.

    Args:
        opponent_counts: số lá còn lại của từng đối thủ (theo thứ tự relative)
        player_id: id của AI player
        num_players: tổng số người chơi
        discard_pile: bài đã đánh (dùng để ước tính bài mạnh còn lại)
    """

    assert num_players in [2, 3, 4]
    assert len(opponent_counts) == num_players - 1

    vec = np.zeros(OPPONENT_VECTOR_SIZE, dtype=np.float32)

    if not opponent_counts:
        return vec

    # --------------------
    # [0-2] Card count per opponent
    # --------------------
    for i, count in enumerate(opponent_counts):
        vec[i] = count / 13.0

    # --------------------
    # [3-5] Has very few cards (<=3) per opponent
    # --------------------
    for i, count in enumerate(opponent_counts):
        vec[3 + i] = 1.0 if count <= 3 else 0.0

    # --------------------
    # [6-8] Danger score per opponent (inverse of card count)
    # --------------------
    for i, count in enumerate(opponent_counts):
        vec[6 + i] = 1.0 - (count / 13.0)

    # --------------------
    # [9-10] Min / max opponent cards
    # --------------------
    vec[9]  = min(opponent_counts) / 13.0
    vec[10] = max(opponent_counts) / 13.0

    # --------------------
    # [11] Urgent: any opponent <= 2 cards
    # [12] Critical: any opponent <= 1 card
    # --------------------
    vec[11] = 1.0 if any(c <= 2 for c in opponent_counts) else 0.0
    vec[12] = 1.0 if any(c <= 1 for c in opponent_counts) else 0.0

    # --------------------
    # [13] Fraction of opponents with <=3 cards
    # --------------------
    few_count = sum(1 for c in opponent_counts if c <= 3)
    vec[13] = few_count / len(opponent_counts)

    # --------------------
    # [14-16] Ước tính bài mạnh còn lại (từ discard pile)
    # Nếu không có discard info, mặc định assume tất cả còn
    # --------------------
    if discard_pile:
        discarded_ranks = Counter(c.rank for c in discard_pile)

        # Ước tính heo còn lại
        twos_played = discarded_ranks.get(15, 0)
        vec[14] = max(0, 4 - twos_played) / 4.0

        # Tứ quý đã xuất hiện chưa
        any_fourofkind = any(v == 4 for v in discarded_ranks.values())
        vec[15] = 0.0 if any_fourofkind else 1.0

        # Tiềm năng đôi thông (ít bài đã ra → nhiều tiềm năng)
        discard_ratio = len(discard_pile) / 52.0
        vec[16] = 1.0 - discard_ratio
    else:
        # Không có info → assume worst case
        vec[14] = 1.0   # assume 4 twos still in (normalized)
        vec[15] = 1.0   # assume four-of-kind possible
        vec[16] = 1.0   # assume double-straight possible

    # --------------------
    # [17-19] Relative rank pressure (đối thủ ít bài → áp lực cao)
    # Dùng exponential: ít bài hơn = nguy hiểm hơn
    # --------------------
    for i, count in enumerate(opponent_counts):
        if count == 0:
            pressure = 1.0
        elif count <= 2:
            pressure = 0.9
        elif count <= 5:
            pressure = 0.6
        elif count <= 8:
            pressure = 0.3
        else:
            pressure = 0.1
        vec[17 + i] = pressure

    # --------------------
    # [20-22] Holding big cards estimate
    # Đối thủ nhiều bài → likely giữ bài lớn và pass (rate ước tính)
    # --------------------
    for i, count in enumerate(opponent_counts):
        vec[20 + i] = (count / 13.0) * 0.5  # proxy: nhiều bài → có thể có nhiều bài mạnh

    # --------------------
    # [23] Average opponent cards / 13
    # --------------------
    vec[23] = sum(opponent_counts) / (len(opponent_counts) * 13.0)

    # [24-29] zero-padded (future use)

    return vec
