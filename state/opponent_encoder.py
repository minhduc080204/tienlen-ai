# state/opponent_encoder.py
import numpy as np


def encode_opponents(
    opponent_counts: list[int],
    player_id: int,
    num_players: int,
) -> np.ndarray:
    """
    Encode thông tin công khai của đối thủ
    """

    assert num_players in [2, 3, 4]
    assert len(opponent_counts) == num_players - 1

    vec = np.zeros(12, dtype=np.float32)

    # --------------------
    # số lá của từng đối thủ (tối đa 3)
    # --------------------
    for i, count in enumerate(opponent_counts):
        vec[i] = count / 13.0

    # --------------------
    # min / max bài còn lại
    # --------------------
    vec[3] = min(opponent_counts) / 13.0
    vec[4] = max(opponent_counts) / 13.0

    # --------------------
    # có ai sắp hết bài không
    # --------------------
    vec[5] = 1.0 if any(c <= 2 for c in opponent_counts) else 0.0

    # --------------------
    # padding cho game < 4 người
    # --------------------
    # vec[6] → vec[11] để trống (0)

    return vec
