# state/state_encoder.py
import numpy as np

from core.card import Card
from state.hand_encoder import encode_hand
from state.trick_encoder import encode_trick
from state.opponent_encoder import encode_opponents
from state.discard_encoder import encode_discard_pile


def encode_state(
    hand: list[Card],
    discard_pile: list[Card],
    opponent_counts: list[int],
    current_trick: list[Card] | None,
    player_id: int,
    num_players: int,
) -> np.ndarray:
    """
    Encode toàn bộ trạng thái game cho RL agent
    """

    # --------------------
    # A️⃣ Hand (52)
    # --------------------
    hand_vec = encode_hand(hand)

    # --------------------
    # B️⃣ Current Trick (17)
    # --------------------
    trick_vec = encode_trick(current_trick)

    # --------------------
    # C️⃣ Opponent info (12)
    # --------------------
    opponent_vec = encode_opponents(
        opponent_counts=opponent_counts,
        player_id=player_id,
        num_players=num_players,
    )

    # --------------------
    # D️⃣ Discard / Memory (60)
    # --------------------
    discard_vec = encode_discard_pile(discard_pile)

    # --------------------
    # CONCAT ALL
    # --------------------
    state = np.concatenate([
        hand_vec,
        trick_vec,
        opponent_vec,
        discard_vec
    ]).astype(np.float32)

    return state
