# state/state_encoder.py
import numpy as np

from core.card import Card
from state.hand_encoder import encode_hand, HAND_VECTOR_SIZE
from state.trick_encoder import encode_trick, TRICK_VECTOR_SIZE
from state.opponent_encoder import encode_opponents, OPPONENT_VECTOR_SIZE
from state.discard_encoder import encode_discard_pile, DISCARD_VECTOR_SIZE


def encode_state(
    hand: list[Card],
    discard_pile: list[Card],
    opponent_counts: list[int],
    current_trick: list[Card] | None,
    player_id: int,
    num_players: int,
) -> np.ndarray:
    """
    Encode toàn bộ trạng thái game cho RL agent.

    State vector layout:
    ─────────────────────────────────────────────────────────
    A  Hand           (65 dims)  — 52 binary + 13 stats
    B  Current Trick  (18 dims)  — type + rank + suit + extras
    C  Opponent Info  (30 dims)  — per-opp count, danger, estimated strength
    D  Discard Pile   (70 dims)  — binary + strategic stats
    ─────────────────────────────────────────────────────────
    Total: 65 + 18 + 30 + 70 = 183 dims
    """

    # A Hand (65)
    hand_vec = encode_hand(hand)

    # B Current Trick (18)
    trick_vec = encode_trick(current_trick)

    # C Opponent info (30)
    opponent_vec = encode_opponents(
        opponent_counts=opponent_counts,
        player_id=player_id,
        num_players=num_players,
        discard_pile=discard_pile,
    )

    # D Discard / Memory (70)
    discard_vec = encode_discard_pile(discard_pile)

    # CONCAT ALL
    state = np.concatenate([
        hand_vec,       # 65
        trick_vec,      # 18
        opponent_vec,   # 30
        discard_vec,    # 70
    ]).astype(np.float32)

    assert state.shape[0] == HAND_VECTOR_SIZE + TRICK_VECTOR_SIZE + OPPONENT_VECTOR_SIZE + DISCARD_VECTOR_SIZE, \
        f"STATE_DIM mismatch: got {state.shape[0]}"

    return state
