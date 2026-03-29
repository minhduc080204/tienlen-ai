import torch
import numpy as np

from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.rules import get_legal_moves


def predict_action(
    agent,
    device,
    hand,
    opponent_counts,
    current_trick,
    player_id,
    num_players
):
    """
    Predict action_id từ trạng thái game
    """

    # ===== BUILD STATE =====
    state_vec = encode_state(
        hand=hand,
        opponent_counts=opponent_counts,
        current_trick=current_trick,
        player_id=player_id,
        num_players=num_players,
        discard_pile=[]
    )

    # ===== BUILD ACTION MASK =====
    legal_moves = get_legal_moves(hand, current_trick)

    action_mask = build_action_mask_from_legal_moves(
        legal_moves,
        ACTION_SPACE
    )

    # ===== TO TENSOR =====
    state_tensor = torch.from_numpy(
        np.array(state_vec, dtype=np.float32)
    ).unsqueeze(0).to(device)

    action_mask_tensor = torch.from_numpy(
        action_mask
    ).unsqueeze(0).to(device)

    # ===== PREDICT =====
    with torch.no_grad():
        action_id, _, _ = agent.act(state_tensor, action_mask_tensor)

    return int(action_id)