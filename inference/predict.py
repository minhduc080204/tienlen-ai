# inference/predict.py
import torch
import numpy as np

from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.rules import get_legal_moves
from core.action_executor import resolve_action


def predict_action(
    agent,
    device,
    hand,
    opponent_counts,
    current_trick,
    player_id,
    num_players,
    discard_pile=None,
):
    """
    Dự đoán action cho AI player.

    Args:
        discard_pile: list[Card] — bài đã ra trong ván (optional, default [])
    """
    if discard_pile is None:
        discard_pile = []

    # ── Build state vector ────────────────────────────────────────────────
    state_vec = encode_state(
        hand=hand,
        opponent_counts=opponent_counts,
        current_trick=current_trick,
        player_id=player_id,
        num_players=num_players,
        discard_pile=discard_pile,
    )

    # ── Build action mask ─────────────────────────────────────────────────
    legal_moves = get_legal_moves(hand=hand, current_trick=current_trick)

    action_mask = build_action_mask_from_legal_moves(
        legal_moves=legal_moves,
        action_space=ACTION_SPACE,
    )

    # ── To tensor ─────────────────────────────────────────────────────────
    state_tensor       = torch.tensor(state_vec, dtype=torch.float32, device=device)
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)

    # ── Predict ───────────────────────────────────────────────────────────
    with torch.no_grad():
        action_id, _, _ = agent.act(state_tensor, action_mask_tensor)

    # ── Decode: ID → Spec → Cards ─────────────────────────────────────────
    action_spec = ACTION_SPACE[action_id]
    action_cards = resolve_action(
        action_spec=action_spec,
        hand=hand,
        current_trick=current_trick,
    )

    return action_id, action_cards