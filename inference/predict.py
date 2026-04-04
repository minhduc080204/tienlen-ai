# inference/predict.py
import torch
import numpy as np
from typing import List, Optional, Tuple

from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.rules import get_legal_moves
from core.action_executor import resolve_action
from core.card import Card


def predict_action(
    agent,
    device: torch.device,
    hand: List[Card],
    opponent_counts: List[int],
    current_trick: Optional[List[Card]],
    player_id: int,
    num_players: int,
    discard_pile: Optional[List[Card]] = None,
) -> Tuple[int, List[Card], float]:
    """
    Dự đoán action cho AI player dựa trên state vector 183 chiều.

    Returns:
        action_id: ID của hành động trong ACTION_SPACE
        action_cards: Danh sách Card thực tế sẽ đánh
        value_estimate: Giá trị ước lượng của state (V-value)
    """
    if discard_pile is None:
        discard_pile = []

    # 1. Encode STATE (183 dims)
    state_vec = encode_state(
        hand=hand,
        opponent_counts=opponent_counts,
        current_trick=current_trick,
        player_id=player_id,
        num_players=num_players,
        discard_pile=discard_pile,
    )

    # 2. Build ACTION MASK
    legal_moves = get_legal_moves(hand=hand, current_trick=current_trick)
    action_mask = build_action_mask_from_legal_moves(
        legal_moves=legal_moves,
        action_space=ACTION_SPACE,
    )

    # 3. Model Inference (PPO)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        # Predict action và state value
        action_id, _, value, _ = agent.act(state_tensor, action_mask_tensor)
        
    # Convert value to float
    value_estimate = value.cpu().item() if hasattr(value, "cpu") else float(value)

    # 4. Resolve ACTION (ID -> Cards)
    action_spec = ACTION_SPACE[action_id]
    action_cards = resolve_action(
        action_spec=action_spec,
        hand=hand,
        current_trick=current_trick,
    )

    return int(action_id), action_cards, value_estimate