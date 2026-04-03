# inference/predict.py
import torch
import numpy as np

from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.rules import get_legal_moves

# 👉 Import chính cái hàm resolve_action thần thánh của bạn
from core.action_executor import resolve_action 

def predict_action(
    agent,
    device,
    hand,
    opponent_counts,
    current_trick,
    player_id,
    num_players
):
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
    legal_moves = get_legal_moves(hand=hand, current_trick=current_trick)

    action_mask = build_action_mask_from_legal_moves(
        legal_moves=legal_moves,
        action_space=ACTION_SPACE
    )

    # ===== TO TENSOR =====
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device)
    action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)

    # ===== PREDICT =====
    with torch.no_grad():
        action_id, _, _ = agent.act(state_tensor, action_mask_tensor)

    # ===== DECODE: TỪ ID -> SPEC -> CARDS =====
    action_spec = ACTION_SPACE[action_id]

    # 👉 Dùng đúng hàm resolve_action như lúc train!
    action_cards = resolve_action(
        action_spec=action_spec,
        hand=hand,
        current_trick=current_trick
    )

    # action_cards lúc này sẽ là 1 list[Card] hoặc [] (nếu PASS)
    return action_id, action_cards