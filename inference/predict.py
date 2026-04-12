# inference/predict.py
import torch
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
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int = 3,
) -> Tuple[int, List[Card], float, float, List[Tuple[int, float]]]:
    """
    Dự đoán action cho AI player dựa trên state vector 235 chiều.

    Returns:
        action_id: ID của hành động trong ACTION_SPACE
        action_cards: Danh sách Card thực tế sẽ đánh
        value_estimate: Giá trị ước lượng của state (V-value)
        action_confidence: Xác suất policy của action được chọn
        top_actions: Danh sách top action_id hợp lệ kèm xác suất
    """
    if discard_pile is None:
        discard_pile = []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # 1. Encode STATE (235 dims)
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
    action_mask_np = build_action_mask_from_legal_moves(
        legal_moves=legal_moves,
        action_space=ACTION_SPACE,
    )

    # 3. Model Inference (PPO)
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    action_mask = torch.tensor(action_mask_np, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits, value = agent.model(state_tensor)
        masked_logits = logits.squeeze(0).masked_fill(~action_mask, -1e9)
        scaled_logits = masked_logits / float(temperature)
        probs = torch.softmax(scaled_logits, dim=-1)

        legal_ids = torch.nonzero(action_mask, as_tuple=False).flatten()
        legal_probs = probs[legal_ids]
        top_count = min(max(top_k, 1), legal_ids.numel())
        top_probs, top_idx_in_legal = torch.topk(legal_probs, k=top_count)
        top_action_ids = legal_ids[top_idx_in_legal]

        if greedy:
            selected_id = int(torch.argmax(scaled_logits).item())
        else:
            dist = torch.distributions.Categorical(probs=probs)
            selected_id = int(dist.sample().item())

    # Convert value to float
    value_estimate = value.squeeze(0).cpu().item() if hasattr(value, "cpu") else float(value)

    # 4. Resolve ACTION (ID -> Cards)
    # Nếu action sampled không resolve được combo cụ thể thì fallback theo top probs hợp lệ.
    candidate_ids = [selected_id] + [int(aid.item()) for aid in top_action_ids if int(aid.item()) != selected_id]
    last_error: Optional[Exception] = None
    action_cards: List[Card] = []
    action_id = selected_id
    for candidate_id in candidate_ids:
        try:
            action_spec = ACTION_SPACE[candidate_id]
            action_cards = resolve_action(
                action_spec=action_spec,
                hand=hand,
                current_trick=current_trick,
            )
            action_id = candidate_id
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Could not resolve any legal action from candidates={candidate_ids}") from last_error

    action_confidence = float(probs[action_id].detach().cpu().item())
    top_actions = [
        (int(aid.item()), float(prob.item()))
        for aid, prob in zip(top_action_ids, top_probs)
    ]

    return int(action_id), action_cards, value_estimate, action_confidence, top_actions
