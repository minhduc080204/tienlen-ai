# action/action_mask.py
import numpy as np
from core.card import Card
from core.move_type import MoveType
from core.rules import (
    detect_move_type
)

def build_action_mask_from_legal_moves(legal_moves, action_space):
    mask = np.zeros(len(action_space), dtype=np.float32)

    has_non_pass = any(len(move) > 0 for move in legal_moves)

    for i, spec in enumerate(action_space):

        # PASS
        if spec.move_type == MoveType.PASS:
            mask[i] = 1.0 if not has_non_pass else 0.0
            continue

        for move in legal_moves:
            if not move:
                continue

            if (
                detect_move_type(move) == spec.move_type
                and len(move) == spec.length
            ):
                mask[i] = 1.0
                break

    # fallback: đảm bảo không all-zero
    if mask.sum() == 0:
        for i, spec in enumerate(action_space):
            if spec.move_type == MoveType.PASS:
                mask[i] = 1.0
                break

    return mask