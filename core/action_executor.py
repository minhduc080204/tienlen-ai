# core/action_executor.py
from core.move_type import MoveType
from core.rules import (
    get_legal_moves,
    detect_move_type
)


def resolve_action(action_spec, hand, current_trick):
    legal_moves = get_legal_moves(hand, current_trick)

    if action_spec.move_type == MoveType.PASS:
        return []

    candidates = [
        move for move in legal_moves
        if detect_move_type(move) == action_spec.move_type
        and len(move) == action_spec.length
    ]

    if not candidates:
        raise RuntimeError(
            f"❌ resolve_action failed\n"
            f"ActionSpec={action_spec}\n"
            f"Hand={hand}\n"
            f"LegalMoves={legal_moves}"
        )

    # ưu tiên đánh combo nhỏ nhất (rank nhỏ nhất)
    return min(
        candidates,
        key=lambda cards: min(c.rank for c in cards)
    )
