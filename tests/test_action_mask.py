from core.card import Card
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.move_type import MoveType
from core.rules import get_legal_moves

def test_illegal_pair_not_allowed():
    # 3, 4 không phải là đôi
    hand = [Card("3","♠"), Card("4","♠")]
    legal_moves = get_legal_moves(hand, None)
    mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)

    # Lấy index của các MoveType.PAIR
    pair_indices = [i for i, a in enumerate(ACTION_SPACE) if a.move_type == MoveType.PAIR]
    
    for idx in pair_indices:
        assert mask[idx] == 0, f"Action {ACTION_SPACE[idx]} should be masked out"

def test_four_kind_chops_two():
    # Tứ quý 6 chặt heo
    hand = [
        Card("6","♠"), Card("6","♣"),
        Card("6","♦"), Card("6","♥"),
    ]
    trick = [Card("2","♠")]

    legal_moves = get_legal_moves(hand, trick)
    mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)

    # Tìm index của Tứ quý 6 (MoveType.FOUR_OF_KIND, length 4)
    four_6_idx = -1
    for i, a in enumerate(ACTION_SPACE):
        if a.move_type == MoveType.FOUR_OF_KIND and a.length == 4:
            four_6_idx = i
            break
    
    assert four_6_idx != -1
    assert mask[four_6_idx] == 1, "Four of a kind should be able to chop a TWO"

def test_two_only_hand_can_lead():
    hand = [Card("2", "♠")]
    legal_moves = get_legal_moves(hand, None)
    mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)

    two_idx = -1
    for i, a in enumerate(ACTION_SPACE):
        if a.move_type == MoveType.TWO and a.length == 1:
            two_idx = i
            break
    
    assert two_idx != -1
    assert mask[two_idx] == 1, "Should be able to lead with a TWO"
    
    pass_idx = -1
    for i, a in enumerate(ACTION_SPACE):
        if a.move_type == MoveType.PASS:
            pass_idx = i
            break
    assert mask[pass_idx] == 0, "Should NOT be able to PASS when leading"
