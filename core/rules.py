# core/rules.py
from collections import Counter
from typing import List
from core.card import Card
from core.move_type import MoveType
from itertools import combinations


def is_valid_single(cards: List[Card]) -> bool:
    return len(cards) == 1


def is_valid_pair(cards: List[Card]) -> bool:
    return len(cards) == 2 and cards[0].rank == cards[1].rank


def is_valid_triple(cards: List[Card]) -> bool:
    return len(cards) == 3 and len(set(c.rank for c in cards)) == 1


def is_valid_straight(cards: List[Card]) -> bool:
    if len(cards) < 3:
        return False

    ranks = sorted(c.rank for c in cards)
    if 15 in ranks:  # không cho sảnh chứa heo (2=15)
        return False

    return all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1))


def is_two(cards: List[Card]) -> bool:
    """Heo (2) = rank 15"""
    return len(cards) == 1 and cards[0].rank == 15


def is_four_of_kind(cards: List[Card]) -> bool:
    return len(cards) == 4 and len(set(c.rank for c in cards)) == 1


def is_double_straight(cards: List[Card]) -> bool:
    if len(cards) < 6 or len(cards) % 2 != 0:
        return False

    ranks = sorted(c.rank for c in cards)
    if 15 in ranks:  # không có heo
        return False

    counter = Counter(ranks)
    if any(v != 2 for v in counter.values()):
        return False

    uniq = sorted(counter.keys())
    return all(uniq[i] + 1 == uniq[i + 1] for i in range(len(uniq) - 1))


def detect_move_type(cards: List[Card]):
    if not cards:
        return MoveType.PASS
    if is_two(cards):
        return MoveType.TWO
    if is_four_of_kind(cards):
        return MoveType.FOUR_OF_KIND
    if is_double_straight(cards):
        return MoveType.DOUBLE_STRAIGHT
    if is_valid_single(cards):
        return MoveType.SINGLE
    if is_valid_pair(cards):
        return MoveType.PAIR
    if is_valid_triple(cards):
        return MoveType.TRIPLE
    if is_valid_straight(cards):
        return MoveType.STRAIGHT
    return None


def compare_single(a: Card, b: Card) -> bool:
    """True nếu a thắng b (rank cao hơn, hoặc cùng rank thì suit cao hơn)"""
    if a.rank != b.rank:
        return a.rank > b.rank
    return a.suit > b.suit


def can_beat(prev_cards: List[Card], new_cards: List[Card]) -> bool:
    prev_type = detect_move_type(prev_cards)
    new_type = detect_move_type(new_cards)

    if new_type == MoveType.PASS:
        return False

    # chặt heo
    if prev_type == MoveType.TWO:
        if new_type == MoveType.TWO:
            return compare_single(new_cards[0], prev_cards[0])
        if new_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
            return True
        return False

    # chặt đôi thông
    if prev_type == MoveType.DOUBLE_STRAIGHT:
        return new_type == MoveType.DOUBLE_STRAIGHT and len(new_cards) > len(prev_cards)

    # chặt tứ quý
    if prev_type == MoveType.FOUR_OF_KIND:
        return new_type == MoveType.FOUR_OF_KIND and new_cards[0].rank > prev_cards[0].rank

    # bình thường
    if prev_type != new_type:
        return False

    if prev_type == MoveType.SINGLE:
        return compare_single(new_cards[0], prev_cards[0])

    if prev_type in [MoveType.PAIR, MoveType.TRIPLE]:
        return new_cards[0].rank > prev_cards[0].rank

    if prev_type == MoveType.STRAIGHT:
        return max(c.rank for c in new_cards) > max(c.rank for c in prev_cards)

    return False


def get_legal_moves(
    hand: List[Card],
    current_trick: List[Card] | None
) -> List[List[Card]]:
    """
    Sinh toàn bộ nước đi hợp lệ từ hand.
    PASS được biểu diễn bằng [].
    """

    legal_moves: List[List[Card]] = []

    # 1️⃣ PASS (chỉ khi có bài trước)
    if current_trick is not None:
        legal_moves.append([])

    # 2️⃣ Sinh mọi tổ hợp bài
    n = len(hand)

    for size in range(1, n + 1):
        for combo in combinations(hand, size):
            cards = list(combo)

            move_type = detect_move_type(cards)
            if move_type is None or move_type == MoveType.PASS:
                continue

            # 3️⃣ Không có bài trước → đánh tự do
            if current_trick is None:
                legal_moves.append(cards)
            else:
                if can_beat(current_trick, cards):
                    legal_moves.append(cards)

    return legal_moves