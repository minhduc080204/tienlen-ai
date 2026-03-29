# bots/hand_analyzer.py
from collections import Counter
from core.move_type import MoveType
from core.rules import detect_move_type


class HandPlan:
    def __init__(self):
        self.singles = []
        self.pairs = []
        self.triples = []
        self.straights = []
        self.double_straights = []
        self.four_kinds = []

    def all_combos(self):
        return (
            self.four_kinds
            + self.double_straights
            + self.straights
            + self.triples
            + self.pairs
            + self.singles
        )


class HandAnalyzer:
    """
    Xếp bài sao cho:
    - combo dài lấy trước
    - giảm số quân lẻ
    """

    def analyze(self, hand):
        plan = HandPlan()
        used = set()

        by_rank = {}
        for c in hand:
            by_rank.setdefault(c.rank_value, []).append(c)

        # -------------------------
        # 1️⃣ TỨ QUÝ
        # -------------------------
        for r, cards in by_rank.items():
            if len(cards) == 4:
                plan.four_kinds.append(cards)
                used.update(cards)

        # -------------------------
        # 2️⃣ SÁM
        # -------------------------
        for r, cards in by_rank.items():
            remain = [c for c in cards if c not in used]
            if len(remain) == 3:
                plan.triples.append(remain)
                used.update(remain)

        # -------------------------
        # 3️⃣ ĐÔI
        # -------------------------
        for r, cards in by_rank.items():
            remain = [c for c in cards if c not in used]
            if len(remain) >= 2:
                plan.pairs.append(remain[:2])
                used.update(remain[:2])

        # -------------------------
        # 4️⃣ LẺ
        # -------------------------
        for c in hand:
            if c not in used:
                plan.singles.append([c])

        plan.singles.sort(key=lambda cs: cs[0].rank_value)
        plan.pairs.sort(key=lambda cs: cs[0].rank_value)
        plan.triples.sort(key=lambda cs: cs[0].rank_value)

        return plan
