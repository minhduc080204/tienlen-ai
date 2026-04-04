# bots/hand_analyzer.py
from collections import Counter
from core.card import Card
from core.rules import detect_move_type, MoveType


class HandPlan:
    def __init__(self):
        self.singles = []
        self.pairs = []
        self.triples = []
        self.straights = []
        self.double_straights = []
        self.four_kinds = []

    def get_all_combos(self):
        # Ưu tiên các bộ đặc biệt trước
        return (
            self.four_kinds +
            self.double_straights +
            self.straights +
            self.triples +
            self.pairs +
            self.singles
        )


class HandAnalyzer:
    """
    Phân tích bài thành các bộ (combinations).
    Heuristic: Tìm bộ dài nhất/mạnh nhất trước.
    """

    def analyze(self, hand: list[Card]) -> HandPlan:
        plan = HandPlan()
        if not hand:
            return plan

        # Copy hand to avoid mutation
        cards = sorted(hand, key=lambda c: (c.rank, c.suit))
        used_ids = set()

        def is_used(c): return c.id in used_ids
        def mark_used(cs): used_ids.update(c.id for c in cs)

        # 1. Tìm Đôi Thông (Double Straights) - Ưu tiên hàng đầu vì cực mạnh
        # Cần ít nhất 3 đôi liên tiếp (6 lá)
        by_rank = {}
        for c in cards:
            if c.rank < 15: # Heo không nằm trong sảnh/đôi thông
                by_rank.setdefault(c.rank, []).append(c)
        
        ranks = sorted(by_rank.keys())
        i = 0
        while i < len(ranks):
            j = i
            while j < len(ranks) and len(by_rank[ranks[j]]) >= 2:
                if j > i and ranks[j] != ranks[j-1] + 1:
                    break
                j += 1
            
            # Nếu tìm thấy ít nhất 3 rank liên tiếp có đôi
            if j - i >= 3:
                ds = []
                for k in range(i, j):
                    ds.extend(by_rank[ranks[k]][:2])
                plan.double_straights.append(ds)
                mark_used(ds)
                i = j
            else:
                i += 1

        # 2. Tìm Tứ Quý
        remaining = [c for c in cards if not is_used(c)]
        rank_counts = Counter(c.rank for c in remaining)
        for r, count in rank_counts.items():
            if count == 4:
                four = [c for c in remaining if c.rank == r]
                plan.four_kinds.append(four)
                mark_used(four)

        # 3. Tìm Sảnh (Straights) - Ưu tiên sảnh dài nhất
        remaining = [c for c in cards if not is_used(c) and c.rank < 15]
        if remaining:
            # Group by rank (chọn lá bài có suit nhỏ nhất cho sảnh để giữ lá to)
            rank_to_card = {}
            for c in remaining:
                if c.rank not in rank_to_card:
                    rank_to_card[c.rank] = c
            
            ranks = sorted(rank_to_card.keys())
            i = 0
            while i < len(ranks):
                j = i
                while j + 1 < len(ranks) and ranks[j+1] == ranks[j] + 1:
                    j += 1
                
                if j - i >= 2: # Ít nhất 3 lá
                    straight = [rank_to_card[ranks[k]] for k in range(i, j + 1)]
                    plan.straights.append(straight)
                    mark_used(straight)
                    i = j + 1
                else:
                    i += 1

        # 4. Tìm Sám Cô (Triples)
        remaining = [c for c in cards if not is_used(c)]
        rank_counts = Counter(c.rank for c in remaining)
        for r, count in rank_counts.items():
            if count == 3:
                triple = [c for c in remaining if c.rank == r]
                plan.triples.append(triple)
                mark_used(triple)

        # 5. Tìm Đôi (Pairs)
        remaining = [c for c in cards if not is_used(c)]
        rank_counts = Counter(c.rank for c in remaining)
        for r in sorted(rank_counts.keys()):
            if rank_counts[r] >= 2:
                pair = [c for c in remaining if c.rank == r][:2]
                plan.pairs.append(pair)
                mark_used(pair)

        # 6. Còn lại là rác (Singles)
        remaining = [c for c in cards if not is_used(c)]
        for c in remaining:
            plan.singles.append([c])

        return plan
