# bots/rule_bot.py

from core.rules import (
    get_legal_moves,
    detect_move_type
)
from core.move_type import MoveType
from bots.hand_analyzer import HandAnalyzer


class RuleBot:
    """
    Rule-based bot:
    - Chỉ đánh nước hợp lệ
    - Ưu tiên bài nhỏ
    - Dùng làm đối thủ train PPO
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.analyzer = HandAnalyzer()


    # ========================
    # PUBLIC API (chuẩn hoá)
    # ========================
    def act(self, state):
        """
        Train loop sẽ gọi hàm này
        """
        return self.select_action(state, self.player_id)

    # ========================
    # CORE LOGIC
    # ========================
    def select_action(self, state, player_id):
        hand = state.hands[player_id]
        current_trick = state.current_trick

        if not hand:
            return []

        legal_moves = get_legal_moves(hand, current_trick)
        if not legal_moves:
            return []

        # ==========================
        # 1️⃣ XẾP BÀI
        # ==========================
        plan = self.analyzer.analyze(hand)

        # ==========================
        # 2️⃣ THEO BÀI
        # ==========================
        if current_trick is not None:
            # chỉ đánh SINGLE từ singles
            if detect_move_type(current_trick) == MoveType.SINGLE:
                for combo in plan.singles:
                    card = combo[0]

                    # ❌ nếu lá này thuộc triple → bỏ qua
                    in_triple = any(card in t for t in plan.triples)
                    if in_triple:
                        continue

                    if combo in legal_moves:
                        return combo

            return []

        # ==========================
        # 3️⃣ ĐÁNH ĐẦU
        # ==========================
        if plan.singles:
            return plan.singles[0]

        if plan.pairs:
            return plan.pairs[0]

        # fallback
        return legal_moves[0]

    # ------------------------
    # Helpers
    # ------------------------

    def _priority(self, cards):
        move_type = detect_move_type(cards)

        priority = {
            MoveType.SINGLE: 0,
            MoveType.PAIR: 1,
            MoveType.TRIPLE: 2,
            MoveType.STRAIGHT: 3,
            MoveType.DOUBLE_STRAIGHT: 4,
            MoveType.FOUR_OF_KIND: 5,
            MoveType.TWO: 6,
        }
        return priority.get(move_type, 99)

    def _min_rank(self, cards):
        return min(card.rank for card in cards)
