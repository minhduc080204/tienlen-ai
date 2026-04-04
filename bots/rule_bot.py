# bots/rule_bot.py

from core.rules import (
    get_legal_moves,
    detect_move_type,
    can_beat
)
from core.move_type import MoveType
from bots.hand_analyzer import HandAnalyzer, HandPlan


class RuleBot:
    """
    Rule-based bot nâng cấp:
    - Có chiến thuật End-game (chặn đường về của đối thủ).
    - Biết giữ bài mạnh (Safe play).
    - Biết xé bộ khi cần thiết để giành lượt.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.analyzer = HandAnalyzer()

    def act(self, state):
        return self.select_action(state, self.player_id)

    def select_action(self, state, player_id):
        hand = state.hands[player_id]
        current_trick = state.current_trick
        
        # 1. Tính toán số bài của đối thủ
        num_players = len(state.hands)
        opp_counts = [len(state.hands[i]) for i in range(num_players) if i != player_id]
        is_end_game = any(c <= 3 for c in opp_counts) if opp_counts else False

        if not hand:
            return []

        # 2. Phân tích bài thành các bộ tối ưu
        plan = self.analyzer.analyze(hand)
        
        # 3. Lấy danh sách nước đi hợp lệ
        legal_moves = get_legal_moves(hand, current_trick)
        if not legal_moves or (current_trick is not None and len(legal_moves) == 1 and not legal_moves[0]):
            return [] # Chỉ có duy nhất nước PASS

        # ==========================================
        # TRƯỜNG HỢP: ĐƯỢC CẦM CÁI (FREE TURN)
        # ==========================================
        if current_trick is None:
            all_combos = plan.get_all_combos()
            if not all_combos:
                # Fallback: if somehow everything failed, play any legal move
                return legal_moves[0] if legal_moves else []

            if is_end_game:
                # Chế độ End-game: Đánh bộ lớn nhất hoặc lá lớn nhất để về nhanh
                # Ưu tiên đánh sảnh dài nhất hoặc bộ nhiều lá nhất trước
                all_combos.sort(key=lambda x: (len(x), max(c.rank for c in x)), reverse=True)
                return all_combos[0]
            else:
                # Chế độ bình thường: Ưu tiên xả rác nhỏ hoặc bộ nhỏ
                # Ưu tiên: Sảnh nhỏ -> Ba nhỏ -> Đôi nhỏ -> Rác nhỏ
                if plan.straights: return plan.straights[0]
                if plan.triples: return plan.triples[0]
                if plan.pairs: return plan.pairs[0]
                return all_combos[0] # Lấy bộ nhỏ nhất có thể

        # ==========================================
        # TRƯỜNG HỢP: THEO BÀI (FOLLOW TURN)
        # ==========================================
        target_type = detect_move_type(current_trick)
        
        # Lọc ra các nước có thể thắng (bỏ qua PASS)
        winning_moves = [m for m in legal_moves if m]
        if not winning_moves:
            return []

        # Logic chọn nước đi tối ưu dựa trên loại bộ đang đánh
        if is_end_game:
            # Nếu đối thủ sắp về, đánh lá LỚN NHẤT có thể để chặn
            winning_moves.sort(key=lambda x: (max(c.rank for c in x), max(c.suit for c in x)), reverse=True)
            return winning_moves[0]
        
        # Chế độ bình thường: Cố gắng thắng bằng lá NHỎ NHẤT có thể
        winning_moves.sort(key=lambda x: (max(c.rank for c in x), max(c.suit for c in x)))

        # "Safe Play": Không đánh 2 (heo) hoặc A bừa bãi nếu bài còn quá nhiều
        best_small_move = winning_moves[0]
        max_rank_in_move = max(c.rank for c in best_small_move)
        
        if max_rank_in_move >= 14: # A hoặc 2
            # Nếu bài còn nhiều (> 7 lá), và không phải là nước cuối, cân nhắc bỏ qua
            if len(hand) > 7 and not self._is_guaranteed_win(hand, best_small_move):
                return [] # PASS để giữ bài to

        # "Xé bộ": Nếu đối thủ đánh lẻ mà mình không có rác để chặn, xem xét xé đôi nhỏ
        if target_type == MoveType.SINGLE and not any(m for m in winning_moves if m[0] in [c[0] for c in plan.singles]):
            # Tìm trong plan.pairs xem có đôi nào nhỏ (rank < 10) có thể xé không
            for pair in plan.pairs:
                for card in pair:
                    if can_beat(current_trick, [card]):
                        return [card]

        return best_small_move

    def _is_guaranteed_win(self, hand, move):
        """Kiểm tra xem nước đi này có giúp mình nắm chắc phần thắng không (Heuristic)"""
        return len(hand) - len(move) <= 2
