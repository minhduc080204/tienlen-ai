# env/reward.py
from core.rules import detect_move_type, MoveType

# ─── Terminal reward ─────────────────────────────────────────────────────────
WIN_REWARD   = 60.0
LOSE_PENALTY = -60.0

# ─── Step rewards ────────────────────────────────────────────────────────────
PLAY_CARD_REWARD = 0.05
PASS_PENALTY     = -0.2

# ─── Opponent pressure ───────────────────────────────────────────────────────
OPPONENT_CRITICAL_PENALTY    = -8.0   # pass khi đối thủ còn 1 lá
OPPONENT_DANGEROUS_PENALTY   = -5.0   # pass khi đối thủ còn 2 lá
OPPONENT_DANGER_GOOD_PLAY    = +3.0   # đánh được khi đối thủ nguy hiểm

# ─── Cutting 2 (heo) ─────────────────────────────────────────────────────────
GOOD_CUT_TWO      = +8.0
BAD_CUT_TWO       = -5.0
CUT_TWO_URGENT    = +12.0   # chặt heo khi đối thủ còn ≤2 bài

# ─── Power card management ───────────────────────────────────────────────────
SAVE_POWER_CARD   = +2.0
WASTE_POWER_CARD  = -3.0

# ─── Efficiency bonus ────────────────────────────────────────────────────────
LARGE_COMBO_BONUS  = +1.0    # thưởng khi đánh sảnh/đôi thông dài
CLEAR_HAND_BONUS   = +0.5    # thưởng mỗi lá đánh khi ≤5 lá còn

# ─── Balance terminal vs shaping ─────────────────────────────────────────────
SHAPING_SCALE = 0.35         # giảm tác động reward shaping
SHAPING_CLIP  = 12.0         # tránh step reward quá lớn gây lệch objective


# ─────────────────────────────────────────────────────────────────────────────
# 1. TERMINAL REWARD
# ─────────────────────────────────────────────────────────────────────────────
def terminal_reward(player_rank: int) -> float:
    if player_rank == 1:
        return WIN_REWARD
    return LOSE_PENALTY


# ─────────────────────────────────────────────────────────────────────────────
# 2. ACTION-LEVEL REWARD (REWARD SHAPING)
# ─────────────────────────────────────────────────────────────────────────────
def action_reward(
    action_cards,
    prev_state,
    next_state,
    player_id: int
) -> float:
    reward = 0.0

    hand_before = prev_state.hands[player_id]
    remaining   = len(hand_before)

    # ─── Thông tin đối thủ ───────────────────────────────────────────────
    opponent_counts = [
        len(prev_state.hands[pid])
        for pid in range(len(prev_state.hands))
        if pid != player_id
    ]
    min_opponent_count = min(opponent_counts) if opponent_counts else 13

    # ─── 2.1 Đánh / Pass cơ bản ─────────────────────────────────────────
    if not action_cards:
        reward += PASS_PENALTY
    else:
        reward += PLAY_CARD_REWARD

    # ─── 2.2 Đối thủ nguy hiểm → phải chặn ─────────────────────────────
    if min_opponent_count <= 1:
        if not action_cards:
            reward += OPPONENT_CRITICAL_PENALTY
        else:
            reward += OPPONENT_DANGER_GOOD_PLAY * 1.5  # chặt được khi crit

    elif min_opponent_count <= 2:
        if not action_cards:
            reward += OPPONENT_DANGEROUS_PENALTY
        else:
            reward += OPPONENT_DANGER_GOOD_PLAY

    elif min_opponent_count <= 3:
        if not action_cards:
            reward += OPPONENT_DANGEROUS_PENALTY * 0.5
        elif action_cards:
            reward += OPPONENT_DANGER_GOOD_PLAY * 0.5

    # ─── 2.3 Chặt heo (risk vs reward) ─────────────────────────────────
    prev_trick = prev_state.current_trick

    if prev_trick and detect_move_type(prev_trick) == MoveType.TWO:
        move_type = detect_move_type(action_cards) if action_cards else None

        if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
            bonus = CUT_TWO_URGENT if min_opponent_count <= 2 else GOOD_CUT_TWO
            reward += bonus

        elif move_type == MoveType.TWO:
            # heo chặt heo → so chất (suit cao hơn = tốt hơn)
            if action_cards[0].suit > prev_trick[0].suit:
                reward += GOOD_CUT_TWO
            else:
                reward += BAD_CUT_TWO

    # ─── 2.4 Quản lý bài mạnh theo giai đoạn game ───────────────────────
    if action_cards:
        move_type = detect_move_type(action_cards)

        # Cuối game (≤5 lá) → cần thoát bài nhanh
        if remaining <= 5:
            reward += len(action_cards) * CLEAR_HAND_BONUS

            # Dùng bài mạnh cuối game là đúng khi đối thủ nguy hiểm
            if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
                if min_opponent_count <= 3:
                    reward += GOOD_CUT_TWO    # tốt: dùng combo mạnh để khoá
                else:
                    reward += WASTE_POWER_CARD  # tệ: phá bài mạnh sớm

        # Giữa game (6-9 lá) → ưu tiên sảnh dài, đôi thông
        elif remaining <= 9:
            if move_type == MoveType.STRAIGHT and len(action_cards) >= 4:
                reward += LARGE_COMBO_BONUS
            elif move_type == MoveType.DOUBLE_STRAIGHT:
                reward += LARGE_COMBO_BONUS * 1.5

        # Đầu game (>9 lá) → đánh bài nhỏ để giành lượt
        else:
            if move_type == MoveType.SINGLE and action_cards[0].rank <= 7:
                reward += SAVE_POWER_CARD * 0.5   # bài nhỏ → giữ bài mạnh
            if move_type in [MoveType.FOUR_OF_KIND, MoveType.DOUBLE_STRAIGHT]:
                reward += WASTE_POWER_CARD * 0.5  # phí bài mạnh đầu game

    # ─── 2.5 Thoát bài hiệu quả (đánh nhiều lá 1 lượt) ─────────────────
    if action_cards and len(action_cards) >= 4:
        reward += LARGE_COMBO_BONUS * (len(action_cards) / 4.0)

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL REWARD (dùng trong env.step)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(
    action_cards,
    prev_state,
    next_state,
    done: bool,
    player_id: int,
    player_rank: int | None = None
) -> float:
    shaped_reward = action_reward(
        action_cards=action_cards,
        prev_state=prev_state,
        next_state=next_state,
        player_id=player_id
    )
    reward = max(-SHAPING_CLIP, min(SHAPING_CLIP, shaped_reward * SHAPING_SCALE))

    if done:
        reward += terminal_reward(player_rank)

    return reward
