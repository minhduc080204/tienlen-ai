# ======================
# GAME
# ======================
NUM_PLAYERS = 4
AI_PLAYER_ID = 0
MAX_TURNS_PER_EP = 120  # giới hạn số lượt mỗi ván

# ======================
# TRAINING PHASES (Scaled for Kaggle T4)
# ======================
WARMUP_EPISODES = 5000      # Giai đoạn 1: vs RuleBot (AI học luật cơ bản)
SELF_PLAY_EPISODES = 35000   # Giai đoạn 2: vs Frozen Model (AI học chiến thuật)
SHARED_MODEL_START = 40000   # Giai đoạn 3: 4-way Shared Model (Hội tụ nâng cao)

# ======================
# SELF-PLAY CONFIG
# ======================
OPPONENT_UPDATE_EVERY = 500  # Cập nhật đối thủ sau mỗi n episodes
WIN_RATE_THRESHOLD = 0.55    # Ngưỡng lưu best_model (WR > 50% là bắt đầu ổn)
WINDOW_SIZE = 100            # Cửa sổ tính WR (lớn hơn để tránh nhiễu)

# ======================
# PPO (Optimized for T4)
# ======================
GAMMA = 0.99
LAMBDA = 0.95

PPO_EPOCHS = 4             # Số lần lặp lại update trên mỗi batch
BATCH_SIZE = 1024          # 🔥 Tăng để tận dụng tối đa GPU T4
LR = 3e-4                  # Tốc độ học tối ưu cho PPO

# ======================
# STABILITY
# ======================
MAX_TURNS_PER_GAME = 200   # Tránh ván bài bị treo quá lâu
ENTROPY_COEF_PHASE_1 = 0.03   # Warm-up: exploration cao
ENTROPY_COEF_PHASE_2 = 0.015  # Self-play: exploration vừa
ENTROPY_COEF_PHASE_3 = 0.005  # Shared model: exploit nhiều hơn
VALUE_COEF = 0.5           # Trọng số của Value Loss
TARGET_KL = 0.015          # Early stop PPO epoch nếu KL tăng quá nhanh
VALUE_CLIP_EPS = 0.2       # Clipping cho value function
NORMALIZE_RETURNS = True   # Chuẩn hóa returns theo batch update

# Learning-rate decay (linear theo tiến độ training)
LR_MIN_FACTOR = 0.1        # LR sẽ decay từ LR xuống LR * LR_MIN_FACTOR

LOG_TURN = False           # Tắt log chi tiết trên Kaggle để tránh lag
LOG_TURN_EPISODE = 1000
