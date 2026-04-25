# ======================
# GAME
# ======================
NUM_PLAYERS = 4
AI_PLAYER_ID = 0
MAX_TURNS_PER_EP = 120  # giới hạn số lượt mỗi ván

# ======================
# TRAINING PHASES (Scaled for Kaggle T4)
# ======================
WARMUP_EPISODES = 10000      # Giai đoạn 1: vs RuleBot (AI học luật cơ bản)
SELF_PLAY_EPISODES = 40000   # Giai đoạn 2: vs Frozen Model (AI học chiến thuật)
SHARED_MODEL_START = 50000   # Giai đoạn 3: 4-way Shared Model (Hội tụ nâng cao)

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
ENTROPY_COEF = 0.01        # Khuyến khích khám phá (Exploration)
VALUE_COEF = 0.5           # Trọng số của Value Loss

LOG_TURN = False           # Tắt log chi tiết trên Kaggle để tránh lag
LOG_TURN_EPISODE = 1000
