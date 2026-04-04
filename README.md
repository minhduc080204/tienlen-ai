# TienLen AI — PPO Training

Dự án train AI chơi Tiến Lên Miền Nam bằng Reinforcement Learning (PPO).

---

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\activate
source venv/bin/activate

# Kích hoạt (Linux/macOS)
source venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

---

## 🏗️ Kiến trúc State (235 dims)

| Thành phần | Dims | Mô tả |
|---|---|---|
| Hand | 65 | 52-bit binary + 13 stats (số heo, đôi, sảnh...) |
| Current Trick | 70 | 52-bit binary + 18 stats (bộ bài hiện tại trên bàn) |
| Opponents | 30 | Card counts, Danger score, **Passed Flags** (quan trọng) |
| Discard Pile | 70 | 52-bit binary + 18 stats (đếm bài, heo/tứ quý đã ra) |
| **Total** | **235** | |

---

## 🚀 Huấn luyện (Training)

Hệ thống hỗ trợ 3 giai đoạn: **Warm-up** (vs RuleBot), **Self-Play** (vs Frozen AI), và **Shared Model** (4-way PPO).

### 1. Trên Kaggle (GPU T4) - Khuyên dùng
Tận dụng tối đa sức mạnh GPU với Batch Size lớn (1024) và 100,000 ván đấu.
```bash
python -m train.train_loop --episodes 100000 --device cuda
```

### 2. Chạy thử ở Local (CPU/GPU)
Dùng để kiểm tra tính đúng đắn của code trước khi đẩy lên server.
```bash
# Kích hoạt venv trước
.\venv\Scripts\activate

# Chạy test 500 ván
python -m train.train_loop --episodes 500 --device cpu
```

---

## 📁 Cấu trúc Project

```
tienlen-ai/
├── core/           # Card, Deck, Rules, MoveType
├── state/          # State encoding (183 dims)
├── env/            # TienLenEnv, GameState, Reward
├── rl/             # PPO Agent, Model, Buffer
├── action/         # Action space, mask
├── bots/           # RuleBot (đối thủ train)
├── train/          # Train loop, Config
├── inference/      # FastAPI service
├── checkpoints/    # Saved model weights
└── logs/           # Training logs
```
