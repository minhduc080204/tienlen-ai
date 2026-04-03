# TienLen AI — PPO Training

Dự án train AI chơi Tiến Lên Miền Nam bằng Reinforcement Learning (PPO).

---

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\activate

# Kích hoạt (Linux/macOS)
source venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

---

## 🖥️ Chạy trên Local (CPU)

### Train nhanh để test

```bash
python -m train.train_loop --episodes 100 --save-every 50 --batch-size 64 --lr 2.5e-4 --device cpu
```

### Train đầy đủ (nếu có GPU local)

```bash
python -m train.train_loop \
  --episodes 5000 \
  --save-every 500 \
  --batch-size 128 \
  --lr 2.5e-4 \
  --gamma 0.99 \
  --lam 0.95 \
  --ppo-epochs 4 \
  --device cuda
```

### Chạy Inference API

```bash
uvicorn inference.ai_service:app --host 0.0.0.0 --port 8000
```

> API nhận POST `/predict` với body JSON:
> ```json
> {
>   "hand": [{"rank": 3, "suit": 1}],
>   "opponent_counts": [13, 12, 11],
>   "current_trick": [],
>   "player_id": 0,
>   "num_players": 4,
>   "discard_pile": []
> }
> ```
> `rank`: 3–15 (2 = 15) · `suit`: 1=♠ 2=♣ 3=♦ 4=♥

---

## ☁️ Chạy trên Kaggle (GPU T4 — miễn phí)

### Bước 1 — Upload code lên Kaggle Dataset

Zip toàn bộ source (không bao gồm `venv/`, `checkpoints/`, `.git/`) rồi tạo **Kaggle Dataset** mới.

### Bước 2 — Tạo Notebook mới, chọn GPU T4 x2

Dán đoạn code sau vào cell đầu tiên:

```python
import subprocess, sys, os

# Mount dataset (thay YOUR_USERNAME/tienlen-ai bằng tên dataset của bạn)
DATASET = "YOUR_USERNAME/tienlen-ai"

# Cài thư viện
subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                "/kaggle/input/tienlen-ai/requirements.txt", "-q"])

# Copy code ra working dir để import được
os.makedirs("/kaggle/working/tienlen", exist_ok=True)
subprocess.run(["cp", "-r", "/kaggle/input/tienlen-ai/.", "/kaggle/working/tienlen/"])
os.chdir("/kaggle/working/tienlen")
sys.path.insert(0, "/kaggle/working/tienlen")
```

```python
# Train chính
subprocess.run([
    sys.executable, "-m", "train.train_loop",
    "--episodes",   "30000",
    "--save-every", "3000",
    "--batch-size", "256",
    "--lr",         "3e-4",
    "--gamma",      "0.99",
    "--lam",        "0.95",
    "--ppo-epochs", "4",
    "--device",     "cuda",
])
```

```python
# Download checkpoint sau khi train xong
from kaggle_secrets import UserSecretsClient  # nếu cần
import shutil

# Copy checkpoint ra output để download
shutil.copy("checkpoints/latest.pt", "/kaggle/working/latest.pt")
print("✅ Checkpoint saved to /kaggle/working/latest.pt")
```

### Các tham số CLI (train_loop.py)

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--episodes` | 3000 | Số ván train |
| `--batch-size` | 128 | Batch size PPO |
| `--lr` | 2.5e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--lam` | 0.95 | GAE lambda |
| `--ppo-epochs` | 4 | Số lần update mỗi batch |
| `--save-every` | 100 | Lưu checkpoint mỗi N episodes |
| `--device` | auto | `cpu` / `cuda` / `auto` |

---

## 🏗️ Kiến trúc State (183 dims)

| Thành phần | Dims | Mô tả |
|---|---|---|
| Hand | 52 | Binary: lá nào đang cầm |
| Hand stats | 13 | Số lá, heo, đôi, sảnh tiềm năng... |
| Current trick | 18 | Type, rank, suit, độ mạnh |
| Opponents | 30 | Số lá, mức nguy hiểm, ước tính bài mạnh |
| Discard pile | 52 | Binary: lá nào đã ra |
| Discard stats | 18 | Heo đã ra, tứ quý, tỉ lệ bài, nhóm rank |
| **Total** | **183** | |

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
