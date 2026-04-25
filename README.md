# Tien Len AI (PPO)

An end-to-end reinforcement learning project for **Tien Len Mien Nam (Vietnamese Southern Poker)** using **Proximal Policy Optimization (PPO)**.

This repository includes:

1. A full game environment and rule engine
2. Multi-phase PPO training pipeline
3. FastAPI inference service for real-time action prediction

## Highlights

- **235-dim state encoder** with hand, current trick, opponent signals, and discard memory
- **Action masking** so the model only chooses legal moves
- **Multi-phase training**: warm-up vs RuleBot, self-play, and shared-policy learning
- **Checkpointing** and training metrics logging

## State Representation (235 dims)

| Component | Dims | Description |
|---|---:|---|
| Hand | 65 | 52-bit card presence + 13 hand statistics |
| Current Trick | 70 | 52-bit trick presence + 18 trick features |
| Opponent Info | 30 | Opponent card counts, danger score, passed flags |
| Discard Pile | 70 | 52-bit seen cards + 18 aggregate features |
| **Total** | **235** | |

## Project Structure

```text
tienlen-ai/
├── action/       # Action space and legal action mask
├── bots/         # Rule-based opponents for warm-up phase
├── core/         # Cards, deck, rules, action resolution
├── env/          # TienLen environment and rewards
├── inference/    # FastAPI model-serving API
├── rl/           # PPO model, agent, and rollout buffer
├── state/        # State encoding (235 dims)
├── train/        # Training loops and config
├── checkpoints/  # Saved model weights
└── logs/         # Training metrics/logs
```

## Installation

```bash
python -m venv venv
```

Activate the environment:

- **Windows (PowerShell)**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**
  ```bash
  source venv/bin/activate
  ```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

### Quick local run (CPU)

```bash
python -m train.train_loop --episodes 500 --device cpu
```

### Longer run (GPU)

```bash
python -m train.train_loop --episodes 100000 --device cuda
```

### Initialize from an existing checkpoint

```bash
python -m train.train_loop --episodes 3000 --init-model-path checkpoints/latest.pt
```

### Alternative shared-model training entrypoint

```bash
python -m train.train_loop_share_model --episodes 5000 --device auto
```

## Run Inference API

Start the FastAPI server:

```bash
uvicorn inference.ai_service:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Sample prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hand": [{"rank": 3, "suit": 1}, {"rank": 7, "suit": 4}, {"rank": 15, "suit": 2}],
    "opponent_counts": [13, 13, 13],
    "current_trick": [],
    "discard_pile": [],
    "player_id": 0,
    "num_players": 4,
    "inference_mode": "greedy",
    "temperature": 1.0,
    "top_k_actions": 3
  }'
```

> Card input format uses **rank 3-15** (where 2 is 15) and **suit 1-4**.

## Tests

```bash
pytest
```
