from fastapi import FastAPI, HTTPException
import torch
from core.card import Card

# ===== IMPORT PROJECT =====
from rl.model import TienLenPolicy
from rl.agent import PPOAgent
from state.state_dim import STATE_DIM

from inference.predict import predict_action
from action.action_space import ACTION_SPACE

# ===== INIT APP =====
app = FastAPI(title="TienLen AI Service")

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 AI Service using device:", device)

# ===== LOAD MODEL =====
model = TienLenPolicy(
    state_dim=STATE_DIM,
    action_dim=len(ACTION_SPACE)
).to(device)

optimizer = torch.optim.Adam(model.parameters())

agent = PPOAgent(model=model, optimizer=optimizer)

EASY_CHECKPOINT_PATH = "checkpoints/latest.pt"
MEDIUM_CHECKPOINT_PATH = "checkpoints/ppo.pt"
HARD_CHECKPOINT_PATH = "checkpoints/ppo_shared.pt"

try:
    agent.load(MEDIUM_CHECKPOINT_PATH)
    model.eval()
    print(f"✅ Model loaded from {MEDIUM_CHECKPOINT_PATH}")
except Exception as e:
    print("❌ Failed to load model:", e)


# ===== HEALTH CHECK =====
@app.get("/")
def health():
    return {"status": "ok"}


# ===== PREDICT API =====
@app.post("/predict")
def predict(data: dict):
    """
    Input:
    {
        "hand": [...],
        "opponent_counts": [1, 2, 3],
        "current_trick": [...],
        "player_id": int,
        "num_players": int
    }
    """

    try:
        # 1. Parse dữ liệu cực kỳ gọn gàng
        hand_cards = [Card.from_ints(c["rank"], c["suit"]) for c in data.get("hand", [])]
        current_trick_cards = [Card.from_ints(c["rank"], c["suit"]) for c in data.get("current_trick", [])]

        opponent_counts = data["opponent_counts"]
        player_id = data["player_id"]
        num_players = data["num_players"]

        action_id, cards_to_play = predict_action(
            agent=agent,
            device=device,
            hand=hand_cards,
            opponent_counts=opponent_counts,
            current_trick=current_trick_cards,
            player_id=player_id,
            num_players=num_players
        )

        action_cards = [
            {"rank": card.rank_value, "suit": card.suit_value} 
            for card in cards_to_play
        ]

        return {
            "action_id": action_id,
            "action_cards": action_cards,
            "message": "Pass" if len(action_cards) == 0 else "Play cards"
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))