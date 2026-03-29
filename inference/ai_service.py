from fastapi import FastAPI, HTTPException
import torch

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

CHECKPOINT_PATH = "checkpoints/latest.pt"

try:
    agent.load(CHECKPOINT_PATH)
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT_PATH}")
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
        "opponent_counts": [[...], [...], [...]],
        "current_trick": [...],
        "player_id": int,
        "num_players": int
    }
    """

    try:
        hand = data["hand"]
        opponent_counts = data["opponent_counts"]
        current_trick = data["current_trick"]
        player_id = data["player_id"]
        num_players = data["num_players"]

        action_id = predict_action(
            agent=agent,
            device=device,
            hand=hand,
            opponent_counts=opponent_counts,
            current_trick=current_trick,
            player_id=player_id,
            num_players=num_players
        )

        return {
            "action_id": action_id
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))