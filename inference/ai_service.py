# inference/ai_service.py
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

CHECKPOINT_PATH = "checkpoints/latest.pt"

try:
    agent.load(CHECKPOINT_PATH)
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT_PATH}")
except Exception as e:
    print("⚠️  No checkpoint loaded (fresh model):", e)


# ===== HEALTH CHECK =====
@app.get("/")
def health():
    return {"status": "ok", "state_dim": STATE_DIM, "action_dim": len(ACTION_SPACE)}


def _parse_card(c: dict) -> Card:
    """
    Parse card từ JSON.

    Format mới: {"rank": 3-15, "suit": 1-4}
    Format cũ (backward compat): {"rank": 0-12, "suit": 0-3}
    """
    rank = c["rank"]
    suit = c["suit"]

    # Detect format cũ: rank 0-12, suit 0-3
    if rank <= 12 and suit <= 3:
        return Card.from_old_ints(rank, suit)

    # Format mới: rank 3-15, suit 1-4
    return Card(rank=rank, suit=suit)


# ===== PREDICT API =====
@app.post("/predict")
def predict(data: dict):
    """
    Input:
    {
        "hand": [{"rank": 3, "suit": 1}, ...],    # rank 3-15, suit 1-4
        "opponent_counts": [13, 12, 11],
        "current_trick": [...] | [],
        "player_id": 0,
        "num_players": 4,
        "discard_pile": [...]    # optional, default []
    }

    Output:
    {
        "action_id": int,
        "action_cards": [{"rank": int, "suit": int, "id": int}],
        "message": "Pass" | "Play cards"
    }
    """

    try:
        hand_cards          = [_parse_card(c) for c in data.get("hand", [])]
        current_trick_cards = [_parse_card(c) for c in data.get("current_trick", [])]
        discard_cards       = [_parse_card(c) for c in data.get("discard_pile", [])]

        opponent_counts = data["opponent_counts"]
        player_id       = data["player_id"]
        num_players     = data["num_players"]

        action_id, cards_to_play = predict_action(
            agent=agent,
            device=device,
            hand=hand_cards,
            opponent_counts=opponent_counts,
            current_trick=current_trick_cards if current_trick_cards else None,
            player_id=player_id,
            num_players=num_players,
            discard_pile=discard_cards,
        )

        action_cards = [
            {"rank": card.rank, "suit": card.suit, "id": card.id}
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