# inference/ai_service.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import torch
import time
import logging
from typing import List, Optional

from core.card import Card
from rl.model import TienLenPolicy
from rl.agent import PPOAgent
from state.state_dim import STATE_DIM
from inference.predict import predict_action
from action.action_space import ACTION_SPACE

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TienLenAI")

# ======================
# APP INIT
# ======================
app = FastAPI(
    title="TienLen AI Service",
    description="Refactored API for smarter Tiến Lên model (183 dims)",
    version="2.0.0"
)

# ======================
# SCHEMAS
# ======================
class CardInput(BaseModel):
    rank: int = Field(..., description="Rank 3-15 (2=15)")
    suit: int = Field(..., description="Suit 1-4 (♠=1, ♣=2, ♦=3, ♥=4)")

class PredictRequest(BaseModel):
    hand: List[CardInput]
    opponent_counts: List[int]
    current_trick: Optional[List[CardInput]] = []
    player_id: int
    num_players: int
    discard_pile: Optional[List[CardInput]] = []

class PredictResponse(BaseModel):
    action_id: int
    action_cards: List[dict]
    value_estimate: float
    message: str
    process_time_ms: float

# ======================
# MODEL LOAD
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 AI Service using device: {device}")

model = TienLenPolicy(
    state_dim=STATE_DIM,
    action_dim=len(ACTION_SPACE)
).to(device)

# Dummy optimizer for PPOAgent initialization
optimizer = torch.optim.Adam(model.parameters())
agent = PPOAgent(model=model, optimizer=optimizer)

CHECKPOINT_PATH = "checkpoints/latest.pt"

try:
    # Set weights_only=False for custom policy load
    agent.load(CHECKPOINT_PATH)
    model.eval()
    logger.info(f"✅ Model loaded from {CHECKPOINT_PATH}")
except Exception as e:
    logger.warning(f"⚠️ No checkpoint loaded (using fresh model): {e}")


# ======================
# UTILS
# ======================
def _to_card(c: CardInput) -> Card:
    """Handles both old (0-12) and new (3-15) formats"""
    if c.rank <= 12 and c.suit <= 3:
        return Card.from_old_ints(c.rank, c.suit)
    return Card(rank=c.rank, suit=c.suit)

# ======================
# ENDPOINTS
# ======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "state_dim": STATE_DIM,
        "action_dim": len(ACTION_SPACE),
        "checkpoint": CHECKPOINT_PATH
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    start_time = time.time()
    
    try:
        # 1. Parse inputs
        hand_cards = [_to_card(c) for c in req.hand]
        trick_cards = [_to_card(c) for c in req.current_trick] if req.current_trick else None
        discard_cards = [_to_card(c) for c in req.discard_pile] if req.discard_pile else []

        # 2. Predict with Agent
        action_id, cards_to_play, value_est = predict_action(
            agent=agent,
            device=device,
            hand=hand_cards,
            opponent_counts=req.opponent_counts,
            current_trick=trick_cards,
            player_id=req.player_id,
            num_players=req.num_players,
            discard_pile=discard_cards,
        )

        # 3. Format output
        output_cards = [
            {"rank": c.rank, "suit": c.suit, "id": c.id}
            for c in cards_to_play
        ]

        duration = (time.time() - start_time) * 1000
        
        return PredictResponse(
            action_id=action_id,
            action_cards=output_cards,
            value_estimate=float(value_est),
            message="Pass" if not output_cards else f"Play {len(output_cards)} cards",
            process_time_ms=round(duration, 2)
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))