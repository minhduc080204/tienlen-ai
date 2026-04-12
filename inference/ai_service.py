# inference/ai_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import time
import logging
from typing import List, Optional, Literal

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
    description="Refactored API for smarter Tiến Lên model (235 dims)",
    version="2.0.0"
)

# ======================
# SCHEMAS
# ======================
class CardInput(BaseModel):
    rank: int = Field(..., description="New format: 3-15 (2=15). Legacy format: 0-12.")
    suit: int = Field(..., description="New format: 1-4 (♠=1, ♣=2, ♦=3, ♥=4). Legacy format: 0-3.")

class PredictRequest(BaseModel):
    hand: List[CardInput]
    opponent_counts: List[int]
    current_trick: Optional[List[CardInput]] = []
    player_id: int
    num_players: int
    discard_pile: Optional[List[CardInput]] = []
    inference_mode: Literal["greedy", "sample"] = "greedy"
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_k_actions: int = Field(3, ge=1, le=10)

class PredictResponse(BaseModel):
    action_id: int
    action_cards: List[dict]
    value_estimate: float
    action_confidence: float
    top_actions: List[dict]
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
# CHECKPOINT_PATH = "checkpoints/best_model.pt"

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
    """Parse ưu tiên format mới; chỉ fallback format cũ khi input không mơ hồ."""
    if 3 <= c.rank <= 15 and 1 <= c.suit <= 4:
        return Card(rank=c.rank, suit=c.suit)

    # Legacy format cũ (rank 0-12, suit 0-3) chỉ dùng khi rõ ràng không phải format mới.
    if 0 <= c.rank <= 12 and 0 <= c.suit <= 3 and (c.rank <= 2 or c.suit == 0):
        return Card.from_old_ints(c.rank, c.suit)
    raise ValueError(
        f"Invalid or ambiguous card input rank={c.rank}, suit={c.suit}. "
        "Use new format rank 3-15 and suit 1-4."
    )

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
        if req.num_players not in [2, 3, 4]:
            raise HTTPException(status_code=400, detail="num_players must be in [2, 3, 4]")
        if len(req.opponent_counts) != req.num_players - 1:
            raise HTTPException(
                status_code=400,
                detail=f"opponent_counts must have length {req.num_players - 1}",
            )
        if req.player_id < 0 or req.player_id >= req.num_players:
            raise HTTPException(
                status_code=400,
                detail=f"player_id must be in [0, {req.num_players - 1}]",
            )

        # 1. Parse inputs
        hand_cards = [_to_card(c) for c in req.hand]
        trick_cards = [_to_card(c) for c in req.current_trick] if req.current_trick else None
        discard_cards = [_to_card(c) for c in req.discard_pile] if req.discard_pile else []

        # 2. Predict with Agent
        action_id, cards_to_play, value_est, action_confidence, top_actions = predict_action(
            agent=agent,
            device=device,
            hand=hand_cards,
            opponent_counts=req.opponent_counts,
            current_trick=trick_cards,
            player_id=req.player_id,
            num_players=req.num_players,
            discard_pile=discard_cards,
            greedy=req.inference_mode == "greedy",
            temperature=req.temperature,
            top_k=req.top_k_actions,
        )

        # 3. Format output
        output_cards = [
            {"rank": c.rank, "suit": c.suit, "id": c.id}
            for c in cards_to_play
        ]
        output_top_actions = [
            {
                "action_id": aid,
                "move_type": ACTION_SPACE[aid].move_type.name,
                "length": ACTION_SPACE[aid].length,
                "probability": round(float(prob), 6),
            }
            for aid, prob in top_actions
        ]

        duration = (time.time() - start_time) * 1000
        
        return PredictResponse(
            action_id=action_id,
            action_cards=output_cards,
            value_estimate=float(value_est),
            action_confidence=float(action_confidence),
            top_actions=output_top_actions,
            message="Pass" if not output_cards else f"Play {len(output_cards)} cards",
            process_time_ms=round(duration, 2)
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
