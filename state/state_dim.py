# state/state_dim.py
from state.hand_encoder import HAND_VECTOR_SIZE
from state.trick_encoder import TRICK_VECTOR_SIZE
from state.opponent_encoder import OPPONENT_VECTOR_SIZE
from state.discard_encoder import DISCARD_VECTOR_SIZE

# 65 + 18 + 30 + 70 = 183
STATE_DIM = HAND_VECTOR_SIZE + TRICK_VECTOR_SIZE + OPPONENT_VECTOR_SIZE + DISCARD_VECTOR_SIZE
