import numpy as np
from core.card import Card
from state.state_encoder import encode_state


def test_state_encoder_shape():
    hand = [Card("3","♠"), Card("3","♥")]
    discard = [Card("2","♣")]
    opponents = [5, 2, 7]

    state = encode_state(
        hand=hand,
        discard_pile=discard,
        opponent_counts=opponents,
        current_trick=None,
        player_id=0,
        num_players=4
    )

    assert isinstance(state, np.ndarray)
    assert state.shape == (141,)
