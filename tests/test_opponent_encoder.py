from core.card import Card
from state.opponent_encoder import encode_opponents


def test_encode_opponents_shape():
    opponents = [
        5,
        2,
        7,
    ]

    vec = encode_opponents(
        opponent_counts=opponents,
        player_id=0,
        num_players=4,
    )

    assert vec.shape == (12,)
    assert vec[1] == 2 / 13
    assert vec[5] == 1.0  # có người <= 2 lá
