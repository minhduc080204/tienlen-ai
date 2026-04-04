from core.card import Card
from env.tienlen_env import TienLenEnv
from env.game_state import GameState

def test_four_kind_chop_two():
    env = TienLenEnv(2)

    env.state = GameState(
        hands=[
            [Card("6","♠"),Card("6","♣"),Card("6","♦"),Card("6","♥")],
            [Card("2","♠"), Card("3","♦")]   # còn bài
        ],
        current_player=1,
        current_trick=None,
        last_player=None,
        finished=False
    )

    env.step([Card("2","♠")])
    result = env.step([
        Card("6","♠"),Card("6","♣"),Card("6","♦"),Card("6","♥")
    ])

    assert result.done is True
    assert result.state.winner == 0
