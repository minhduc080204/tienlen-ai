from env.reward import compute_reward
from env.game_state import GameState
from core.card import Card

def test_win_reward():
    prev_state = GameState(hands=[[Card('3', '♠')], [Card('4', '♠')]], current_player=0)
    next_state = GameState(hands=[[], [Card('4', '♠')]], current_player=1, finished=True, winner=0)
    
    r = compute_reward(
        action_cards=[Card('3', '♠')],
        prev_state=prev_state,
        next_state=next_state,
        done=True,
        player_id=0,
        player_rank=1
    )
    # 30 (win) + 0.05 (play card) = 30.05
    assert r == 30.05

def test_lose_reward():
    prev_state = GameState(hands=[[Card('3', '♠')], [Card('4', '♠')]], current_player=1)
    next_state = GameState(hands=[[Card('3', '♠')], []], current_player=0, finished=True, winner=1)
    
    r = compute_reward(
        action_cards=[Card('4', '♠')],
        prev_state=prev_state,
        next_state=next_state,
        done=True,
        player_id=1,
        player_rank=1 # Player 1 thắng
    )
    assert r == 30.05

    # Reward cho player 0 khi player 1 thắng
    r0 = compute_reward(
        action_cards=[], # Giả sử player 0 pass hoặc không làm gì ở turn cuối
        prev_state=prev_state,
        next_state=next_state,
        done=True,
        player_id=0,
        player_rank=2 # Thua
    )
    # -30 (lose) - 0.2 (pass) = -30.2
    assert r0 == -30.2
