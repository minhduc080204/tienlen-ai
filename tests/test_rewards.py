from env.reward import (
    compute_reward,
    action_reward,
    terminal_reward,
    SHAPING_SCALE,
    SHAPING_CLIP,
)
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
    shaped = action_reward(
        action_cards=[Card('3', '♠')],
        prev_state=prev_state,
        next_state=next_state,
        player_id=0,
    )
    expected = terminal_reward(1) + max(-SHAPING_CLIP, min(SHAPING_CLIP, shaped * SHAPING_SCALE))
    assert r == expected

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
    shaped_win = action_reward(
        action_cards=[Card('4', '♠')],
        prev_state=prev_state,
        next_state=next_state,
        player_id=1,
    )
    expected_win = terminal_reward(1) + max(-SHAPING_CLIP, min(SHAPING_CLIP, shaped_win * SHAPING_SCALE))
    assert r == expected_win

    # Reward cho player 0 khi player 1 thắng
    r0 = compute_reward(
        action_cards=[], # Giả sử player 0 pass hoặc không làm gì ở turn cuối
        prev_state=prev_state,
        next_state=next_state,
        done=True,
        player_id=0,
        player_rank=2 # Thua
    )
    shaped_lose = action_reward(
        action_cards=[],
        prev_state=prev_state,
        next_state=next_state,
        player_id=0,
    )
    expected_lose = terminal_reward(2) + max(-SHAPING_CLIP, min(SHAPING_CLIP, shaped_lose * SHAPING_SCALE))
    assert r0 == expected_lose
