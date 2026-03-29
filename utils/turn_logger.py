def format_cards(cards):
    return "[" + ", ".join(str(c) for c in cards) + "]"


def log_turn(episode, turn, env, action_spec, action_cards):
    state = env.state
    player = state.current_player
    hand = state.hands[player]
    spec_str = action_spec if action_spec else "N/A (rule bot)"
    print(
        f"\n[EP {episode} | TURN {turn}]"
        f"\n Player={player}"
        f"\n Hand before: {format_cards(hand)}"
        f"\n Current trick: {format_cards(state.current_trick) if state.current_trick else 'None'}"
        f"\n Action spec: {spec_str}"
        f"\n Action card: {format_cards(action_cards) if action_cards else 'PASS'}"
    )


def log_turn_result(reward, done):
    print(
        f" Reward={reward:.3f}"
        f"\n Done={done}"
        f"\n-----------------------------"
    )
