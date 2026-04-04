# env/tienlen_env.py
import copy
from core.deck import Deck
from core.rules import can_beat
from core.starting_rules import find_starting_player
from core.instant_win import is_six_pairs, is_five_double_straight
from env.game_state import GameState
from env.step_result import StepResult
from env.reward import compute_reward


class TienLenEnv:
    def __init__(self, num_players: int = 4):
        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.state: GameState | None = None

    def reset(self) -> GameState:
        deck = Deck()
        deck.shuffle()
        hands = deck.deal(self.num_players)

        # ăn trắng
        for i, hand in enumerate(hands):
            if is_six_pairs(hand) or is_five_double_straight(hand):
                self.state = GameState(
                    hands=hands,
                    current_player=i,
                    finished=True,
                    winner=i,
                    discard_pile=[],
                )
                return self.state

        start_player = find_starting_player(hands)

        self.state = GameState(
            hands=hands,
            current_player=start_player,
            current_trick=None,
            last_player=None,
            finished=False,
            winner=None,
            discard_pile=[],
        )
        return self.state

    def step(self, action_cards: list) -> StepResult:
        state = self.state
        assert state is not None
        assert not state.finished

        player = state.current_player
        hand = state.hands[player]

        # COPY STATE TRƯỚC ACTION (CHO REWARD)
        prev_state = copy.deepcopy(state)

        # =====================
        # PASS
        # =====================
        if not action_cards:
            if state.current_trick is None:
                # không được pass khi không có trick trước
                reward = -0.1
                return StepResult(
                    state=state,
                    reward=-100.0, 
                    done=True,
                    info={"action": "INVALID_PASS_TERMINATED"}
                )

            next_player = (player + 1) % self.num_players

            # Nếu vòng quanh về tới người đánh trick → clear trick
            if next_player == state.last_player:
                state.current_trick = None
                state.last_player = None

            state.current_player = next_player

            reward = compute_reward(
                action_cards=[],
                prev_state=prev_state,
                next_state=state,
                done=False,
                player_id=player,
                player_rank=None
            )

            return StepResult(
                state=state,
                reward=reward,
                done=False,
                info={"action": "PASS"}
            )

        # =====================
        # CHECK HỢP LỆ
        # =====================
        if state.current_trick is not None:
            assert can_beat(state.current_trick, action_cards)

        # =====================
        # REMOVE BÀI KHỎI TAY
        # =====================
        for c in action_cards:
            for h in hand:
                if h.rank == c.rank and h.suit == c.suit:
                    hand.remove(h)
                    break
            else:
                raise RuntimeError(
                    f"❌ Card {c} not found in player {player}'s hand"
                )

        # Tích lũy vào discard pile
        state.discard_pile.extend(action_cards)

        # Cập nhật trick
        state.current_trick = action_cards
        state.last_player = player

        # =====================
        # CHECK WIN
        # =====================
        done = len(hand) == 0
        if done:
            state.finished = True
            state.winner = player
        else:
            state.current_player = (player + 1) % self.num_players

        # =====================
        # REWARD
        # =====================
        reward = compute_reward(
            action_cards=action_cards,
            prev_state=prev_state,
            next_state=state,
            done=done,
            player_id=player,
            player_rank=1 if done else None
        )

        return StepResult(
            state=state,
            reward=reward,
            done=done,
            info={"winner": player} if done else {}
        )
