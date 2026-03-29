# train/train_loop.py
import torch

from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE

from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from rl.buffer import RolloutBuffer
from env.reward import compute_reward

from core.action_executor import resolve_action
from env.step_result import StepResult

from bots.rule_bot import RuleBot
from state.state_dim import STATE_DIM
import argparse
import train.config as config
from utils.logger import setup_logger
from utils.turn_logger import log_turn
import os
from core.move_type import MoveType
from core.rules import (
    get_legal_moves,
)

def parse_args():
    parser = argparse.ArgumentParser("PPO Train Tiến Lên")

    parser.add_argument("--episodes", type=int, default=config.MAX_EPISODES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)

    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--lam", type=float, default=config.LAMBDA)

    parser.add_argument("--ppo-epochs", type=int, default=config.PPO_EPOCHS)

    parser.add_argument("--save-every", type=int, default=config.SAVE_EVERY)
    parser.add_argument("--eval-every", type=int, default=config.EVAL_EVERY)

    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def train():
    logger = setup_logger(
        name="ppo_train",
        log_dir="logs"
    )

    args = parse_args()
    MAX_EPISODES = args.episodes
    BATCH_SIZE = args.batch_size
    LR = args.lr
    GAMMA = args.gamma
    LAMBDA = args.lam
    PPO_EPOCHS = args.ppo_epochs
    SAVE_EVERY=args.save_every

    NUM_PLAYERS=config.NUM_PLAYERS
    MAX_TURNS_PER_EP=config.MAX_TURNS_PER_EP
    AI_PLAYER_ID=config.AI_PLAYER_ID

    LOG_TURN = config.LOG_TURN
    LOG_TURN_EPISODE = config.LOG_TURN_EPISODE
    
    print("🚀 Start PPO Training")

    # =========================
    # 1️⃣ INIT
    # =========================
    env = TienLenEnv(num_players=NUM_PLAYERS)

    model = TienLenPolicy(
        state_dim=STATE_DIM,
        action_dim=len(ACTION_SPACE)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        gamma=GAMMA,
        clip_eps=0.2
    )

    latest_ckpt = "checkpoints/latest.pt"

    if os.path.exists(latest_ckpt):
        print("🔄 Loading checkpoint...")
        agent.load(latest_ckpt)

    rule_bot = RuleBot(player_id=1)
    buffer = RolloutBuffer()

    # =========================
    # 2️⃣ TRAIN LOOP
    # =========================
    for episode in range(1, MAX_EPISODES + 1):

        state = env.reset()

        # instant win → skip
        if state.finished:
            continue

        buffer.clear()
        done = False
        turn_count = 0

        # =========================
        # 3️⃣ PLAY ONE GAME
        # =========================
        while not done:
            turn_count += 1
            current_pid = env.state.current_player
            if turn_count >= MAX_TURNS_PER_EP:
                print(f"⚠️ Episode {episode} force stop (turn limit)")
                break

            # =====================
            # AI TURN
            # =====================
            if current_pid == AI_PLAYER_ID:
                player_hand = state.hands[AI_PLAYER_ID]                

                opponent_hands = [
                    hand
                    for pid, hand in enumerate(state.hands)
                    if pid != AI_PLAYER_ID
                ]

                # discard_pile = env.discard_pile   # hoặc [] nếu chưa có
                discard_pile = []   # hoặc [] nếu chưa có

                state_vec = encode_state(
                    hand=player_hand,
                    discard_pile=discard_pile,
                    opponent_hands=opponent_hands,
                    current_trick=state.current_trick,
                    player_id=AI_PLAYER_ID,
                    num_players=NUM_PLAYERS
                )

                state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0)
                # state_tensor = torch.tensor(
                #     state_vec,
                #     dtype=torch.float32
                # ).unsqueeze(0)   # (1, STATE_DIM)

                # action_mask = build_action_mask(
                #     hand=env.state.hands[AI_PLAYER_ID],
                #     current_trick=env.state.current_trick
                # )

                legal_moves = get_legal_moves(
                    hand=env.state.hands[AI_PLAYER_ID],
                    current_trick=env.state.current_trick
                )

                action_mask = build_action_mask_from_legal_moves(
                    legal_moves=legal_moves,
                    action_space=ACTION_SPACE
                )

                action_mask = torch.from_numpy(action_mask).unsqueeze(0)
                # action_mask = torch.tensor(
                #     action_mask,
                #     dtype=torch.float32
                # ).unsqueeze(0)   # (1, action_dim)


                action_id, logprob, value = agent.act(
                    state_tensor,
                    action_mask
                )

                action_spec = ACTION_SPACE[action_id]

                action_cards = resolve_action(
                    action_spec=action_spec,
                    hand=env.state.hands[AI_PLAYER_ID],
                    current_trick=env.state.current_trick
                )
                # action_cards = build_cards_from_spec(
                #     spec=action_spec,
                #     hand=env.state.hands[AI_PLAYER_ID],
                #     current_trick=env.state.current_trick
                # )
                if LOG_TURN and episode == LOG_TURN_EPISODE:
                    log_turn(
                        episode=episode,
                        turn=turn_count,
                        env=env,
                        action_spec=action_spec,
                        action_cards=action_cards
                    )


            # =====================
            # OPPONENT TURN
            # =====================
            else:
                action_cards = rule_bot.select_action(
                    state,
                    current_pid
                )
                if LOG_TURN and episode == LOG_TURN_EPISODE:
                    log_turn(
                        episode=episode,
                        turn=turn_count,
                        env=env,
                        action_spec=None,
                        action_cards=action_cards
                    )

            step_result = env.step(action_cards)

            reward = step_result.reward

            # =====================
            # STORE ROLLOUT (AI ONLY)
            # =====================
            if current_pid == AI_PLAYER_ID:
                buffer.add(
                    state=state_vec,
                    action=action_id,
                    logprob=logprob.detach(),
                    reward=reward,
                    done=step_result.done,
                    value=value.detach(),
                    action_mask=action_mask
                )

            state = step_result.state
            done = step_result.done

        # =========================
        # 4️⃣ PPO UPDATE
        # =========================
        if len(buffer) > 0:
            advantages, returns = buffer.compute_gae(
                gamma=GAMMA,
                lam=LAMBDA
            )

            agent.update(
                states=buffer.states,
                actions=buffer.actions,
                old_logprobs=buffer.logprobs,
                returns=returns,
                advantages=advantages,
                action_masks=buffer.action_masks,
                epochs=PPO_EPOCHS,
                batch_size=BATCH_SIZE
            )

            buffer.clear()


        # =========================
        # 5️⃣ LOG
        # =========================
        if episode % 1 == 0:
            win = state.winner == AI_PLAYER_ID
            print(
                f"[Episode {episode}] "
                f"Turns={turn_count} "
                f"Win={win}"
            )
            logger.info(
                f"Episode={episode} "
                f"Turns={turn_count} "
                f"Win={win} "
                f"TotalReward={reward:.3f}"
            )


        # =========================
        # 6️⃣ SAVE CHECKPOINT
        # =========================
        if episode % SAVE_EVERY == 0:
            path = f"checkpoints/ppo_ep{episode}.pt"
            agent.save(path)
            print(f"💾 Saved checkpoint: {path}")

    print("✅ Training finished")


if __name__ == "__main__":
    train()
