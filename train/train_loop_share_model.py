import torch
import os
import argparse
import train.config as config
from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from rl.buffer import RolloutBuffer
from core.action_executor import resolve_action
from bots.rule_bot import RuleBot
from state.state_dim import STATE_DIM
from utils.logger import setup_logger
from utils.turn_logger import log_turn
from core.rules import get_legal_moves
from utils.metrics import MetricTracker
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("PPO Shared Model Train Tiến Lên")
    parser.add_argument("--episodes", type=int, default=config.MAX_EPISODES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--gamma", type=float, default=config.GAMMA)
    parser.add_argument("--lam", type=float, default=config.LAMBDA)
    parser.add_argument("--ppo-epochs", type=int, default=config.PPO_EPOCHS)
    parser.add_argument("--save-every", type=int, default=config.SAVE_EVERY)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def train():
    logger = setup_logger(name="ppo_shared_train", log_dir="logs")
    args = parse_args()
    tracker = MetricTracker(log_dir="logs", filename_prefix="shared_metrics")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt = os.path.join(checkpoint_dir, "latest.pt")
    has_checkpoint = os.path.exists(latest_ckpt)

    shared_model = TienLenPolicy(state_dim=STATE_DIM, action_dim=len(ACTION_SPACE)).to(device)
    shared_opt = torch.optim.Adam(shared_model.parameters(), lr=args.lr)
    shared_agent = PPOAgent(model=shared_model, optimizer=shared_opt, gamma=args.gamma, clip_eps=0.2)

    if has_checkpoint:
        print(f"🔄 Loading shared checkpoint: {latest_ckpt}")
        shared_agent.load(latest_ckpt)

    env = TienLenEnv(num_players=config.NUM_PLAYERS)
    player_buffers = [RolloutBuffer() for _ in range(config.NUM_PLAYERS)]

    print(f"🚀 Start Shared Model Training. All 4 players learn from the SAME brain.")

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        if state.finished:
            winner = state.winner
            reward_0 = 30.0 if winner == 0 else -30.0
            tracker.record_episode(episode, winner, reward_0, 0)
            continue

        for b in player_buffers: b.clear()
        done = False
        turn_count = 0
        ep_reward_0 = 0

        while not done:
            turn_count += 1
            current_pid = env.state.current_player
            if turn_count >= config.MAX_TURNS_PER_EP: break

            opponent_counts = [len(hand) for pid, hand in enumerate(state.hands) if pid != current_pid]
            state_vec = encode_state(
                hand=state.hands[current_pid],
                discard_pile=[],
                opponent_counts=opponent_counts,
                current_trick=state.current_trick,
                player_id=current_pid,
                num_players=config.NUM_PLAYERS
            )

            state_tensor = torch.from_numpy(state_vec).float().to(device).unsqueeze(0)
            legal_moves = get_legal_moves(hand=state.hands[current_pid], current_trick=state.current_trick)
            action_mask = build_action_mask_from_legal_moves(legal_moves=legal_moves, action_space=ACTION_SPACE)
            action_mask_tensor = torch.from_numpy(action_mask).to(device).unsqueeze(0)

            action_id, logprob, value, entropy = shared_agent.act(state_tensor, action_mask_tensor)
            action_spec = ACTION_SPACE[action_id]
            action_cards = resolve_action(action_spec=action_spec, hand=state.hands[current_pid], current_trick=state.current_trick)

            if current_pid == 0:
                tracker.record_move(action_spec.move_type)
                tracker.record_entropy(entropy)

            player_buffers[current_pid].add(
                state=state_vec,
                action=action_id,
                logprob=logprob.detach(),
                reward=0,
                done=False,
                value=value.detach(),
                action_mask=action_mask_tensor
            )

            step_result = env.step(action_cards)
            state = step_result.state
            done = step_result.done

            player_buffers[current_pid].rewards[-1] = step_result.reward

            if current_pid == 0:
                ep_reward_0 += step_result.reward

        winner = state.winner
        for i in range(config.NUM_PLAYERS):
            if len(player_buffers[i]) > 0:
                if i == winner:
                    player_buffers[i].rewards[-1] += 30.0
                    player_buffers[i].dones[-1] = True
                    if i == 0: ep_reward_0 += 30.0
                else:
                    player_buffers[i].rewards[-1] += -30.0
                    player_buffers[i].dones[-1] = True
                    if i == 0: ep_reward_0 += -30.0

        tracker.record_episode(episode, state.winner, ep_reward_0, turn_count)

        all_states, all_actions, all_logprobs, all_returns, all_advantages, all_masks = [], [], [], [], [], []

        for i in range(config.NUM_PLAYERS):
            if len(player_buffers[i]) > 0:
                adv, ret = player_buffers[i].compute_gae(gamma=args.gamma, lam=args.lam)
                all_states.extend(player_buffers[i].states)
                all_actions.extend(player_buffers[i].actions)
                all_logprobs.extend(player_buffers[i].logprobs)
                all_returns.append(ret)
                all_advantages.append(adv)
                all_masks.extend(player_buffers[i].action_masks)

        if all_states:
            shared_agent.update(
                states=all_states,
                actions=all_actions,
                old_logprobs=all_logprobs,
                returns=torch.cat(all_returns),
                advantages=torch.cat(all_advantages),
                action_masks=all_masks,
                epochs=args.ppo_epochs,
                batch_size=args.batch_size
            )

        if episode % 10 == 0:
            summary = tracker.get_summary(last_n=10)
            print(f"🔹 Shared Ep {episode} | Winner: P{state.winner} | WR: {summary['win_rate']:.2f} | Rew: {summary['avg_reward']:.1f} | Entropy: {summary['avg_entropy']:.3f}")
            tracker.save_to_csv(episode, summary['win_rate'], summary['avg_reward'], summary['avg_turns'], summary['avg_entropy'], summary['move_stats'])

        if episode % args.save_every == 0:
            save_path = os.path.join(checkpoint_dir, f"ppo_shared_ep{episode}.pt")
            shared_agent.save(save_path)
            shared_agent.save(latest_ckpt)
            print(f"💾 Checkpoint saved: {save_path}")

    print("✅ Shared Model Training Complete.")

if __name__ == "__main__":
    train()
