import torch
import os
import argparse
import numpy as np
from collections import deque
from typing import List

import train.config as config
from env.tienlen_env import TienLenEnv
from state.state_encoder import encode_state
from state.state_dim import STATE_DIM
from action.action_mask import build_action_mask_from_legal_moves
from action.action_space import ACTION_SPACE
from core.action_executor import resolve_action
from core.rules import get_legal_moves

from rl.agent import PPOAgent
from rl.model import TienLenPolicy
from rl.buffer import RolloutBuffer
from bots.rule_bot import RuleBot
from utils.logger import setup_logger
from utils.metrics import MetricTracker

def parse_args():
    parser = argparse.ArgumentParser("PPO Multi-Phase Training for Tien Len")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def setup_agents(device, lr, has_checkpoint=False, checkpoint_path=None):
    """Khởi tạo model và agent chính."""
    model = TienLenPolicy(state_dim=STATE_DIM, action_dim=len(ACTION_SPACE)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    main_agent = PPOAgent(model=model, optimizer=optimizer, gamma=config.GAMMA, clip_eps=0.2)
    
    if has_checkpoint and checkpoint_path:
        main_agent.load(checkpoint_path)
        print(f"🔄 Loaded checkpoint: {checkpoint_path}")
    
    return main_agent

def train():
    args = parse_args()
    device = torch.device(args.device)
    logger = setup_logger(name="ppo_multi_phase", log_dir="logs")
    tracker = MetricTracker(log_dir="logs")
    
    # Checkpoints path
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    
    # 1. Khởi tạo Agent chính
    main_agent = setup_agents(device, config.LR, os.path.exists(latest_path), latest_path)
    
    # 2. Tracking win rate
    win_history = deque(maxlen=config.WINDOW_SIZE)
    best_win_rate = 0.0
    
    env = TienLenEnv(num_players=config.NUM_PLAYERS)
    
    # Buffer tích lũy để update model
    cumulative_buffer = RolloutBuffer()

    print(f"🚀 Starting Multi-Phase Training on {device}")

    for episode in range(1, args.episodes + 1):
        # Xác định Giai đoạn (Phase)
        if episode <= config.WARMUP_EPISODES:
            phase = 1 # Warm-up: vs RuleBot
        elif episode <= config.SELF_PLAY_EPISODES:
            phase = 2 # Self-Play: vs Frozen Version
        else:
            phase = 3 # Shared Model: 4-way PPO update

        # Thiết lập danh sách người chơi cho episode này
        episode_agents = [None] * config.NUM_PLAYERS
        episode_buffers = [RolloutBuffer() for _ in range(config.NUM_PLAYERS)]
        
        # P0 luôn là Main Agent
        episode_agents[0] = main_agent
        
        for i in range(1, config.NUM_PLAYERS):
            if phase == 1:
                episode_agents[i] = RuleBot(player_id=i)
            else:
                # Giai đoạn 2 và 3 đều dùng PPO Agent (Shared Weights)
                episode_agents[i] = main_agent

        # Reset Game
        state = env.reset()
        done = state.finished # Initialize 'done' from 'state.finished'
        turn_count = 0
        ep_reward_0 = 0

        # --- GAME LOOP ---
        while not done and turn_count < config.MAX_TURNS_PER_GAME:
            turn_count += 1
            curr_pid = env.state.current_player
            agent = episode_agents[curr_pid]
            
            if isinstance(agent, PPOAgent):
                # 1. Encoding State
                opp_counts = [len(h) for p, h in enumerate(state.hands) if p != curr_pid]
                state_vec = encode_state(
                    hand=state.hands[curr_pid],
                    discard_pile=state.discard_pile,
                    opponent_counts=opp_counts,
                    current_trick=state.current_trick,
                    player_id=curr_pid,
                    num_players=config.NUM_PLAYERS,
                    passed_players=state.passed_players
                )
                
                # 2. Action Masking
                legal_moves = get_legal_moves(state.hands[curr_pid], state.current_trick)
                mask = build_action_mask_from_legal_moves(legal_moves, ACTION_SPACE)
                
                # 3. Inference
                state_t = torch.as_tensor(state_vec, device=device, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool).unsqueeze(0)
                
                with torch.no_grad():
                    action_id, logprob, val, _ = agent.act(state_t, mask_t)
                
                # 4. Resolve Action & Step
                action_cards = resolve_action(ACTION_SPACE[action_id], state.hands[curr_pid], state.current_trick)
                step_res = env.step(action_cards)
                
                # 5. Store Experience (Phase 1,2: chỉ Player 0 | Phase 3: Tất cả)
                if curr_pid == 0 or phase == 3:
                    episode_buffers[curr_pid].add(
                        state=state_vec,
                        action=action_id,
                        logprob=logprob, # Truyền Tensor
                        reward=step_res.reward,
                        done=step_res.done,
                        value=val,       # Truyền Tensor
                        action_mask=mask
                    )
                
                if curr_pid == 0: ep_reward_0 += step_res.reward
                state = step_res.state
                done = step_res.done
            else:
                # RuleBot Turn
                action_cards = agent.select_action(state, curr_pid)
                step_res = env.step(action_cards)
                state = step_res.state
                done = step_res.done

        # --- END OF EPISODE ---
        winner = state.winner
        win_history.append(1 if winner == 0 else 0)
        
        # Terminal Reward & GAE
        for i in range(config.NUM_PLAYERS):
            if len(episode_buffers[i]) > 0:
                final_reward = 30.0 if i == winner else -30.0
                episode_buffers[i].rewards[-1] += final_reward
                
                # Tính GAE cục bộ cho episode
                adv, ret = episode_buffers[i].compute_gae(config.GAMMA, config.LAMBDA)
                
                # Tích lũy vào Buffer huấn luyện chính
                cumulative_buffer.extend(episode_buffers[i], adv, ret)

        # --- MODEL UPDATE ---
        if len(cumulative_buffer) >= config.BATCH_SIZE:
            # Giai đoạn update sử dụng hàm update chuẩn của PPOAgent
            main_agent.update(
                states=np.array(cumulative_buffer.states),
                actions=torch.tensor(cumulative_buffer.actions, device=device),
                old_logprobs=torch.stack(cumulative_buffer.logprobs).to(device).flatten(),
                returns=torch.stack(cumulative_buffer.returns).to(device).flatten(),
                advantages=torch.stack(cumulative_buffer.advantages).to(device).flatten(),
                action_masks=torch.tensor(np.array(cumulative_buffer.action_masks), device=device),
                epochs=config.PPO_EPOCHS,
                batch_size=config.BATCH_SIZE
            )
            cumulative_buffer.clear()
            if device.type == "cuda": torch.cuda.empty_cache()

        # --- LOGGING ---
        if episode % 20 == 0:
            avg_win_rate = sum(win_history) / len(win_history) if win_history else 0
            print(f"Ep {episode} [{phase}] | WR: {avg_win_rate:.2f} | Best WR: {best_win_rate:.2f} | Rew: {ep_reward_0:.1f}")
            
            main_agent.save(latest_path)
            if avg_win_rate > best_win_rate and episode > config.WINDOW_SIZE:
                best_win_rate = avg_win_rate
                if best_win_rate >= config.WIN_RATE_THRESHOLD:
                    main_agent.save(best_path)
                    print(f"⭐ New Best Model: {best_win_rate:.2f}")

    print("✅ Training Finished.")

if __name__ == "__main__":
    train()
