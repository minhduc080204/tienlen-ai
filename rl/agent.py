import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import os

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

class PPOAgent:
    def __init__(
        self,
        model,
        optimizer,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        target_kl=0.015,
        value_clip_eps=0.2,
        normalize_returns=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.value_clip_eps = value_clip_eps
        self.normalize_returns = normalize_returns

    def get_device(self):
        return next(self.model.parameters()).device

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        device = self.get_device()
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def set_entropy_coef(self, entropy_coef: float):
        self.entropy_coef = float(entropy_coef)

    def act(self, state, action_mask, greedy=False):
        device = self.get_device()
        
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.model(state)
        logits = logits.squeeze(0)
        value = value.squeeze(0)

        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        else:
            action_mask = action_mask.to(device).bool()
            
        if action_mask.dim() > 1:
            action_mask = action_mask.flatten()

        masked_logits = logits.masked_fill(~action_mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        
        if greedy:
            action = torch.argmax(masked_logits)
        else:
            action = dist.sample()
            
        entropy = dist.entropy()

        return action.item(), dist.log_prob(action), value, entropy.item()

    def update(
        self,
        states,
        actions,
        old_logprobs,
        returns,
        advantages,
        action_masks,
        old_values=None,
        epochs=4,
        batch_size=64
    ):
        device = self.get_device()

        # Tối ưu hóa việc chuyển đổi Tensor: Tránh gọi np.array() lên GPU Tensor
        def to_device_tensor(data, dtype):
            if torch.is_tensor(data):
                return data.to(device=device, dtype=dtype)
            return torch.as_tensor(np.array(data), dtype=dtype, device=device)

        states = to_device_tensor(states, torch.float32)
        actions = to_device_tensor(actions, torch.long).flatten()
        
        if isinstance(old_logprobs, list):
            old_logprobs = torch.stack(old_logprobs).to(device).flatten()
        else:
            old_logprobs = to_device_tensor(old_logprobs, torch.float32).flatten()
            
        returns = to_device_tensor(returns, torch.float32).flatten()
        advantages = to_device_tensor(advantages, torch.float32).flatten()
        if old_values is not None:
            if isinstance(old_values, list):
                old_values = torch.stack(old_values).to(device).flatten()
            else:
                old_values = to_device_tensor(old_values, torch.float32).flatten()

        # Normalize returns (reward scale stabilization)
        if self.normalize_returns and returns.numel() > 1:
            ret_std = returns.std()
            if ret_std > 1e-8:
                returns = (returns - returns.mean()) / ret_std
            else:
                returns = returns - returns.mean()

        # Normalize advantages
        if advantages.numel() > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()

        action_masks = to_device_tensor(action_masks, torch.bool)
        if action_masks.dim() == 3:
            action_masks = action_masks.squeeze(1)

        B = states.size(0)

        # ========= PPO OPTIMIZATION =========
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_approx_kl = 0
        num_batches = 0

        for _ in range(epochs):
            sampler = BatchSampler(
                SubsetRandomSampler(range(B)),
                batch_size,
                drop_last=False
            )

            for idx in sampler:
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_masks = action_masks[idx]
                batch_old_values = old_values[idx] if old_values is not None else None

                logits, values = self.model(batch_states)
                values = values.squeeze(-1)

                masked_logits = logits.masked_fill(~batch_masks, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)

                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(logprobs - batch_old_logprobs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                if batch_old_values is not None:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.value_clip_eps,
                        self.value_clip_eps
                    )
                    value_loss_unclipped = (values - batch_returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, batch_returns)

                approx_kl = (batch_old_logprobs - logprobs).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_approx_kl += approx_kl.item()
                num_batches += 1

            if self.target_kl is not None and num_batches > 0:
                mean_approx_kl = total_approx_kl / num_batches
                if mean_approx_kl > self.target_kl:
                    break

        return {
            "policy_loss": total_policy_loss / num_batches if num_batches > 0 else 0,
            "value_loss": total_value_loss / num_batches if num_batches > 0 else 0,
            "entropy_loss": total_entropy_loss / num_batches if num_batches > 0 else 0,
            "approx_kl": total_approx_kl / num_batches if num_batches > 0 else 0
        }
