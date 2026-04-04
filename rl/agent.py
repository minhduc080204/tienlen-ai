import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import os

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5):
        # We don't set a global device here, we use the model's device
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

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

    def act(self, state, action_mask, greedy=False):
        device = self.get_device()
        
        # Ensure state is a tensor on the correct device
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        # Handle batch dimension if missing
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.model(state)
        logits = logits.squeeze(0)
        value = value.squeeze(0)

        # Ensure action_mask is a boolean tensor
        if not torch.is_tensor(action_mask):
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        else:
            action_mask = action_mask.to(device).bool()
            
        if action_mask.dim() > 1:
            action_mask = action_mask.flatten()

        if action_mask.sum() == 0:
            # Fallback: if no legal moves (shouldn't happen with get_legal_moves), mask nothing
            masked_logits = logits
        else:
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
        epochs=4,
        batch_size=64
    ):
        device = self.get_device()

        # Convert everything to tensors on the correct device
        # Use as_tensor to avoid unnecessary copies if already tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device).flatten()
        
        if isinstance(old_logprobs, list):
            old_logprobs = torch.stack(old_logprobs).to(device).flatten()
        else:
            old_logprobs = torch.as_tensor(old_logprobs, device=device).flatten()
            
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device).flatten()
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device).flatten()

        # Normalize advantages
        if advantages.numel() > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()

        action_masks = torch.as_tensor(np.array(action_masks), dtype=torch.bool, device=device)
        if action_masks.dim() == 3:
            action_masks = action_masks.squeeze(1)

        B = states.size(0)

        for _ in range(epochs):
            sampler = BatchSampler(
                SubsetRandomSampler(range(B)),
                batch_size,
                drop_last=False
            )

            for idx in sampler:
                # Subset indices
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_masks = action_masks[idx]

                # forward
                logits, values = self.model(batch_states)
                values = values.squeeze(-1)

                # mask invalid actions
                masked_logits = logits.masked_fill(~batch_masks, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)

                logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO ratio
                ratios = torch.exp(logprobs - batch_old_logprobs)

                # surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
