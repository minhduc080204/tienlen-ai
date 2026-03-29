import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
import os
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 PPO using device:", device)

class PPOAgent:
    def __init__(self, model, optimizer, gamma=0.99, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def act(self, state, action_mask):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state = state.to(device)

        state = state.unsqueeze(0)  # (1, state_dim)


        logits, value = self.model(state)

        # 👉 đảm bảo shape (1, action_dim)
        logits = logits.squeeze(0)

        action_mask = action_mask.detach().bool().to(device)

        # 🔒 BẮT BUỘC: không được all-zero
        if action_mask.sum() == 0:
            raise RuntimeError("❌ action_mask is all zero")

        # ✅ MASK ĐÚNG CÁCH
        masked_logits = logits.masked_fill(~action_mask, -1e9)

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value.squeeze(0)


    # =========================
    # PPO UPDATE
    # =========================
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
        device = next(self.model.parameters()).device

        # ========= CONVERT INPUTS =========
        states = torch.from_numpy(np.array(states)).float().to(device)          # (B, state_dim)
        actions = torch.tensor(actions, dtype=torch.long, device=device)        # (B,)
        old_logprobs = torch.stack(old_logprobs).detach().to(device)             # (B,)
        returns = returns.detach().to(device)                                    # (B,)

        advantages = advantages.detach().to(device).float()

        # normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages * 0.0

        # ========= ACTION MASKS =========
        action_masks = torch.from_numpy(
            np.stack(action_masks)
        ).bool().to(device)

        # FIX SHAPE (B, 1, action_dim) -> (B, action_dim)
        if action_masks.dim() == 3:
            action_masks = action_masks.squeeze(1)

        assert action_masks.dim() == 2, f"action_masks shape error: {action_masks.shape}"
        assert action_masks.sum(dim=1).min() > 0, "Found all-zero action mask"

        B = states.size(0)

        # ========= PPO OPTIMIZATION =========
        for _ in range(epochs):
            sampler = BatchSampler(
                SubsetRandomSampler(range(B)),
                batch_size,
                drop_last=False
            )

            for idx in sampler:
                idx = torch.tensor(idx, device=device)

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
                batch_returns = batch_returns.view(-1)
                values = values.view(-1)

                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
