import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, logprob, reward, done, value, action_mask):
        """Lưu một step vào buffer. Chấp nhận cả Tensor và Numpy/Float."""
        self.states.append(state)
        self.actions.append(action)

        # Chuyển Logprob về CPU Tensor 0-dim
        if torch.is_tensor(logprob):
            logprob = logprob.detach().cpu().view(-1)
        else:
            logprob = torch.tensor([logprob], dtype=torch.float32)
        self.logprobs.append(logprob)

        self.rewards.append(reward)

        # Chuyển Value về CPU Tensor 0-dim
        if torch.is_tensor(value):
            value = value.detach().cpu().view(-1)
        else:
            value = torch.tensor([value], dtype=torch.float32)
        self.values.append(value)

        self.dones.append(done)

        # Xử lý Action Mask: Đảm bảo là Numpy array bool
        if torch.is_tensor(action_mask):
            action_mask = action_mask.detach().cpu().numpy()
        self.action_masks.append(np.asarray(action_mask, dtype=np.bool_).flatten())

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
        
        # Thêm storage cho GAE/Returns khi dùng extend
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.states)

    def extend(self, other_buffer, advantages, returns):
        """Gộp dữ liệu từ buffer khác vào buffer này (Dùng cho Shared Model Phase 3)."""
        self.states.extend(other_buffer.states)
        self.actions.extend(other_buffer.actions)
        self.logprobs.extend(other_buffer.logprobs)
        self.rewards.extend(other_buffer.rewards)
        self.values.extend(other_buffer.values)
        self.dones.extend(other_buffer.dones)
        self.action_masks.extend(other_buffer.action_masks)
        
        # Lưu trữ kết quả tính toán GAE
        if torch.is_tensor(advantages):
            self.advantages.extend(advantages.detach().cpu())
            self.returns.extend(returns.detach().cpu())
        else:
            self.advantages.extend(advantages)
            self.returns.extend(returns)

    def compute_gae(self, gamma=0.99, lam=0.95):
        """Tính toán GAE (Generalized Advantage Estimation)."""
        if not self.rewards:
            return torch.tensor([]), torch.tensor([])

        advantages = []
        gae = 0.0

        # values: list of 1-item tensors
        v_t_plus_1 = torch.tensor([0.0]) # V(s_{T+1}) = 0

        for t in reversed(range(len(self.rewards))):
            v_t = self.values[t]
            mask = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * v_t_plus_1 * mask - v_t
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
            v_t_plus_1 = v_t

        # Trả về các Tensor đã được gộp lại
        adv_tensor = torch.stack(advantages).flatten()
        ret_tensor = adv_tensor + torch.stack(self.values).flatten()

        return adv_tensor, ret_tensor
