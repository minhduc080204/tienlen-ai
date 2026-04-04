import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, logprob, reward, done, value, action_mask):
        # Đảm bảo state là numpy
        self.states.append(state)
        self.actions.append(action)

        # Chuyển Logprob và Value về CPU và detach khỏi đồ thị tính toán
        if torch.is_tensor(logprob):
            logprob = logprob.detach().cpu()
        if torch.is_tensor(value):
            value = value.detach().cpu()

        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

        # Xử lý Action Mask: Chuyển từ Tensor GPU -> CPU -> Numpy
        if torch.is_tensor(action_mask):
            action_mask = action_mask.detach().cpu().numpy()

        self.action_masks.append(
            np.asarray(action_mask, dtype=np.bool_)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def __len__(self):
        return len(self.states)

    def compute_gae(self, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0

        # Vì ta đã lưu values là CPU Tensor ở bước add, nên ở đây xử lý rất mượt
        values = [v.flatten() for v in self.values]
        values.append(torch.tensor([0.0])) # V(s_{T+1}) = 0

        for t in reversed(range(len(self.rewards))):
            # values[t+1] và values[t] bây giờ đều là CPU Tensor
            delta = (
                self.rewards[t]
                + gamma * values[t + 1] * (1 - int(self.dones[t]))
                - values[t]
            )

            gae = delta + gamma * lam * (1 - int(self.dones[t])) * gae
            advantages.insert(0, gae)

        # Trả về các Tensor đã được gộp lại
        advantages = torch.stack(advantages).flatten()
        returns = advantages + torch.stack(values[:-1]).flatten()

        return advantages, returns
