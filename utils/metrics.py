import os
import csv
import numpy as np
from datetime import datetime
from core.move_type import MoveType

class MetricTracker:
    def __init__(self, log_dir="logs", filename_prefix="metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = os.path.join(
            log_dir,
            f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        self.headers = [
            "episode", "win_rate", "avg_reward", "avg_turns", "avg_entropy"
        ]
        # Add move types to headers
        for move_type in MoveType:
            self.headers.append(f"move_{move_type.name.lower()}")

        self.history = []

        # Current episode metrics (for training)
        self.reset_episode_stats()

        # Initialize CSV
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def reset_episode_stats(self):
        self.ep_rewards = []
        self.ep_turns = 0
        self.ep_moves = {mt: 0 for mt in MoveType}
        self.ep_entropies = []
        self.ep_winners = []

    def record_move(self, move_type):
        if move_type in self.ep_moves:
            self.ep_moves[move_type] += 1

    def record_entropy(self, entropy):
        self.ep_entropies.append(entropy)

    def record_episode(self, episode, winner, total_reward, turns):
        self.ep_winners.append(winner)
        self.history.append({
            "episode": episode,
            "winner": winner,
            "reward": total_reward,
            "turns": turns,
            "moves": self.ep_moves.copy(),
            "entropy": np.mean(self.ep_entropies) if self.ep_entropies else 0
        })
        self.reset_episode_stats()

    def save_to_csv(self, episode, win_rate, avg_reward, avg_turns, avg_entropy, move_stats):
        row = [episode, win_rate, avg_reward, avg_turns, avg_entropy]
        for move_type in MoveType:
            row.append(move_stats.get(move_type, 0))

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_summary(self, last_n=100):
        if not self.history:
            return None

        recent = self.history[-last_n:]

        win_rate = sum(1 for h in recent if h["winner"] == 0) / len(recent)
        avg_reward = np.mean([h["reward"] for h in recent])
        avg_turns = np.mean([h["turns"] for h in recent])
        avg_entropy = np.mean([h["entropy"] for h in recent])

        move_stats = {mt: 0 for mt in MoveType}
        for h in recent:
            for mt, count in h["moves"].items():
                move_stats[mt] += count

        # Normalize move stats by number of episodes
        for mt in move_stats:
            move_stats[mt] /= len(recent)

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_turns": avg_turns,
            "avg_entropy": avg_entropy,
            "move_stats": move_stats
        }
