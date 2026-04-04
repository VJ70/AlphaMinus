"""
Replay buffer storing self-play experience tuples (s, π, z).

s: encoded board state  — shape (8, 6, 7)
π: MCTS visit distribution  — shape (7,)
z: game outcome from perspective of the player who moved  — scalar in {-1, 0, 1}
"""

import random
import numpy as np
from collections import deque
from typing import List, Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, game_data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add all (s, π, z) tuples from a completed game."""
        for item in game_data:
            self.buffer.append(item)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, pis, values = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(pis).astype(np.float32),
            np.array(values, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_samples: int) -> bool:
        return len(self) >= min_samples