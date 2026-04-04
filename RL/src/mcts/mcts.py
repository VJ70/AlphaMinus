"""
Monte Carlo Tree Search for Model.

Each node stores:
  N  - visit count
  W  - total value accumulated
  Q  - mean value (W / N)
  P  - prior probability from neural network policy head

Selection uses the PUCT formula:
  UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N(s, a))
"""

import math
import numpy as np
from typing import Dict, Optional, List
import torch

from src.game.connect4 import Connect4


class MCTSNode:
    __slots__ = ("state", "prior", "N", "W", "Q", "children", "is_terminal")

    def __init__(self, state: Connect4, prior: float = 0.0):
        self.state = state
        self.prior = prior
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_terminal: bool = state.done

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def ucb_score(self, parent_N: int, c_puct: float) -> float:
        """PUCT score used in AlphaZero."""
        exploration = c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return self.Q + exploration

    def update(self, value: float):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class MCTS:
    def __init__(
        self,
        net,
        n_simulations: int = 800,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "cpu",
    ):
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    @torch.no_grad()
    def _evaluate(self, state: Connect4):
        """Query neural net for (policy, value) at given state."""
        x = torch.tensor(state.encode(), dtype=torch.float32).unsqueeze(0).to(self.device)
        log_p, v = self.net(x)
        policy = torch.exp(log_p).squeeze(0).cpu().numpy()  # shape: (n_actions,)
        value = v.squeeze().item()
        # Mask illegal moves and renormalize
        legal = state.legal_moves()
        mask = np.zeros(Connect4.COLS, dtype=np.float32)
        mask[legal] = 1.0
        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = mask / mask.sum()
        return policy, value

    def _select(self, node: MCTSNode) -> List["MCTSNode"]:
        """Traverse tree from root to a leaf using UCB."""
        path = [node]
        while node.is_expanded() and not node.is_terminal:
            parent_N = node.N
            best_score = -float("inf")
            best_action = None
            for action, child in node.children.items():
                score = child.ucb_score(parent_N, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action
            node = node.children[best_action]
            path.append(node)
        return path

    def _expand(self, node: MCTSNode, policy: np.ndarray):
        """Add children for all legal moves with their prior probabilities."""
        for action in node.state.legal_moves():
            child_state = node.state.clone()
            child_state.make_move(action)
            node.children[action] = MCTSNode(child_state, prior=float(policy[action]))

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Walk back up the path, flipping sign at each player switch."""
        for node in reversed(path):
            node.update(value)
            value = -value  # flip perspective at each ply

    def search(self, root_state: Connect4, temperature: float = 1.0) -> np.ndarray:
        """
        Run n_simulations from root_state and return a visit-count distribution π.
        temperature controls exploration: high T = more uniform, low T = greedy.
        """
        root = MCTSNode(root_state.clone())
        policy, value = self._evaluate(root_state)

        # Add Dirichlet noise at root for exploration during self-play
        if self.dirichlet_epsilon > 0:
            legal = root_state.legal_moves()
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            noisy_policy = policy.copy()
            for i, a in enumerate(legal):
                noisy_policy[a] = (
                    (1 - self.dirichlet_epsilon) * policy[a]
                    + self.dirichlet_epsilon * noise[i]
                )
            policy = noisy_policy

        self._expand(root, policy)

        for _ in range(self.n_simulations):
            path = self._select(root)
            leaf = path[-1]

            if leaf.is_terminal:
                # Terminal node: value from game outcome
                leaf_value = leaf.state.outcome()
            else:
                leaf_policy, leaf_value = self._evaluate(leaf.state)
                self._expand(leaf, leaf_policy)

            self._backpropagate(path, leaf_value)

        # Build π from visit counts
        counts = np.array([
            root.children[a].N if a in root.children else 0
            for a in range(Connect4.COLS)
        ], dtype=np.float32)

        if temperature == 0:
            # Greedy: one-hot on most visited
            pi = np.zeros_like(counts)
            pi[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / temperature)
            pi = counts / counts.sum()

        return pi

    def best_move(self, state: Connect4, temperature: float = 0.0) -> int:
        """Return the best move (greedy by default)."""
        pi = self.search(state, temperature=temperature)
        return int(np.argmax(pi))