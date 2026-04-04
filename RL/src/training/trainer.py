"""
training loop.

Loss = policy_loss + value_loss
  policy_loss: cross-entropy between MCTS π and network log p
  value_loss:  MSE between game outcome z and network value v
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List

from src.network.model import AlphaZeroNet
from src.training.replay_buffer import ReplayBuffer


def compute_loss(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    target_pi: torch.Tensor,
    target_z: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined AlphaZero loss.

    policy_loss: -sum(π * log p)  — cross entropy
    value_loss:  mean((z - v)^2)  — mean squared error
    """
    policy_loss = -(target_pi * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value.squeeze(1), target_z)
    total_loss = policy_loss + value_loss
    return {
        "total": total_loss,
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
    }


class Trainer:
    def __init__(
        self,
        net: AlphaZeroNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        n_epochs_per_iter: int = 5,
        device: str = "cpu",
        checkpoint_dir: str = "results/checkpoints",
    ):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.n_epochs_per_iter = n_epochs_per_iter
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.history: List[Dict] = []

    def train_on_buffer(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Run n_epochs_per_iter passes over the buffer and return mean losses."""
        self.net.train()
        epoch_losses = {"total": [], "policy": [], "value": []}

        for epoch in range(self.n_epochs_per_iter):
            states, pis, zs = buffer.sample(min(self.batch_size, len(buffer)))

            s = torch.tensor(states, dtype=torch.float32).to(self.device)
            pi = torch.tensor(pis, dtype=torch.float32).to(self.device)
            z = torch.tensor(zs, dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()
            log_p, v = self.net(s)
            losses = compute_loss(log_p, v, pi, z)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in epoch_losses:
                epoch_losses[k].append(losses[k].item())

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    def save_checkpoint(self, iteration: int, extra: Dict = None):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:04d}.pt")
        payload = {
            "iteration": iteration,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.history = payload.get("history", [])
        return payload["iteration"]