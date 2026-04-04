"""
Neural Network for Connect4.

Architecture:
  - Input: (batch, 8, 6, 7) tensor — encoded board planes
  - Conv block: 128 filters, 3×3, BN, ReLU
  - Residual tower: N residual blocks (each = 2× Conv-BN-ReLU + skip)
  - Policy head: conv → flatten → linear → log_softmax over 7 actions
  - Value head: conv → flatten → linear → tanh scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, board_h: int, board_w: int, n_actions: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_h * board_w, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        return F.log_softmax(self.fc(x), dim=1)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, board_h: int, board_w: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_h * board_w, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class AlphaZeroNet(nn.Module):
    """
    Combined policy-value network.
    Returns (log_policy, value) tuple.
    log_policy: (batch, n_actions) log probabilities
    value:      (batch, 1) in [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 8,
        n_res_blocks: int = 6,
        n_filters: int = 128,
        board_h: int = 6,
        board_w: int = 7,
        n_actions: int = 7,
    ):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, n_filters)
        self.res_tower = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_filters, board_h, board_w, n_actions)
        self.value_head = ValueHead(n_filters, board_h, board_w)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        x = self.res_tower(x)
        log_policy = self.policy_head(x)
        value = self.value_head(x)
        return log_policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)