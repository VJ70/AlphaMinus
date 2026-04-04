"""
Self-play: generate training data by having the agent play against itself.

Each game produces a list of (state, π, z) tuples where:
  - state is the encoded board before the move
  - π is the MCTS visit distribution (policy target)
  - z is the final game outcome from that player's perspective
"""

import numpy as np
from typing import List, Tuple

from src.game.connect4 import Connect4
from src.mcts.mcts import MCTS


def self_play_game(
    mcts: MCTS,
    temp_threshold: int = 30,
    temp_high: float = 1.0,
    temp_low: float = 0.1,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Play one full game using MCTS and return training tuples.

    For the first `temp_threshold` moves, use high temperature for exploration.
    After that, use low temperature (near-greedy) for precision.
    """
    game = Connect4()
    memory: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (state, pi, player_who_moved)

    while not game.done:
        temp = temp_high if game.move_count < temp_threshold else temp_low
        pi = mcts.search(game, temperature=temp)

        # Store encoded state from perspective of current player
        memory.append((game.encode().copy(), pi.copy(), game.current_player))

        # Sample move from π
        action = np.random.choice(Connect4.COLS, p=pi)
        game.make_move(action)

    # Assign outcome z to each stored state
    # winner == 0 → draw, else winner is player id
    training_data = []
    for state, pi, player in memory:
        if game.winner == 0:
            z = 0.0
        else:
            z = 1.0 if game.winner == player else -1.0
        training_data.append((state, pi, z))

    return training_data


def run_self_play_games(
    mcts: MCTS,
    n_games: int,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Run multiple self-play games and aggregate results."""
    all_data = []
    wins = {1: 0, -1: 0, 0: 0}

    for i in range(n_games):
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

        # Track outcome from first move perspective (proxy for game result)
        if game_data:
            z_first = game_data[0][2]
            if z_first == 1.0:
                wins[1] += 1
            elif z_first == -1.0:
                wins[-1] += 1
            else:
                wins[0] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Self-play game {i+1}/{n_games} | "
                  f"P1 wins: {wins[1]}, P2 wins: {wins[-1]}, Draws: {wins[0]} | "
                  f"Total samples: {len(all_data)}")

    return all_data