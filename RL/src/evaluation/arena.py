"""
Evaluation utilities 

Arena:    Play N games between two agents and report win/draw/loss.
Metrics:  Elo estimation, policy entropy, value calibration.
"""

import numpy as np
from typing import Callable, Dict, Tuple, List

from src.game.connect4 import Connect4
from src.mcts.mcts import MCTS


def play_match(
    agent1: MCTS,
    agent2: MCTS,
    n_games: int = 100,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Play n_games between agent1 (plays as P1 in half, P2 in other half).
    Returns wins/draws/losses for agent1.
    """
    results = {"wins": 0, "draws": 0, "losses": 0}

    for game_idx in range(n_games):
        game = Connect4()
        # Alternate who plays first
        agents = [agent1, agent2] if game_idx % 2 == 0 else [agent2, agent1]
        flip = game_idx % 2 != 0

        while not game.done:
            current_agent = agents[game.current_player == 1]
            move = current_agent.best_move(game, temperature=0.0)
            game.make_move(move)

        # Determine result for agent1
        if game.winner == 0:
            results["draws"] += 1
        else:
            # agent1 played as P1 in even games, P2 in odd
            agent1_player = 1 if not flip else -1
            if game.winner == agent1_player:
                results["wins"] += 1
            else:
                results["losses"] += 1

        if verbose and (game_idx + 1) % 20 == 0:
            print(f"  Match {game_idx+1}/{n_games}: {results}")

    return results


def win_rate(results: Dict[str, int]) -> float:
    total = sum(results.values())
    if total == 0:
        return 0.0
    return (results["wins"] + 0.5 * results["draws"]) / total


def elo_from_win_rate(wr: float, base_elo: int = 0) -> float:
    """Estimate Elo delta from win rate using standard formula."""
    if wr <= 0:
        return -800.0
    if wr >= 1:
        return 800.0
    return base_elo + 400.0 * np.log10(wr / (1.0 - wr))


class RandomAgent:
    """Baseline: choose uniformly from legal moves."""

    def best_move(self, state: Connect4, temperature: float = 0.0) -> int:
        legal = state.legal_moves()
        return np.random.choice(legal)


class GreedyAgent:
    """Baseline: choose the move that wins immediately, else random."""

    def best_move(self, state: Connect4, temperature: float = 0.0) -> int:
        for move in state.legal_moves():
            clone = state.clone()
            _, reward, done = clone.make_move(move)
            if done and reward > 0:
                return move
        return np.random.choice(state.legal_moves())


def evaluate_vs_random(mcts: MCTS, n_games: int = 100) -> Dict:
    random_agent = RandomAgent()
    results = {"wins": 0, "draws": 0, "losses": 0}

    for game_idx in range(n_games):
        game = Connect4()
        mcts_player = 1 if game_idx % 2 == 0 else -1

        while not game.done:
            if game.current_player == mcts_player:
                move = mcts.best_move(game, temperature=0.0)
            else:
                move = random_agent.best_move(game)
            game.make_move(move)

        if game.winner == 0:
            results["draws"] += 1
        elif game.winner == mcts_player:
            results["wins"] += 1
        else:
            results["losses"] += 1

    wr = win_rate(results)
    return {**results, "win_rate": wr, "elo": elo_from_win_rate(wr)}


def evaluate_vs_greedy(mcts: MCTS, n_games: int = 100) -> Dict:
    greedy_agent = GreedyAgent()
    results = {"wins": 0, "draws": 0, "losses": 0}

    for game_idx in range(n_games):
        game = Connect4()
        mcts_player = 1 if game_idx % 2 == 0 else -1

        while not game.done:
            if game.current_player == mcts_player:
                move = mcts.best_move(game, temperature=0.0)
            else:
                move = greedy_agent.best_move(game)
            game.make_move(move)

        if game.winner == 0:
            results["draws"] += 1
        elif game.winner == mcts_player:
            results["wins"] += 1
        else:
            results["losses"] += 1

    wr = win_rate(results)
    return {**results, "win_rate": wr, "elo": elo_from_win_rate(wr)}


def policy_entropy(mcts: MCTS, states: List[Connect4]) -> float:
    """Average entropy of MCTS policy distributions — measures exploration."""
    entropies = []
    for state in states:
        pi = mcts.search(state, temperature=1.0)
        pi = pi[pi > 0]
        entropies.append(-np.sum(pi * np.log(pi + 1e-8)))
    return float(np.mean(entropies))