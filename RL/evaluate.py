

import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.game.connect4 import Connect4
from src.network.model import AlphaZeroNet
from src.mcts.mcts import MCTS
from src.evaluation.arena import (
    evaluate_vs_random,
    evaluate_vs_greedy,
    play_match,
    win_rate,
    elo_from_win_rate,
)


def load_agent(path: str, sims: int, device: str) -> MCTS:
    payload = torch.load(path, map_location=device)
    net = AlphaZeroNet()
    net.load_state_dict(payload["model_state"])
    net.eval()
    return MCTS(net, n_simulations=sims, device=device, dirichlet_epsilon=0.0)


def print_results(label: str, results: dict):
    total = results["wins"] + results["draws"] + results["losses"]
    print(f"\n  {label}")
    print(f"    W: {results['wins']}  D: {results['draws']}  L: {results['losses']}  (n={total})")
    print(f"    Win rate: {results['win_rate']:.2%}")
    print(f"    Elo (est): {results['elo']:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--checkpoint2", type=str, default=None, help="For head-to-head match")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--n_games", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, args.simulations, args.device)

    print(f"\nEvaluating ({args.n_games} games each)...")

    r = evaluate_vs_random(agent, n_games=args.n_games)
    g = evaluate_vs_greedy(agent, n_games=args.n_games)

    print_results("vs Random agent", r)
    print_results("vs Greedy agent", g)

    if args.checkpoint2:
        print(f"\nLoading opponent: {args.checkpoint2}")
        agent2 = load_agent(args.checkpoint2, args.simulations, args.device)
        h2h = play_match(agent, agent2, n_games=args.n_games, verbose=True)
        wr = win_rate(h2h)
        h2h["win_rate"] = wr
        h2h["elo"] = elo_from_win_rate(wr)
        print_results("Head-to-head (agent1 vs agent2)", h2h)

    # Summary JSON
    summary = {
        "checkpoint": args.checkpoint,
        "simulations": args.simulations,
        "vs_random": r,
        "vs_greedy": g,
    }
    with open("results/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to results/eval_summary.json")


if __name__ == "__main__":
    main()