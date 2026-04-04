

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.game.connect4 import Connect4
from src.network.model import AlphaZeroNet
from src.mcts.mcts import MCTS
from src.training.trainer import Trainer
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import run_self_play_games
from src.evaluation.arena import evaluate_vs_random, evaluate_vs_greedy


def parse_args():
    p = argparse.ArgumentParser(description="Train AlphaZero on Connect4")
    p.add_argument("--iterations", type=int, default=30, help="Number of training iterations")
    p.add_argument("--self_play_games", type=int, default=50, help="Self-play games per iteration")
    p.add_argument("--simulations", type=int, default=200, help="MCTS simulations per move")
    p.add_argument("--n_res_blocks", type=int, default=6, help="Residual blocks in network")
    p.add_argument("--n_filters", type=int, default=128, help="Conv filters")
    p.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--buffer_capacity", type=int, default=100_000, help="Replay buffer size")
    p.add_argument("--min_buffer", type=int, default=1000, help="Min samples before training starts")
    p.add_argument("--eval_every", type=int, default=5, help="Evaluate every N iterations")
    p.add_argument("--eval_games", type=int, default=50, help="Games per evaluation match")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    p.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def plot_training_curves(history, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [h["iteration"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("AlphaZero Connect4 — Training Curves", fontsize=14)

    # Loss curves
    ax = axes[0, 0]
    ax.plot(iterations, [h["total_loss"] for h in history], label="Total loss", color="#534AB7")
    ax.plot(iterations, [h["policy_loss"] for h in history], label="Policy loss", color="#1D9E75", linestyle="--")
    ax.plot(iterations, [h["value_loss"] for h in history], label="Value loss", color="#D85A30", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Win rate vs random
    eval_iters = [h["iteration"] for h in history if "wr_random" in h]
    if eval_iters:
        ax = axes[0, 1]
        ax.plot(eval_iters, [h["wr_random"] for h in history if "wr_random" in h],
                color="#534AB7", marker="o", label="vs Random")
        ax.plot(eval_iters, [h.get("wr_greedy", 0) for h in history if "wr_random" in h],
                color="#1D9E75", marker="s", label="vs Greedy")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win rate")
        ax.set_title("Win rate vs baselines")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Buffer size
    ax = axes[1, 0]
    ax.plot(iterations, [h["buffer_size"] for h in history], color="#D85A30", marker=".")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Samples")
    ax.set_title("Replay buffer size")
    ax.grid(True, alpha=0.3)

    # Elo
    if eval_iters:
        ax = axes[1, 1]
        ax.plot(eval_iters, [h.get("elo_random", 0) for h in history if "wr_random" in h],
                color="#534AB7", marker="o", label="vs Random")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Estimated Elo")
        ax.set_title("Elo estimate")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {path}")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"\n{'='*60}")
    print(f"  AlphaZero — Connect4")
    print(f"  Device: {device}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Self-play games/iter: {args.self_play_games}")
    print(f"  MCTS simulations/move: {args.simulations}")
    print(f"{'='*60}\n")

    # Build components
    net = AlphaZeroNet(
        in_channels=8,
        n_res_blocks=args.n_res_blocks,
        n_filters=args.n_filters,
        board_h=Connect4.ROWS,
        board_w=Connect4.COLS,
        n_actions=Connect4.COLS,
    )
    print(f"Network parameters: {net.count_parameters():,}")

    trainer = Trainer(
        net,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    buffer = ReplayBuffer(capacity=args.buffer_capacity)
    mcts = MCTS(net, n_simulations=args.simulations, device=device)

    start_iter = 0
    history = []

    if args.resume:
        print(f"Resuming from: {args.resume}")
        start_iter = trainer.load_checkpoint(args.resume)
        history = trainer.history

    # Main training loop
    for iteration in range(start_iter + 1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n[Iteration {iteration}/{args.iterations}]")

        # --- Self-play ---
        print(f"  Generating {args.self_play_games} self-play games...")
        net.eval()
        game_data = run_self_play_games(mcts, n_games=args.self_play_games, verbose=True)
        buffer.add_game(game_data)
        print(f"  Buffer size: {len(buffer):,} samples")

        # --- Training ---
        if buffer.is_ready(args.min_buffer):
            print(f"  Training network...")
            net.train()
            losses = trainer.train_on_buffer(buffer)
            print(f"  Loss → total: {losses['total']:.4f} | "
                  f"policy: {losses['policy']:.4f} | value: {losses['value']:.4f}")
        else:
            print(f"  Skipping training (buffer has {len(buffer)} < {args.min_buffer} samples)")
            losses = {"total": 0.0, "policy": 0.0, "value": 0.0}

        # --- Record metrics ---
        record = {
            "iteration": iteration,
            "total_loss": losses["total"],
            "policy_loss": losses["policy"],
            "value_loss": losses["value"],
            "buffer_size": len(buffer),
            "elapsed": time.time() - iter_start,
        }

        # --- Evaluation ---
        if iteration % args.eval_every == 0:
            print(f"  Evaluating vs baselines ({args.eval_games} games each)...")
            net.eval()
            r_result = evaluate_vs_random(mcts, n_games=args.eval_games)
            g_result = evaluate_vs_greedy(mcts, n_games=args.eval_games)
            print(f"  vs Random → W:{r_result['wins']} D:{r_result['draws']} L:{r_result['losses']} "
                  f"| WR: {r_result['win_rate']:.2%} | Elo: {r_result['elo']:.0f}")
            print(f"  vs Greedy → W:{g_result['wins']} D:{g_result['draws']} L:{g_result['losses']} "
                  f"| WR: {g_result['win_rate']:.2%} | Elo: {g_result['elo']:.0f}")
            record.update({
                "wr_random": r_result["win_rate"],
                "wr_greedy": g_result["win_rate"],
                "elo_random": r_result["elo"],
                "elo_greedy": g_result["elo"],
            })

        history.append(record)
        trainer.history = history

        # --- Checkpoint ---
        ckpt_path = trainer.save_checkpoint(iteration, extra={"args": vars(args)})
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Iteration time: {record['elapsed']:.1f}s")

        # --- Plot ---
        if iteration % args.eval_every == 0:
            plot_training_curves(history)

    # Save final history as JSON
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to results/training_history.json")
    plot_training_curves(history)


if __name__ == "__main__":
    main()