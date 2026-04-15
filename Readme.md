# AlphaMinus

An implementation of the **AlphaZero algorithm** applied to **Connect4**, built from scratch in PyTorch. This project implements the complete self-play reinforcement learning pipeline: Monte Carlo Tree Search guided by a residual neural network, trained entirely through self-play with no human data.  

---

## What This Project Implements

AlphaZero learns to play Connect4 by playing against itself, using the results to improve a neural network, and using that improved network to guide better search a closed improvement loop with no human knowledge beyond the rules.

| Component | Description |
|---|---|
| **MCTS** | Monte Carlo Tree Search with PUCT selection + Dirichlet noise |
| **Neural Network** | ResNet with shared trunk, policy head (move probs), value head (win pred) |
| **Self-Play** | Temperature-controlled exploration, full game data generation |
| **Replay Buffer** | Circular buffer of `(state, π, z)` tuples |
| **Training** | Combined policy + value loss, gradient clipping, checkpointing |
| **Evaluation** | Arena matches vs Random / Greedy baselines, Elo estimation |

---

## Architecture

```
Board State (8 × 6 × 7)
       │
   Conv Block
  128 filters, 3×3
       │
 ┌─────┴─────┐
 │  Residual  │  × 6
 │   Tower    │
 └─────┬─────┘
       │
  ┌────┴────┐
  │         │
Policy    Value
 Head      Head
  │         │
  ▼         ▼
log p(a)   v ∈ [-1,1]
(7 moves)
```

**Input encoding** — 8 planes of shape `(6, 7)`:
- Plane 0: current player's pieces
- Plane 1: opponent's pieces
- Planes 2–6: reserved for move history
- Plane 7: whose turn (constant 0 or 1)

**MCTS selection** uses the PUCT formula:
```
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
```

**Loss function**:
```
L = cross_entropy(π, p) + MSE(z, v)
```
where `π` is the MCTS visit distribution and `z` is the game outcome.

---

## Results

After 30 training iterations (~1,500 self-play games):

| Metric | Value |
|---|---|
| Win rate vs Random agent | ~95% |
| Win rate vs Greedy agent | ~80% |
| Estimated Elo (vs Random=0) | +600 |

Training curves are saved to `results/plots/training_curves.png` after each evaluation round.

---


## Project Structure

```
RL/
│
├── src/
│   ├── game/
│   │   └── connect4.py         # Environment: board state, legal moves, win detection
│   │
│   ├── mcts/
│   │   └── mcts.py             # MCTS: selection (PUCT), expansion, backprop
│   │
│   ├── network/
│   │   └── model.py            # ResNet: ConvBlock, ResBlock, PolicyHead, ValueHead
│   │
│   ├── training/
│   │   ├── self_play.py        # Self-play game generation
│   │   ├── replay_buffer.py    # Circular experience buffer
│   │   └── trainer.py          # Loss computation, optimizer, checkpointing
│   │
│   └── evaluation/
│       └── arena.py            # Match playing, win rate, Elo estimation
│
├── train.py                    # Main training script
├── play.py                     # Play against trained agent
├── evaluate.py                 # Benchmark a checkpoint
├── config.yaml                 # All hyperparameters
├── notebooks/
│   └── analysis.ipynb          # Training curves, value calibration, policy viz
│
├── tests/
│   └── test_all.py             # Unit tests: game, network, MCTS, buffer
│
├── results/
│   ├── checkpoints/            # Saved model weights (*.pt)
│   └── plots/                  # Training curve images
│
├── requirements.txt
└── setup.py
```

---

## Hyperparameters

All hyperparameters are documented in `config.yaml`. Key ones:

| Parameter | Default | Description |
|---|---|---|
| `mcts.simulations` | 200 | MCTS rollouts per move |
| `mcts.c_puct` | 1.25 | Exploration constant |
| `mcts.dirichlet_alpha` | 0.3 | Noise concentration at root |
| `mcts.dirichlet_epsilon` | 0.25 | Fraction of policy replaced by noise |
| `network.n_res_blocks` | 6 | Residual blocks |
| `network.n_filters` | 128 | Conv filters per layer |
| `training.self_play_games` | 50 | Games generated per iteration |
| `optimizer.lr` | 1e-3 | Adam learning rate |
| `optimizer.batch_size` | 512 | Training batch size |

---

## How AlphaZero Works

### Self-play loop (one iteration)

```
1. GENERATE  →  Play N games using MCTS + current network
                Each move: run 200 MCTS simulations to get π
                Store (board_state, π, game_outcome) tuples

2. TRAIN     →  Sample batch from replay buffer
                Loss = -Σ πᵢ log pᵢ  +  (z - v)²
                Update network weights

3. EVALUATE  →  Play 50 games vs Random and Greedy agents
                Track win rate and Elo

4. REPEAT
```

### MCTS (one simulation)

```
SELECT   →  Traverse tree using UCB until unexpanded leaf
EXPAND   →  Query network: get p (priors) and v (leaf value)
            Add all legal children with priors from p
BACKPROP →  Walk back up path, adding v to W, incrementing N
            Flip sign at each level (opponent's value is negative)
```

After 200–800 simulations, the visit counts form the policy target `π`.

---

## Extending This Project

- **Larger board games**: Swap `Connect4` for an Othello or Gomoku environment — the MCTS and network are game-agnostic.
- **Parallel self-play**: Use Python `multiprocessing` to generate games in parallel, significantly speeding up data collection.
- **Stronger network**: Increase `n_res_blocks` to 19 and `n_filters` to 256 for a network closer to the original AlphaZero scale.
- **History planes**: Fill in the 6 history planes in `encode()` with the last 3 board states for each player — this helps the network detect repetition.
- **Temperature scheduling**: Implement a curriculum that starts with high temperature and anneals it over training iterations.

