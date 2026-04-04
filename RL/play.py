

import argparse
import torch
import numpy as np

from src.game.connect4 import Connect4
from src.network.model import AlphaZeroNet
from src.mcts.mcts import MCTS


def load_agent(checkpoint_path: str, simulations: int, device: str) -> MCTS:
    payload = torch.load(checkpoint_path, map_location=device)
    net = AlphaZeroNet()
    net.load_state_dict(payload["model_state"])
    net.eval()
    return MCTS(net, n_simulations=simulations, device=device, dirichlet_epsilon=0.0)


def human_move(game: Connect4) -> int:
    legal = game.legal_moves()
    while True:
        try:
            col = int(input(f"Your move (columns {legal}): "))
            if col in legal:
                return col
            print(f"  Column {col} is not legal. Choose from {legal}.")
        except (ValueError, EOFError):
            print("  Please enter a column number.")


def play_human_vs_agent(agent: MCTS, human_player: int = 1):
    game = Connect4()
    print("\nConnect4 — You are", "X (P1)" if human_player == 1 else "O (P2)")
    print(game.render())

    while not game.done:
        print()
        if game.current_player == human_player:
            col = human_move(game)
        else:
            print("Agent thinking...")
            col = agent.best_move(game, temperature=0.0)
            print(f"Agent plays column {col}")

        game.make_move(col)
        print(game.render())

    print()
    if game.winner == 0:
        print("It's a draw!")
    elif game.winner == human_player:
        print("You win!")
    else:
        print("Agent wins!")


def play_agent_vs_agent(agent1: MCTS, agent2: MCTS):
    game = Connect4()
    print("\nAgent vs Agent\n")
    print(game.render())

    move_num = 0
    while not game.done:
        print()
        agent = agent1 if game.current_player == 1 else agent2
        col = agent.best_move(game, temperature=0.0)
        player_label = "Agent1 (X)" if game.current_player == 1 else "Agent2 (O)"
        print(f"Move {move_num + 1}: {player_label} → column {col}")
        game.make_move(col)
        print(game.render())
        move_num += 1
        input("Press Enter to continue...")

    print()
    if game.winner == 0:
        print("Draw!")
    else:
        winner = "Agent1 (X)" if game.winner == 1 else "Agent2 (O)"
        print(f"{winner} wins!")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    p.add_argument("--checkpoint2", type=str, default=None, help="Second checkpoint for agent vs agent")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--human_player", type=int, default=1, choices=[1, 2])
    p.add_argument("--agent_vs_agent", action="store_true")
    args = p.parse_args()

    agent = load_agent(args.checkpoint, args.simulations, args.device)

    if args.agent_vs_agent:
        if args.checkpoint2:
            agent2 = load_agent(args.checkpoint2, args.simulations, args.device)
        else:
            agent2 = agent
        play_agent_vs_agent(agent, agent2)
    else:
        human_player = args.human_player
        play_human_vs_agent(agent, human_player=human_player)


if __name__ == "__main__":
    main()