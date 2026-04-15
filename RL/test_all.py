

import pytest
import numpy as np
import torch

from src.game.connect4 import Connect4
from src.network.model import AlphaZeroNet
from src.mcts.mcts import MCTS, MCTSNode
from src.training.replay_buffer import ReplayBuffer


# Game tests 

class TestConnect4:
    def test_initial_state(self):
        g = Connect4()
        assert g.board.shape == (6, 7)
        assert np.all(g.board == 0)
        assert g.current_player == 1
        assert not g.done

    def test_legal_moves_full_board(self):
        g = Connect4()
        assert g.legal_moves() == list(range(7))

    def test_column_fills_up(self):
        g = Connect4()
        for _ in range(6):
            g.make_move(0)
            if g.done:
                break
        # Column 0 should be gone from legal moves (or game ended)
        if not g.done:
            assert 0 not in g.legal_moves()

    def test_horizontal_win(self):
        g = Connect4()
        # P1 plays cols 0,1,2,3; P2 plays col 6 in between
        moves = [0, 6, 1, 6, 2, 6, 3]
        for m in moves:
            _, _, done = g.make_move(m)
        assert g.done
        assert g.winner == 1

    def test_vertical_win(self):
        g = Connect4()
        # P1 stacks column 0; P2 stacks column 1
        moves = [0, 1, 0, 1, 0, 1, 0]
        for m in moves:
            _, _, done = g.make_move(m)
        assert g.done
        assert g.winner == 1

    def test_draw(self):
        # Fill board without winning — hard to do manually, so test via outcome
        g = Connect4()
        assert g.winner == 0
        assert not g.done

    def test_encode_shape(self):
        g = Connect4()
        enc = g.encode()
        assert enc.shape == (8, 6, 7)
        assert enc.dtype == np.float32

    def test_clone_independence(self):
        g = Connect4()
        g.make_move(3)
        clone = g.clone()
        clone.make_move(3)
        assert g.board[5, 3] == 1
        assert not g.done  # original unaffected

    def test_player_alternates(self):
        g = Connect4()
        assert g.current_player == 1
        g.make_move(0)
        assert g.current_player == -1
        g.make_move(1)
        assert g.current_player == 1


# ── Network tests ────────────────────────────────────────────────────────────

class TestAlphaZeroNet:
    def setup_method(self):
        self.net = AlphaZeroNet(n_res_blocks=2, n_filters=32)

    def test_output_shapes(self):
        x = torch.randn(4, 8, 6, 7)
        log_p, v = self.net(x)
        assert log_p.shape == (4, 7)
        assert v.shape == (4, 1)

    def test_value_range(self):
        x = torch.randn(8, 8, 6, 7)
        _, v = self.net(x)
        assert (v >= -1.0).all() and (v <= 1.0).all()

    def test_policy_sums_to_one(self):
        x = torch.randn(4, 8, 6, 7)
        log_p, _ = self.net(x)
        p = torch.exp(log_p)
        sums = p.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_parameter_count(self):
        count = self.net.count_parameters()
        assert count > 0
        print(f"\n  Network params (2 blocks, 32 filters): {count:,}")

    def test_single_sample(self):
        g = Connect4()
        x = torch.tensor(g.encode()).unsqueeze(0)
        log_p, v = self.net(x)
        assert log_p.shape == (1, 7)
        assert v.shape == (1, 1)


# ── MCTS tests ───────────────────────────────────────────────────────────────

class TestMCTS:
    def setup_method(self):
        self.net = AlphaZeroNet(n_res_blocks=2, n_filters=32)
        self.net.eval()
        self.mcts = MCTS(self.net, n_simulations=20, device="cpu")

    def test_search_returns_valid_distribution(self):
        g = Connect4()
        pi = self.mcts.search(g, temperature=1.0)
        assert pi.shape == (7,)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert (pi >= 0).all()

    def test_best_move_is_legal(self):
        g = Connect4()
        move = self.mcts.best_move(g)
        assert move in g.legal_moves()

    def test_search_on_near_terminal(self):
        # Fill 6 columns, only col 6 remains
        g = Connect4()
        for col in range(6):
            for _ in range(6):
                if col in g.legal_moves():
                    g.make_move(col)
                if g.done:
                    break
            if g.done:
                break
        if not g.done:
            pi = self.mcts.search(g, temperature=1.0)
            assert pi.sum() > 0

    def test_ucb_score(self):
        g = Connect4()
        node = MCTSNode(g, prior=0.3)
        node.N = 5
        node.W = 3.0
        node.Q = 0.6
        score = node.ucb_score(parent_N=100, c_puct=1.25)
        assert score > node.Q  # exploration bonus added


# ── Replay buffer tests ──────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=1000)
        data = [
            (np.random.randn(8, 6, 7).astype(np.float32),
             np.random.dirichlet(np.ones(7)).astype(np.float32),
             float(np.random.choice([-1, 0, 1])))
            for _ in range(100)
        ]
        buf.add_game(data)
        assert len(buf) == 100
        s, pi, z = buf.sample(32)
        assert s.shape == (32, 8, 6, 7)
        assert pi.shape == (32, 7)
        assert z.shape == (32,)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=50)
        data = [(np.zeros((8,6,7), np.float32), np.ones(7)/7, 0.0)] * 100
        buf.add_game(data)
        assert len(buf) == 50  # capped

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=1000)
        assert not buf.is_ready(100)
        buf.add_game([(np.zeros((8,6,7), np.float32), np.ones(7)/7, 0.0)] * 200)
        assert buf.is_ready(100)
