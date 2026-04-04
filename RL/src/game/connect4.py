import numpy as np
from typing import Optional, List, Tuple
 
 
class Connect4:
    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4
 
    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.last_move = None
        self.move_count = 0
 
    def clone(self) -> "Connect4":
        g = Connect4()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.done = self.done
        g.winner = self.winner
        g.last_move = self.last_move
        g.move_count = self.move_count
        return g
 
    def legal_moves(self) -> List[int]:
        if self.done:
            return []
        return [c for c in range(self.COLS) if self.board[0, c] == 0]
 
    def make_move(self, col: int) -> Tuple[np.ndarray, float, bool]:
        assert col in self.legal_moves(), f"Illegal move: column {col}"
        row = self._drop_row(col)
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
 
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.legal_moves()) == 0:
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0
            self.current_player *= -1
 
        return self.board.copy(), reward, self.done
 
    def _drop_row(self, col: int) -> int:
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full")
 
    def _check_win(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.WIN_LENGTH:
                return True
        return False
 
    def encode(self) -> np.ndarray:
        """
        Encode board as (8, 6, 7) tensor for the neural network.
        Planes: current player pieces, opponent pieces, 6 history planes (placeholder).
        Returns float32 array shaped (8, ROWS, COLS).
        """
        planes = np.zeros((8, self.ROWS, self.COLS), dtype=np.float32)
        planes[0] = (self.board == self.current_player).astype(np.float32)
        planes[1] = (self.board == -self.current_player).astype(np.float32)
        # Plane 7: whose turn (1 = player 1, 0 = player 2)
        planes[7] = float(self.current_player == 1)
        return planes
 
    def outcome(self) -> float:
        """Return game outcome from the perspective of player who just moved."""
        if self.winner == 0:
            return 0.0
        # winner == current_player means the player who just played won
        # after make_move, current_player was already flipped for non-terminal
        # but for terminal the player who played last is self.winner
        return 1.0 if self.winner == self.current_player else -1.0
 
    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(self.ROWS):
            rows.append(" ".join(symbols[v] for v in self.board[r]))
        rows.append("-" * (2 * self.COLS - 1))
        rows.append(" ".join(str(c) for c in range(self.COLS)))
        return "\n".join(rows)
 
    def __repr__(self) -> str:
        return self.render()
 