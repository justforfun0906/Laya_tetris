"""Minimal placement-level Tetris engine (10x20, 7-bag, hard drops, no tucks/spins).

An action is a placement: (rotation index, left column). The engine enumerates every legal
placement of the current piece, and applying one hard-drops it, clears lines and advances the bag.
Row 0 is the top of the board.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

ROWS, COLS = 20, 10

# Cells as (row, col) offsets for the spawn orientation; rotations are derived below.
_BASE = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}
PIECES = list(_BASE)


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotations(cells):
    out, cur = [], _normalize(cells)
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        cur = _normalize([(c, -r) for r, c in cur])  # rotate 90 degrees clockwise
    return out


ROTATIONS = {p: _rotations(c) for p, c in _BASE.items()}


@dataclass
class Placement:
    rot: int
    col: int
    cells: Tuple[Tuple[int, int], ...]  # absolute (row, col) cells where the piece lands
    board: np.ndarray                   # board after the drop and line clears
    lines: int                          # lines cleared by this placement
    eroded: int                         # piece cells removed by those line clears


def column_tops(board: np.ndarray) -> np.ndarray:
    """Index of the highest filled row per column (ROWS for an empty column)."""
    filled = board.any(axis=0)
    return np.where(filled, board.argmax(axis=0), ROWS)


def placements(board: np.ndarray, piece: str) -> List[Placement]:
    """Every distinct hard-drop placement of `piece` that fits entirely inside the board."""
    tops = column_tops(board)
    out = []
    for rot, shape in enumerate(ROTATIONS[piece]):
        width = max(c for _, c in shape) + 1
        bottom = {}  # piece column -> lowest cell row offset
        for r, c in shape:
            bottom[c] = max(bottom.get(c, -1), r)
        for col in range(COLS - width + 1):
            y = min(tops[col + dc] - br - 1 for dc, br in bottom.items())
            cells = tuple((y + r, col + c) for r, c in shape)
            if any(r < 0 for r, _ in cells):
                continue  # would stick out of the top: not legal
            nb = board.copy()
            for r, c in cells:
                nb[r, c] = True
            full = nb.all(axis=1)
            lines = int(full.sum())
            eroded = 0
            if lines:
                eroded = sum(1 for r, _ in cells if full[r])
                nb = np.vstack([np.zeros((lines, COLS), dtype=bool), nb[~full]])
            out.append(Placement(rot, col, cells, nb, lines, eroded))
    return out


class Bag:
    def __init__(self, rng: random.Random):
        self.rng, self.queue = rng, []

    def next(self) -> str:
        if not self.queue:
            self.queue = PIECES[:]
            self.rng.shuffle(self.queue)
        return self.queue.pop()


@dataclass
class Game:
    seed: Optional[int] = None
    board: np.ndarray = field(default_factory=lambda: np.zeros((ROWS, COLS), dtype=bool))
    lines: int = 0
    pieces: int = 0
    over: bool = False

    def __post_init__(self):
        self.bag = Bag(random.Random(self.seed))
        self.piece = self.bag.next()
        self.next_piece = self.bag.next()
        self._check_over()

    def legal(self) -> List[Placement]:
        return placements(self.board, self.piece)

    def step(self, p: Placement) -> int:
        self.board = p.board
        self.lines += p.lines
        self.pieces += 1
        self.piece, self.next_piece = self.next_piece, self.bag.next()
        self._check_over()
        return p.lines

    def _check_over(self):
        self.over = not self.legal()

    def render(self) -> str:
        return "\n".join("|" + "".join("#" if x else "." for x in row) + "|" for row in self.board)
