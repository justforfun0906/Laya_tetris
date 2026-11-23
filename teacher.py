"""Teacher policy: Pierre Dellacherie's heuristic with the El-Tetris weights.

Scores every placement with a fixed linear evaluator over six board features. It is a
one-piece greedy search, yet routinely clears hundreds of thousands of lines, which makes it a
strong, cheap and fully deterministic teacher. Weights from Yiyuan Lee's El-Tetris write-up
(https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/).
"""
from typing import Dict, List

import numpy as np

from tetris import COLS, ROWS, Placement

WEIGHTS = {
    "landing_height": -4.500158825082766,
    "eroded_cells": 3.4181268101392694,
    "row_transitions": -3.2178882868487753,
    "col_transitions": -9.348695305445199,
    "holes": -7.899265427351652,
    "wells": -3.3855972247263626,
}


def row_transitions(b: np.ndarray) -> int:
    padded = np.ones((ROWS, COLS + 2), dtype=bool)  # walls count as filled
    padded[:, 1:-1] = b
    return int((padded[:, 1:] != padded[:, :-1]).sum())


def col_transitions(b: np.ndarray) -> int:
    padded = np.ones((ROWS + 1, COLS), dtype=bool)  # floor counts as filled
    padded[:-1] = b
    return int((padded[1:] != padded[:-1]).sum())


def holes(b: np.ndarray) -> int:
    covered = np.maximum.accumulate(b, axis=0)  # True at and below the first filled cell
    return int((covered & ~b).sum())


def wells(b: np.ndarray) -> int:
    padded = np.ones((ROWS, COLS + 2), dtype=bool)
    padded[:, 1:-1] = b
    well = ~b & padded[:, :-2] & padded[:, 2:]
    total = 0
    for c in range(COLS):
        depth = 0
        for r in range(ROWS):
            if well[r, c]:
                depth += 1
                total += depth  # a well of depth d contributes 1 + 2 + ... + d
            elif b[r, c]:
                break
            else:
                depth = 0
    return total


def features(p: Placement) -> Dict[str, float]:
    heights = [ROWS - r for r, _ in p.cells]
    return {
        "landing_height": (max(heights) + min(heights)) / 2.0,
        "eroded_cells": float(p.lines * p.eroded),
        "row_transitions": float(row_transitions(p.board)),
        "col_transitions": float(col_transitions(p.board)),
        "holes": float(holes(p.board)),
        "wells": float(wells(p.board)),
    }


def score(p: Placement) -> float:
    f = features(p)
    return sum(WEIGHTS[k] * f[k] for k in WEIGHTS)


def scores(cands: List[Placement]) -> np.ndarray:
    return np.array([score(p) for p in cands], dtype=np.float64)


def best(cands: List[Placement]) -> int:
    return int(scores(cands).argmax())
