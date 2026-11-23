"""Turn a Tetris position into a Laya `choice` question.

Laya scores a fixed list of options against a text state in one forward pass, so each move is
posed as: state = board + current/next piece, options = up to K candidate placements, each with
a few plain-language consequences (lines cleared, holes, heights...). A cheap, teacher-independent
prefilter picks the K candidates, so the student never needs the teacher at inference time.
"""
import string
from typing import Dict, List, Tuple

import numpy as np

from teacher import holes
from tetris import COLS, ROWS, Game, Placement, column_tops

K_DEFAULT = 10
LABELS = string.ascii_uppercase
MAX_LEN = 640        # full token budget per question (ModernBERT handles far more)
HEAD_MAX_LEN = 288   # budget for instruction + options; K=10 options take ~230 tokens

INSTRUCTIONS = ("You are playing Tetris. Choose the placement for the current piece that keeps "
                "the board low, flat and free of holes, and clears lines when possible.")


def summary(p: Placement) -> Dict[str, int]:
    tops = column_tops(p.board)
    heights = ROWS - tops
    return {
        "lines": p.lines,
        "holes": holes(p.board),
        "land": ROWS - min(r for r, _ in p.cells),
        "top": int(heights.max()),
        "bump": int(np.abs(np.diff(heights)).sum()),
    }


def shortlist(cands: List[Placement], k: int = K_DEFAULT) -> List[Placement]:
    """Keep the k most promising placements by a crude rule, then order them by (rot, col).

    The final ordering deliberately hides the prefilter rank so the model has to judge options.
    """
    def key(p):
        s = summary(p)
        return (s["holes"] - 2 * s["lines"], s["top"], s["bump"])
    kept = sorted(cands, key=key)[:k]
    return sorted(kept, key=lambda p: (p.rot, p.col))


def render_state(game: Game) -> str:
    tops = column_tops(game.board)
    first = max(0, min(int(tops.min()), ROWS) - 1)  # one empty row above the stack
    rows = ["".join("#" if x else "." for x in game.board[r]) for r in range(first, ROWS)]
    return ("Tetris board %dx%d, current piece %s, next piece %s. Stack height %d. "
            "Board rows from top of stack to floor (# filled, . empty), columns 0-9:\n%s"
            % (COLS, ROWS, game.piece, game.next_piece, ROWS - int(tops.min()), "\n".join(rows)))


def render_option(p: Placement) -> str:
    s = summary(p)
    return ("rotation %d column %d; clears %d, holes %d, lands at %d, max height %d, bumpiness %d"
            % (p.rot, p.col, s["lines"], s["holes"], s["land"], s["top"], s["bump"]))


def build_question(cands: List[Placement]) -> Tuple[Dict, Dict[str, Placement]]:
    criteria, by_label = {}, {}
    for label, p in zip(LABELS, cands):
        criteria[label] = render_option(p)
        by_label[label] = p
    return {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}, by_label
