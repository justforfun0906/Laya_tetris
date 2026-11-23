"""Watch a policy play Tetris live in the terminal, forever.

    python watch.py                                   # student (runs/laya-tetris)
    python watch.py --policy teacher                  # the heuristic, for comparison
    python watch.py --delay 0.25                      # slow it down to watch each move

No piece cap: each game runs until it tops out, then a fresh game starts automatically with a
new seed. Cumulative stats (games, total lines, best game) persist across restarts. Ctrl-C quits.
"""
import argparse
import random
import sys
import time

import numpy as np

import teacher
from laya_format import K_DEFAULT, build_question, render_state, shortlist
from tetris import COLS, ROWS, Game, PIECES

# 256-colour code per tetromino (classic-ish), used to paint the settled stack.
PIECE_COLOR = {"I": 51, "O": 226, "T": 201, "S": 46, "Z": 196, "J": 33, "L": 208}
HIDE, SHOW, HOME = "\x1b[?25l", "\x1b[?25h", "\x1b[H"
RESET = "\x1b[0m"


def cell(color_idx):
    if color_idx == 0:
        return "\x1b[38;5;236m· \x1b[0m"          # empty: dim dot
    return "\x1b[38;5;%dm██%s" % (color_idx, RESET)


def frame(game, colors, stats, latency_ms):
    """Build one full screen as a string (fixed width so HOME-redraw doesn't leave debris)."""
    lines = []
    lines.append("\x1b[1m LAYA plays TETRIS \x1b[0m  (%s policy)" % stats["policy"])
    lines.append("")
    top = "┌" + "──" * COLS + "┐"
    bot = "└" + "──" * COLS + "┘"
    lines.append("  " + top)
    for r in range(ROWS):
        row = "".join(cell(int(colors[r, c])) for c in range(COLS))
        lines.append("  │" + row + "│")
    lines.append("  " + bot)
    lines.append("")
    lines.append("  current \x1b[38;5;%dm%s\x1b[0m    next \x1b[38;5;%dm%s\x1b[0m    %d ms/move"
                 % (PIECE_COLOR[game.piece], game.piece,
                    PIECE_COLOR[game.next_piece], game.next_piece, latency_ms))
    lines.append("")
    lpp = game.lines / game.pieces if game.pieces else 0.0
    lines.append("  game %-4d  pieces %-5d  lines %-5d  (%.2f lines/piece)"
                 % (stats["game"], game.pieces, game.lines, lpp))
    lines.append("  totals: %d games, %d lines cleared, best game %d lines"
                 % (stats["game"] - 1 + (0 if game.over else 1),
                    stats["total_lines"] + game.lines, max(stats["best"], game.lines)))
    lines.append("\x1b[38;5;240m  Ctrl-C to quit\x1b[0m")
    # pad every line to a fixed visible width is unnecessary because we clear-to-EOL per line:
    return HOME + "\n".join(l + "\x1b[K" for l in lines) + "\x1b[J"


def apply_color(colors, game, p):
    """Mirror the engine's drop+line-clear onto the parallel colour grid."""
    idx = PIECE_COLOR[game.piece]
    pre = game.board.copy()
    for r, c in p.cells:
        pre[r, c] = True
        colors[r, c] = idx
    full = pre.all(axis=1)
    if full.any():
        kept = colors[~full]
        colors[:] = 0
        colors[ROWS - kept.shape[0]:] = kept


def choose(policy, game, agent, k, rng):
    legal = game.legal()
    if policy == "teacher":
        return legal[teacher.best(legal)], 0.0
    if policy == "random":
        return rng.choice(legal), 0.0
    cands = shortlist(legal, k)
    questions, by_label = build_question(cands)
    t0 = time.perf_counter()
    ans = agent.predict(render_state(game), questions)["answers"]["move"]["choice"]
    return by_label[ans], (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["student", "teacher", "random"], default="student")
    ap.add_argument("--model", default="runs/laya-tetris")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--seed", type=int, default=None, help="starting seed; each game increments it")
    ap.add_argument("--delay", type=float, default=0.0, help="extra seconds to pause per move")
    args = ap.parse_args()

    agent = None
    if args.policy == "student":
        import laya
        from laya_format import HEAD_MAX_LEN, MAX_LEN
        sys.stdout.write("loading model %s ...\n" % args.model)
        sys.stdout.flush()
        agent = laya.load(args.model)
        agent.cfg["max_len"], agent.cfg["head_max_len"] = MAX_LEN, HEAD_MAX_LEN

    rng = random.Random(args.seed)
    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    stats = {"policy": args.policy, "game": 0, "total_lines": 0, "best": 0}

    sys.stdout.write("\x1b[2J" + HIDE)
    try:
        while True:
            stats["game"] += 1
            game = Game(seed=base_seed + stats["game"])
            colors = np.zeros((ROWS, COLS), dtype=int)
            latency = 0.0
            while not game.over:
                p, latency = choose(args.policy, game, agent, args.k, rng)
                apply_color(colors, game, p)          # paint the locked piece (pre-clear)
                game.step(p)                          # advance the engine
                sys.stdout.write(frame(game, colors, stats, int(latency)))
                sys.stdout.flush()
                if args.delay:
                    time.sleep(args.delay)
            # topped out: bank the result, flash, and start over
            stats["total_lines"] += game.lines
            stats["best"] = max(stats["best"], game.lines)
            sys.stdout.write("\n  \x1b[1;38;5;196mTOP OUT\x1b[0m after %d lines — restarting...\x1b[K\n"
                             % game.lines)
            sys.stdout.flush()
            time.sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW + "\n")
        sys.stdout.flush()
        print("stopped: %d games, %d total lines, best game %d lines"
              % (stats["game"], stats["total_lines"], stats["best"]))


if __name__ == "__main__":
    main()
