"""Play/evaluate a policy.

    python play.py --policy teacher --games 5
    python play.py --policy student --model runs/laya-tetris --games 3 --show
    python play.py --policy laya-base          # the untuned checkpoint, for comparison
"""
import argparse
import random
import time

import numpy as np

import teacher
from laya_format import K_DEFAULT, build_question, render_state, shortlist
from tetris import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["teacher", "student", "laya-base", "random"], default="teacher")
    ap.add_argument("--model", default="runs/laya-tetris")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-pieces", type=int, default=500)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--seed", type=int, default=12345, help="eval seeds differ from gen_data's")
    ap.add_argument("--show", action="store_true", help="print the board after every move")
    args = ap.parse_args()

    agent = None
    if args.policy in ("student", "laya-base"):
        import laya
        from laya_format import HEAD_MAX_LEN, MAX_LEN
        agent = laya.load(args.model if args.policy == "student" else "convaiinnovations/laya")
        agent.cfg["max_len"], agent.cfg["head_max_len"] = MAX_LEN, HEAD_MAX_LEN

    rng = random.Random(args.seed)
    results, agree, decisions, t_model = [], 0, 0, 0.0
    for g in range(args.games):
        game = Game(seed=args.seed + g)
        while not game.over and game.pieces < args.max_pieces:
            legal = game.legal()
            if args.policy == "teacher":
                move = legal[teacher.best(legal)]
            elif args.policy == "random":
                move = rng.choice(legal)
            else:
                cands = shortlist(legal, args.k)
                questions, by_label = build_question(cands)
                t0 = time.perf_counter()
                ans = agent.predict(render_state(game), questions)["answers"]["move"]["choice"]
                t_model += time.perf_counter() - t0
                move = by_label[ans]
                agree += cands.index(move) == teacher.best(cands)
                decisions += 1
            game.step(move)
            if args.show:
                print("\x1b[2J\x1b[H" + game.render())
                print("piece %d  lines %d  next %s" % (game.pieces, game.lines, game.piece))
        results.append((game.lines, game.pieces, game.over))
        print("game %d: %d lines, %d pieces, %s" % (g, game.lines, game.pieces,
                                                  "topped out" if game.over else "piece cap reached"))

    lines = np.array([r[0] for r in results])
    print("\n%s: mean lines %.1f (min %d, max %d) over %d games, cap %d pieces"
          % (args.policy, lines.mean(), lines.min(), lines.max(), len(results), args.max_pieces))
    if decisions:
        print("agreement with teacher (within shortlist): %.1f%%, %.0f ms/move"
              % (100.0 * agree / decisions, 1000 * t_model / decisions))


if __name__ == "__main__":
    main()
