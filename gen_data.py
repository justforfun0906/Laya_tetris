"""Generate distillation data: positions labelled with teacher scores over the shortlisted moves.

    python gen_data.py --games 60 --out data/teacher.jsonl
    # DAgger round: let the student drive, the teacher still labels every position
    python gen_data.py --games 30 --student runs/laya-tetris --student-prob 0.7 --out data/dagger1.jsonl
"""
import argparse
import json
import os
import random

from tqdm import tqdm

import teacher
from laya_format import K_DEFAULT, build_question, render_state, shortlist
from tetris import Game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--max-pieces", type=int, default=400, help="cap per game so data covers many openings")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--epsilon", type=float, default=0.15, help="chance of a random shortlisted move, for state diversity")
    ap.add_argument("--student", default=None, help="fine-tuned Laya dir; enables DAgger rollouts")
    ap.add_argument("--student-prob", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/teacher.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    agent = None
    if args.student:
        import laya
        agent = laya.load(args.student)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n, covered = 0, 0
    with open(args.out, "w") as f:
        for g in tqdm(range(args.games), desc="games"):
            game = Game(seed=args.seed * 100003 + g)
            while not game.over and game.pieces < args.max_pieces:
                legal = game.legal()
                cands = shortlist(legal, args.k)
                sc = teacher.scores(cands)
                label = int(sc.argmax())
                global_best = legal[teacher.best(legal)]
                in_list = any(p.rot == global_best.rot and p.col == global_best.col for p in cands)
                covered += in_list
                n += 1

                state = render_state(game)
                questions, by_label = build_question(cands)
                f.write(json.dumps({
                    "state": state,
                    "options": list(questions["move"]["criteria"].values()),
                    "teacher_scores": [round(float(s), 4) for s in sc],
                    "label": label,
                    "teacher_best_in_shortlist": in_list,
                }) + "\n")

                # Pick the move actually played (the label is always the teacher's choice).
                if agent is not None and rng.random() < args.student_prob:
                    ans = agent.predict(state, questions)["answers"]["move"]["choice"]
                    move = by_label[ans]
                elif rng.random() < args.epsilon:
                    move = rng.choice(cands)
                else:
                    move = global_best
                game.step(move)

    print("wrote %d positions to %s; teacher's global best inside shortlist: %.1f%%"
          % (n, args.out, 100.0 * covered / max(1, n)))


if __name__ == "__main__":
    main()
