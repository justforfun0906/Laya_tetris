# Laya Tetris

Teaching a language-model-style decision model to play Tetris by imitating a classic hand-tuned bot.

We take [Laya](https://huggingface.co/convaiinnovations/laya), a 421M-parameter
ModernBERT-large model that picks one answer from a list of options, and fine-tune it on
positions labelled by a strong heuristic Tetris player (the "teacher"). The fine-tuned
model (the "student") then plays Tetris entirely on its own, on a laptop.

## Results

All numbers are from an Apple M2 MacBook (16 GB), training on the MPS backend.

| Policy | Lines cleared (500-piece games) | Tops out? | Agreement with teacher |
|---|---|---|---|
| Random moves | 0 | within ~30 pieces | – |
| Laya, untuned | 0 | within ~30 pieces | 11.9% (≈ random pick from 10) |
| **Laya, fine-tuned (this repo)** | **198 (197–199) avg over 3 games** | **never** | **84.3%** |
| Teacher (heuristic) | ~200 (≈0.4 lines/piece, the max possible) | never | 100% |

- **What 0.4 lines/piece means:** 500 pieces × 4 cells = 2000 cells, and one line is 10
  cells, so ~200 lines is the most you can clear in 500 pieces. The student is essentially
  playing perfectly-efficient Tetris over this horizon.
- **Training:** one run over 15,000 teacher-labelled positions (14,250 train / 750 validation),
  1,336 optimizer steps, about 2 h 50 min.
- **Learning curve:** validation agreement with the teacher went
  11.9% → 70.3% (step 200) → 80.7% (step 600) → 83.5% (step 1200) → 82.8% (final).
- **Speed:** about 170–200 ms per move at inference.

`runs/eval_student.log` has the raw evaluation output.

## How it works

```
 tetris.py ──► legal placements ──► laya_format.shortlist ──► top 10 options
                                                                   │
            teacher.py scores each option (training labels)        │
                                                                   ▼
              text board + options ──► Laya "choice" question ──► chosen move
```

### 1. The game (`tetris.py`)
A ~150-line numpy Tetris engine: 10×20 board, 7-bag randomizer, hard drops only (no
tucks or spins). Instead of frame-by-frame controls, an **action is a placement**: a
(rotation, column) pair. The engine lists every legal placement and its resulting board.

### 2. The teacher (`teacher.py`)
Pierre Dellacherie's heuristic with the
[El-Tetris](https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/)
weights: a weighted sum of six board features (landing height, eroded cells, row and column
transitions, holes, wells). It only looks one piece ahead, yet it is known to clear hundreds
of thousands of lines, so it's a strong, cheap, deterministic teacher.

### 3. Turning Tetris into text (`laya_format.py`)
Laya answers multiple-choice questions over a text "state". Each turn becomes one question:

- **State:** the board as `#`/`.` rows (from just above the stack down to the floor), plus the
  current piece, next piece, and stack height.
- **Options:** up to 10 candidate placements, each described by its consequences, e.g.
  ```
  A: rotation 0 column 3; clears 0, holes 0, lands at 2, max height 5, bumpiness 4
  ```

A Tetris piece usually has 9–34 legal placements, so a simple rule that **doesn't use the
teacher** (fewest holes, most lines, lowest and flattest stack) shortlists 10 of them. The
shortlist is then reordered by (rotation, column) so the model can't just pick the first option.
The teacher's overall best move is inside this shortlist 99.8% of the time.

### 4. Data (`gen_data.py`)
The teacher plays games (with 5–15% random moves for variety). For every position we save the
question and the teacher's score for **each** of the 10 options, not just the best one.

### 5. Training (`train.py`)
**Soft-label distillation:** the loss is cross-entropy between the model's option probabilities
and `softmax(teacher_scores / tau)`. This way the student learns the teacher's whole ranking,
including which moves are nearly as good, rather than only its top choice.

To fit in 16 GB of memory, only the top 8 of 28 encoder layers plus the decision head are
trained (124M of 421M parameters). The head uses a 10× higher learning rate. The output folder
can be loaded directly with `laya.load()`.

### 6. Playing (`play.py`, `watch.py`)
`play.py` runs evaluation games and reports lines cleared and agreement with the teacher.
`watch.py` shows a colourful live game in the terminal that restarts after each top-out.

## Getting started

Developed on Python 3.13. Training runs on Apple Silicon (MPS), CUDA, or CPU
(slowly). The first run downloads the base Laya model (~1.7 GB) from Hugging Face.

```bash
git clone https://github.com/justforfun0906/Laya_tetris.git
cd Laya_tetris
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick check: watch the teacher play (no model needed)

```bash
python play.py --policy teacher --games 3
python watch.py --policy teacher --delay 0.05
```

### Reproduce the trained model

The trained weights (1.6 GB) are too large for GitHub, so rebuild them:

```bash
# 1. Generate 15k teacher-labelled positions (CPU only, no model needed)
python gen_data.py --games 50 --max-pieces 300 --epsilon 0.05 --out data/teacher.jsonl

# 2. Fine-tune Laya (≈3 h on an M2; saves to runs/laya-tetris)
python train.py --data data/teacher.jsonl --epochs 1.5 --out runs/laya-tetris

# 3. Evaluate
python play.py --policy student --model runs/laya-tetris --games 3

# 4. Watch it play live (Ctrl-C to quit)
python watch.py --model runs/laya-tetris
```

For a quick smoke test of the training pipeline, add `--max-rows 200 --epochs 0.2` to `train.py`.

### Optional: DAgger (improve on the student's own mistakes)

Let the student drive while the teacher labels every position the student actually reaches,
then continue training on both datasets:

```bash
python gen_data.py --games 30 --student runs/laya-tetris --student-prob 0.7 --out data/dagger1.jsonl
python train.py --data data/teacher.jsonl data/dagger1.jsonl --init runs/laya-tetris --epochs 1 --out runs/laya-tetris-d1
```

### Use the model from Python

```python
import laya
from tetris import Game
from laya_format import shortlist, build_question, render_state, MAX_LEN, HEAD_MAX_LEN

agent = laya.load("runs/laya-tetris")
agent.cfg["max_len"], agent.cfg["head_max_len"] = MAX_LEN, HEAD_MAX_LEN

game = Game(seed=0)
while not game.over:
    questions, by_label = build_question(shortlist(game.legal()))
    answer = agent.predict(render_state(game), questions)["answers"]["move"]["choice"]
    game.step(by_label[answer])
print(game.lines, "lines")
```

## Useful options

| Script | Flag | Default | Meaning |
|---|---|---|---|
| `gen_data.py` | `--games`, `--max-pieces` | 60, 400 | how much data to generate |
| | `--epsilon` | 0.15 | chance of a random shortlisted move (state diversity) |
| | `--k` | 10 | options per question |
| `train.py` | `--train-layers` | 8 | encoder layers to unfreeze (more = more memory/time) |
| | `--tau` | 2.0 | softness of teacher targets (median gap between the teacher's top two scores is ~11) |
| | `--lr`, `--batch`, `--accum` | 3e-5, 4, 4 | optimizer settings (effective batch 16) |
| | `--init` | base Laya | start from a previous run (for DAgger) |
| `play.py` | `--policy` | teacher | `teacher`, `student`, `laya-base`, or `random` |
| | `--show` | off | print the board after every move |
| `watch.py` | `--delay` | 0 | seconds to pause per move |

## Project layout

```
tetris.py        game engine
teacher.py       El-Tetris heuristic (the teacher)
laya_format.py   board → text question, candidate shortlist
gen_data.py      generate teacher-labelled data (and DAgger data)
train.py         soft-label distillation fine-tuning
play.py          evaluation
watch.py         live terminal viewer
runs/            trained models and logs (git-ignored except eval_student.log)
data/            generated datasets (git-ignored)
```

## Limitations

- Evaluation used 3 games capped at 500 pieces. Longer games would better show how long the
  student survives compared with the teacher.
- The engine has no hold piece, soft drops, tucks or T-spins.
- The option text includes summary numbers (holes, heights, bumpiness), so part of the work is
  done by the feature descriptions, not by the model reading the raw board.
- The model only picks among the 10 shortlisted moves, so it can never do better than the best
  move on that list.

## Credits

- [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-large) by Answer.AI
- Pierre Dellacherie's Tetris heuristic and Yiyuan Lee's
  [El-Tetris](https://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/) weights
