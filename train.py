"""Distil the teacher into Laya: fine-tune on teacher-scored positions, save a laya.load()-able dir.

    python train.py --data data/teacher.jsonl --out runs/laya-tetris
    python train.py --data data/teacher.jsonl data/dagger1.jsonl --init runs/laya-tetris --out runs/laya-tetris-d1

Loss is cross-entropy against softmax(teacher_score / tau) over the shortlisted options, i.e.
soft-label distillation: the student learns the teacher's whole ranking, not just its argmax.
Only the top `--train-layers` encoder layers plus the decision head are trained, which keeps
memory within a 16 GB Apple Silicon machine.
"""
import argparse
import json
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import laya
from laya.common import QTYPES, build_sequence, collate_items
from laya_format import HEAD_MAX_LEN, INSTRUCTIONS, LABELS, MAX_LEN

BASE = "convaiinnovations/laya"
FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def encode(tok, row, tau):
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": dict(zip(LABELS, row["options"]))}
    ids, markers = build_sequence(tok, row["state"], q, MAX_LEN, HEAD_MAX_LEN)
    if len(markers) != len(row["options"]):
        raise ValueError("options overflow head_max_len=%d; lower --k or raise HEAD_MAX_LEN" % HEAD_MAX_LEN)
    s = np.asarray(row["teacher_scores"], dtype=np.float64)
    target = np.exp((s - s.max()) / tau)
    return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
            "target": (target / target.sum()).tolist(), "label": row["label"]}


def freeze(model, train_layers):
    for p in model.parameters():
        p.requires_grad = False
    enc = model.encoder
    trainable = list(enc.layers[len(enc.layers) - train_layers:]) if train_layers > 0 else []
    trainable += [enc.final_norm, model.head, model.type_emb, model.scorer]
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("training %.1fM of %.1fM parameters (top %d/%d encoder layers + head)"
          % (n / 1e6, sum(p.numel() for p in model.parameters()) / 1e6, train_layers, len(enc.layers)))


@torch.no_grad()
def evaluate(model, items, pad_id, device, bs):
    model.eval()
    hit = 0
    for i in range(0, len(items), bs):
        b = collate_items([[it] for it in items[i:i + bs]], pad_id)
        logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        hit += (logits.argmax(-1).cpu() == b["label"]).sum().item()
    model.train()
    return hit / max(1, len(items))


def save(model, cfg, src_dir, out):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(src_dir, sub), os.path.join(out, sub), dirs_exist_ok=True)
    cfg = dict(cfg, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN,
               # the shipped calibration was fitted on the base task; it does not transfer
               temperature=[1.0, 1.0, 1.0], temperature_by_options={})
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(out, "model.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/teacher.jsonl"])
    ap.add_argument("--init", default=BASE, help="base checkpoint or a previous run dir")
    ap.add_argument("--out", default="runs/laya-tetris")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder LR; head uses 10x")
    ap.add_argument("--train-layers", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.0, help="teacher softmax temperature (score units)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=0, help="subsample for quick runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = laya.load(args.init, device=args.device)
    model, tok, device = agent.model, agent.tok, agent.device
    if os.path.isdir(args.init):
        src_dir = args.init
    else:
        from huggingface_hub import snapshot_download
        src_dir = snapshot_download(args.init, allow_patterns=FILES)

    rows = load_rows(args.data)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[:args.max_rows]
    items = [encode(tok, r, args.tau) for r in tqdm(rows, desc="tokenize")]
    n_val = max(1, int(len(items) * args.val_frac))
    val, train = items[:n_val], items[n_val:]
    print("%d train / %d val positions, mean length %.0f tokens, device %s"
          % (len(train), len(val), np.mean([len(it["ids"]) for it in items]), device))

    freeze(model, args.train_layers)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 10}], weight_decay=0.01)
    steps = math.ceil(len(train) * args.epochs / (args.batch * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / steps))

    print("val top-1 agreement before training: %.1f%%" % (100 * evaluate(model, val, tok.pad_token_id, device, args.batch)))
    model.train()
    order, micro, total_micro = [], 0, steps * args.accum
    bar = tqdm(total=steps, desc="train")
    run_loss = None
    while micro < total_micro:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        idx, order = order[:args.batch], order[args.batch:]
        b = collate_items([[train[i]] for i in idx], tok.pad_token_id)
        logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
        logp = F.log_softmax(logits, -1)
        target = b["target"].to(device)
        loss = -(target * logp.masked_fill(~b["marker_mask"].to(device), 0)).sum(-1).mean()
        (loss / args.accum).backward()
        micro += 1
        run_loss = loss.item() if run_loss is None else 0.95 * run_loss + 0.05 * loss.item()
        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            bar.update(1)
            bar.set_postfix(loss="%.3f" % run_loss)
            if bar.n % 200 == 0 or bar.n == steps:
                acc = evaluate(model, val, tok.pad_token_id, device, args.batch)
                tqdm.write("step %d: loss %.3f, val top-1 agreement %.1f%%" % (bar.n, run_loss, 100 * acc))
                save(model, agent.cfg, src_dir, args.out)
    bar.close()
    save(model, agent.cfg, src_dir, args.out)
    print("saved fine-tuned model to %s (load with laya.load(%r))" % (args.out, args.out))


if __name__ == "__main__":
    main()
