#!/usr/bin/env python3
"""Count real token counts across all JSONL shards by tokenizing with the trained
SentencePiece model.  Uses a random sample (default 5000 docs/language) to keep
runtime under ~2 minutes; extrapolates to the full corpus using the sample ratio.

Run on the cluster (inside the activated env):
    python deployment/analysis/count_corpus_tokens.py \
        --shards-dir /storage_server/da25m016/indiclm/data/shards \
        --tokenizer  /storage_server/da25m016/indiclm/data/tokenizer/tokenizer.model \
        --output     /storage_server/da25m016/indiclm/experiments/scaling_results/corpus_token_count.json

To count fully (no sampling, takes ~15 min for 1.3M docs):
    ... --sample-per-lang 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import sentencepiece as spm


_BUDGETS = [50_000_000, 200_000_000, 500_000_000, 1_000_000_000]


def count_language(
    shard_path: Path,
    sp: spm.SentencePieceProcessor,
    sample_n: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Tokenize up to sample_n docs from a JSONL shard; return token stats."""
    texts: list[str] = []
    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if text:
                texts.append(text)

    total_docs = len(texts)
    if sample_n and total_docs > sample_n:
        texts = rng.sample(texts, sample_n)
        sampled = len(texts)
    else:
        sampled = total_docs

    token_counts = [len(sp.encode(t, out_type=int)) for t in texts]
    sampled_tokens = sum(token_counts)
    avg_per_doc = sampled_tokens / max(sampled, 1)
    est_total_tokens = int(avg_per_doc * total_docs)

    return {
        "total_docs":         total_docs,
        "sampled_docs":       sampled,
        "sampled_tokens":     sampled_tokens,
        "avg_tokens_per_doc": round(avg_per_doc, 1),
        "est_total_tokens":   est_total_tokens,
        "exact":              sampled == total_docs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir",      required=True, type=Path)
    ap.add_argument("--tokenizer",       required=True, type=Path)
    ap.add_argument("--train-split",     type=float, default=0.9,
                    help="Fraction of tokens used for training (default 0.9)")
    ap.add_argument("--sample-per-lang", type=int, default=5000,
                    help="Docs to sample per language (0 = exact count, slow)")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--output",          type=Path, default=None)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    rng = random.Random(args.seed)

    shard_files = sorted(args.shards_dir.glob("*.jsonl"))
    if not shard_files:
        print(f"ERROR: no *.jsonl files found under {args.shards_dir}")
        return
    print(f"Found {len(shard_files)} shard file(s) in {args.shards_dir}\n")

    print(f"{'lang':>10}  {'docs':>10}  {'sampled':>8}  "
          f"{'avg tok/doc':>12}  {'est tokens':>14}  exact?")
    print("-" * 68)

    per_lang: dict[str, dict[str, Any]] = {}
    total_docs = 0
    total_est_tokens = 0
    all_exact = True

    for shard_path in shard_files:
        lang = shard_path.stem
        stats = count_language(shard_path, sp, args.sample_per_lang, rng)
        per_lang[lang] = stats
        total_docs += stats["total_docs"]
        total_est_tokens += stats["est_total_tokens"]
        if not stats["exact"]:
            all_exact = False
        exact_marker = "✓" if stats["exact"] else "~"
        print(f"{lang:>10}  {stats['total_docs']:>10,}  {stats['sampled_docs']:>8,}  "
              f"{stats['avg_tokens_per_doc']:>12.1f}  {stats['est_total_tokens']:>14,}  "
              f"{exact_marker}")

    train_tokens = int(total_est_tokens * args.train_split)
    label = "exact" if all_exact else "estimated"

    print("-" * 68)
    print(f"{'TOTAL':>10}  {total_docs:>10,}  {'':>8}  "
          f"{'':>12}  {total_est_tokens:>14,}  ({label})")
    print(f"\nTrain tokens ({args.train_split*100:.0f}% split): {train_tokens:,}  ({label})")

    print("\n── Epoch counts at standard training budgets ──────────────────────")
    for budget in _BUDGETS:
        ep = budget / max(train_tokens, 1)
        rep = "  *** DATA REPETITION" if ep > 1.0 else ""
        print(f"  {budget/1e6:>5.0f}M tokens → {ep:.3f} epochs{rep}")

    result: dict[str, Any] = {
        "shards_dir":            str(args.shards_dir),
        "tokenizer":             str(args.tokenizer),
        "train_split":           args.train_split,
        "sample_per_lang":       args.sample_per_lang,
        "total_docs":            total_docs,
        "est_total_tokens":      total_est_tokens,
        "est_total_tokens_train": train_tokens,
        "all_exact":             all_exact,
        "per_language":          per_lang,
        "epoch_counts": {
            str(b): round(b / max(train_tokens, 1), 4) for b in _BUDGETS
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
