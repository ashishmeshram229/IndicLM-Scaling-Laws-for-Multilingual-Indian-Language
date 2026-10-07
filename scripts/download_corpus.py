#!/usr/bin/env python3
"""Download full Wikipedia dumps for all 10 IndicLM languages to HF cache.

Run on the WSAI login node (has internet access; compute nodes do not):
    python scripts/download_corpus.py \
        --hf-cache /storage_server/da25m016/indiclm/hf_cache \
        --output-dir /storage_server/da25m016/indiclm/data/raw_wiki

Languages: Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam,
           Gujarati, Punjabi, English.

The script is resumable: it skips any language whose parquet files already
exist in the output directory. Data volume: ~4–8 GB total (Nov 2023 dump).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

LANGUAGES = {
    "hi": "hindi",
    "mr": "marathi",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "pa": "panjabi",
    "en": "english",
}

# Wikipedia dataset on HuggingFace Hub (Nov 2023 dump, CC BY-SA 3.0)
HF_DATASET = "wikimedia/wikipedia"
HF_DATE = "20231101"


def download_language(
    lang_code: str,
    output_dir: Path,
    hf_cache: Path,
    num_proc: int = 4,
) -> dict:
    import datasets  # type: ignore[import]

    lang_dir = output_dir / lang_code
    done_marker = lang_dir / "_download_complete.json"

    if done_marker.exists():
        with open(done_marker) as f:
            stats = json.load(f)
        print(f"  [{lang_code}] already downloaded ({stats['num_rows']:,} rows) — skipping")
        return stats

    lang_dir.mkdir(parents=True, exist_ok=True)
    config = f"{HF_DATE}.{lang_code}"
    print(f"  [{lang_code}] downloading {config} ...")
    t0 = time.time()

    ds = datasets.load_dataset(
        HF_DATASET,
        config,
        cache_dir=str(hf_cache),
        trust_remote_code=False,
        num_proc=num_proc,
    )

    # Save as parquet shards (fast, columnar, ~3× smaller than jsonl)
    ds["train"].to_parquet(str(lang_dir / "wiki.parquet"))

    elapsed = time.time() - t0
    num_rows = len(ds["train"])
    total_chars = sum(len(t) for t in ds["train"]["text"])

    stats = {
        "lang_code": lang_code,
        "lang_name": LANGUAGES[lang_code],
        "num_rows": num_rows,
        "total_chars": total_chars,
        "approx_mb": round(total_chars / 1e6, 1),
        "download_time_sec": round(elapsed, 1),
    }
    with open(done_marker, "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"  [{lang_code}] done — {num_rows:,} articles, "
        f"~{stats['approx_mb']} MB in {elapsed:.0f}s"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hf-cache",
        default="/storage_server/da25m016/indiclm/hf_cache",
        help="HuggingFace cache directory (keep on NFS, not /tmp)",
    )
    parser.add_argument(
        "--output-dir",
        default="/storage_server/da25m016/indiclm/data/raw_wiki",
        help="Directory to write per-language parquet files",
    )
    parser.add_argument(
        "--languages",
        default=",".join(LANGUAGES.keys()),
        help="Comma-separated language codes (default: all 10)",
    )
    parser.add_argument("--num-proc", type=int, default=4, help="Parallel workers for HF datasets")
    args = parser.parse_args()

    hf_cache = Path(args.hf_cache)
    output_dir = Path(args.output_dir)
    hf_cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache)

    lang_codes = [c.strip() for c in args.languages.split(",")]
    unknown = set(lang_codes) - set(LANGUAGES)
    if unknown:
        print(f"ERROR: unknown language codes: {unknown}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading Wikipedia for: {lang_codes}")
    print(f"HF cache: {hf_cache}")
    print(f"Output:   {output_dir}")
    print()

    all_stats = []
    total_t0 = time.time()

    for lang_code in lang_codes:
        try:
            stats = download_language(lang_code, output_dir, hf_cache, args.num_proc)
            all_stats.append(stats)
        except Exception as e:
            print(f"  [{lang_code}] FAILED: {e}", file=sys.stderr)

    total_elapsed = time.time() - total_t0
    total_mb = sum(s["approx_mb"] for s in all_stats)
    total_rows = sum(s["num_rows"] for s in all_stats)

    summary = {
        "languages_downloaded": len(all_stats),
        "total_articles": total_rows,
        "total_mb": round(total_mb, 1),
        "total_time_sec": round(total_elapsed, 1),
        "per_language": all_stats,
    }
    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 50)
    print(f"Downloaded {len(all_stats)}/{len(lang_codes)} languages")
    print(f"Total: {total_rows:,} articles, ~{total_mb:.0f} MB in {total_elapsed:.0f}s")
    print(f"Summary: {summary_path}")
    print()
    print("Next step (login node, no internet needed on compute nodes):")
    print("  sbatch deployment/slurm/data_pipeline.sbatch")


if __name__ == "__main__":
    main()
