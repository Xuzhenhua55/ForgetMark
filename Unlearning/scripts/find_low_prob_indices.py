#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan fsr results and print indices with ROUGE < threshold for each jsonl file.

Default root: /work/zhb/CTCC-main/fsr
File pattern assumed: <root>/<strategy>/<ratio>_prob_rouge.jsonl

Usage:
  python scripts/find_low_prob_indices.py \
    --root /work/zhb/CTCC-main/fsr \
    --threshold 0.01

Optional:
  --json-out path/to/summary.json     # also dump summary as JSON
  --strategies task,ties,...          # only include these strategies (default: all)
  --print-empty true|false            # print files with zero matches (default: false)
  --glob "**/*.jsonl"                 # custom glob under root
  --max-files N                       # limit number of files processed (for quick check)
  --include-indices true|false        # whether to include index list in output/JSON (default: false)
  --metric rouge|prob                 # choose which metric to compare against threshold (default: rouge)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "y", "t")


def parse_args():
    p = argparse.ArgumentParser(
        description="Locate indices with ROUGE < threshold in fsr jsonl outputs"
    )
    p.add_argument("--root", type=str, default="/work/zhb/CTCC-main/fsr")
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--strategies", type=str, default="", help="Comma-separated strategy names to include; empty for all")
    p.add_argument("--glob", type=str, default="**/*.jsonl", help="Glob pattern under --root")
    p.add_argument("--json-out", type=str, default="", help="Optional path to write JSON summary")
    p.add_argument("--print-empty", type=str2bool, default=False)
    p.add_argument("--max-files", type=int, default=0, help="If >0, limit number of files processed")
    p.add_argument("--include-indices", type=str2bool, default=False, help="Include index list in output/JSON")
    p.add_argument("--metric", type=str, default="rouge", help="Metric to use: 'rouge' or 'prob'")
    return p.parse_args()


def extract_strategy_ratio(root: Path, file_path: Path) -> Tuple[str, str]:
    """Try to extract strategy and ratio from path like root/strategy/ratio_prob_rouge.jsonl"""
    try:
        rel = file_path.relative_to(root)
    except Exception:
        return ("unknown", file_path.stem)
    parts = rel.parts
    strategy = parts[0] if len(parts) >= 2 else "unknown"
    stem = file_path.stem  # e.g., 0.5_0.5_prob_rouge
    ratio = stem.replace("_prob_rouge", "")
    return strategy, ratio


def iter_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                # malformed line; skip but report
                print(f"[WARN] JSON parse error in {file_path} line {line_no}: {e}", file=sys.stderr)
                continue
            yield obj


def collect_low_prob_indices(file_path: Path, threshold: float) -> List[int]:
    indices: List[int] = []
    for obj in iter_jsonl(file_path):
        try:
            prob = None
            fqp = obj.get("forget_Q_A_Prob")
            if isinstance(fqp, dict):
                prob = fqp.get("prob")
            if prob is None:
                continue
            if float(prob) < threshold:
                idx = int(obj.get("index")) if "index" in obj else None
                if idx is not None:
                    indices.append(idx)
        except Exception:
            # Be robust: skip bad entries
            continue
    return indices


def collect_low_rouge_indices(file_path: Path, threshold: float) -> List[int]:
    indices: List[int] = []
    for obj in iter_jsonl(file_path):
        try:
            rouge_val = None
            frg = obj.get("forget_Q_A_ROUGE")
            if isinstance(frg, dict):
                # Prefer commonly used metrics; fall back to any numeric rouge*
                for key in ["rougeL_f1", "rougeL_recall", "rouge1_recall"]:
                    if key in frg and frg.get(key) is not None:
                        try:
                            rouge_val = float(frg.get(key))
                            break
                        except Exception:
                            pass
                if rouge_val is None:
                    for k, v in frg.items():
                        if isinstance(v, (int, float)) and k.lower().startswith("rouge"):
                            rouge_val = float(v)
                            break
            if rouge_val is None:
                continue
            if float(rouge_val) < threshold:
                idx = int(obj.get("index")) if "index" in obj else None
                if idx is not None:
                    indices.append(idx)
        except Exception:
            # Be robust: skip bad entries
            continue
    return indices


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        sys.exit(1)

    allow_strategies = None
    if args.strategies.strip():
        allow_strategies = set(s.strip() for s in args.strategies.split(",") if s.strip())

    files = sorted(root.glob(args.glob))
    # keep only files directly under strategy/ subdirs if possible
    filtered_files: List[Path] = []
    for fp in files:
        if not fp.is_file():
            continue
        # skip hidden or non-jsonl
        if not fp.name.endswith(".jsonl"):
            continue
        # if strategies filter is set, enforce it
        if allow_strategies is not None:
            try:
                rel = fp.relative_to(root)
                strategy = rel.parts[0]
            except Exception:
                strategy = "unknown"
            if strategy not in allow_strategies:
                continue
        filtered_files.append(fp)

    if args.max_files > 0:
        filtered_files = filtered_files[: args.max_files]

    summary: Dict[str, Any] = {}
    total_files = 0
    total_hits = 0

    for fp in filtered_files:
        total_files += 1
        metric = (args.metric or "rouge").strip().lower()
        if metric not in ("rouge", "prob"):
            metric = "rouge"
        if metric == "prob":
            indices = collect_low_prob_indices(fp, args.threshold)
        else:
            indices = collect_low_rouge_indices(fp, args.threshold)
        strategy, ratio = extract_strategy_ratio(root, fp)
        key = f"{strategy}/{fp.name}"
        if indices or args.print_empty:
            item = {
                "strategy": strategy,
                "file": str(fp),
                "ratio": ratio,
                "threshold": args.threshold,
                "count": len(indices),
            }
            if args.include_indices:
                item["indices"] = sorted(indices)
            summary[key] = item
            total_hits += len(indices)

    # Pretty print
    if not summary:
        print("[INFO] No files with indices below threshold were found.")
    else:
        header_metric = (args.metric or "rouge").upper()
        print("# Files with %s < %.6g" % (header_metric, args.threshold))
        for key in sorted(summary.keys()):
            item = summary[key]
            print(f"- {item['strategy']} | {item['ratio']} | {item['file']}")
            if args.include_indices and "indices" in item:
                print(f"  count={item['count']}; indices={item['indices']}")
            else:
                print(f"  count={item['count']}")
        print(f"\n[SUMMARY] files={total_files}, total_low_{(args.metric or 'rouge').lower()}={total_hits}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"threshold": args.threshold, "metric": (args.metric or "rouge").lower(), "results": summary}, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] JSON summary -> {out_path}")


if __name__ == "__main__":
    main()
