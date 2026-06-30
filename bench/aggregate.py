#!/usr/bin/env python3
"""Combine per-job results into one leaderboard + concatenated reports.

Globs results/**/metrics.csv and results/**/*.md (downloaded artifacts) and prints
a single markdown document to stdout.
"""
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")

rows = []
for csv_path in root.rglob("metrics.csv"):
    with csv_path.open(newline="") as f:
        rows.extend(csv.DictReader(f))

print("# LLM CI log-triage benchmark\n")
if rows:
    rows.sort(key=lambda r: (r["case"], -float(r["gen_tok_s"] or 0)))
    print("## Leaderboard\n")
    print("| runner | engine | model | case | prefill tok/s | gen tok/s | wall s | peak MB | out tok |")
    print("|---|---|---|---|--:|--:|--:|--:|--:|")
    for r in rows:
        print(f"| {r['runner']} | {r['engine']} | {r['model']} | {r['case']} | "
              f"{r['prefill_tok_s']} | {r['gen_tok_s']} | {r['wall_s']} | "
              f"{r['peak_mb']} | {r['out_tokens']} |")
else:
    print("_No metrics rows found._")

print("\n## Reports\n")
for md in sorted(root.rglob("*.md")):
    print(md.read_text())
    print("\n---\n")
