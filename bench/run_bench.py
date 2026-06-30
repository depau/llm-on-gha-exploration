#!/usr/bin/env python3
"""Run one log-triage inference and record the report + perf metrics.

Engine-agnostic front end over ollama (HTTP) and llama.cpp (llama-cli subprocess).
Both run the same prompt + log and emit a markdown report plus a CSV metrics row,
so a feasibility matrix can be compared after the fact.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_prompt(log_path: Path) -> str:
    instr = (HERE / "prompt.txt").read_text()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    return f"{instr}\n```\n{log}\n```\n"


class PeakRSS(threading.Thread):
    """Poll summed RSS (MB) of processes whose comm matches `name`, every 0.5s.

    # ponytail: crude name-match sampler. Ceiling: misattributes if an unrelated
    # process of the same name runs concurrently — never happens in a CI job. Swap
    # for psutil-by-pid if that stops being true.
    """
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name = name
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(["ps", "-axo", "rss=,comm="], text=True)
                kb = sum(int(l.split(None, 1)[0]) for l in out.splitlines()
                         if self.name in l.split(None, 1)[1])
                self.peak_mb = max(self.peak_mb, kb / 1024)
            except Exception:
                pass
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)
        return round(self.peak_mb, 1)


def run_ollama(model, prompt, num_ctx, num_predict):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0},
    }).encode()
    sampler = PeakRSS("ollama"); sampler.start()
    t0 = time.time()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    # Bypass any HTTP(S)_PROXY for the local ollama server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=1800) as r:
        data = json.load(r)
    wall = time.time() - t0
    peak = sampler.stop()
    pe_n, pe_d = data.get("prompt_eval_count", 0), data.get("prompt_eval_duration", 1)
    e_n, e_d = data.get("eval_count", 0), data.get("eval_duration", 1)
    return {
        "report": data.get("response", "").strip(),
        "prefill_tok_s": round(pe_n / (pe_d / 1e9), 1) if pe_d else 0,
        "gen_tok_s": round(e_n / (e_d / 1e9), 1) if e_d else 0,
        "out_tokens": e_n, "wall_s": round(wall, 1), "peak_mb": peak,
    }


# llama.cpp perf lines, e.g.:
#   ... prompt eval time =  1234.56 ms /   50 tokens ( ... , 40.50 tokens per second)
#   ...        eval time =  5678.90 ms /  100 runs   ( ... , 17.61 tokens per second)
# The gen line says "eval time" but NOT "prompt eval time" -> negative lookbehind.
_PREFILL_TOK_S = re.compile(r"prompt eval time =.*?([\d.]+) tokens per second")
_GEN_TOK_S = re.compile(r"(?<!prompt )eval time =.*?([\d.]+) tokens per second")
_GEN_TOKENS = re.compile(r"(?<!prompt )eval time =.*?/\s*(\d+) runs")


def parse_llama_perf(stderr):
    def first(rx, cast):
        m = rx.search(stderr)
        return cast(m.group(1)) if m else 0
    return {
        "prefill_tok_s": first(_PREFILL_TOK_S, float),
        "gen_tok_s": first(_GEN_TOK_S, float),
        "out_tokens": first(_GEN_TOKENS, int),
    }


def run_llama(llama_bin, model_arg, prompt, num_ctx, num_predict):
    promptfile = Path("_prompt.txt"); promptfile.write_text(prompt)
    cmd = [llama_bin, *model_arg, "-c", str(num_ctx), "-n", str(num_predict),
           "--temp", "0", "-no-cnv", "--no-display-prompt", "-f", str(promptfile)]
    sampler = PeakRSS(Path(llama_bin).name); sampler.start()
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    peak = sampler.stop()
    perf = parse_llama_perf(p.stderr)
    return {"report": p.stdout.strip(), "wall_s": round(wall, 1), "peak_mb": peak, **perf}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["ollama", "llama.cpp"])
    ap.add_argument("--model", required=True, help="ollama tag or label for the report")
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--case", required=True, help="e.g. easy / hard")
    ap.add_argument("--runner-label", default=os.environ.get("RUNNER_LABEL", "local"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=800)
    ap.add_argument("--gguf", help="local GGUF path (llama.cpp)")
    ap.add_argument("--hf-repo", help="HF GGUF repo:quant for llama-cli -hf (llama.cpp)")
    ap.add_argument("--llama-bin", default="llama-cli")
    args = ap.parse_args()

    prompt = read_prompt(args.log)
    if args.engine == "ollama":
        r = run_ollama(args.model, prompt, args.num_ctx, args.num_predict)
    else:
        if args.hf_repo:
            model_arg = ["-hf", args.hf_repo]
        elif args.gguf:
            model_arg = ["-m", args.gguf]
        else:
            ap.error("--gguf or --hf-repo is required for llama.cpp")
        r = run_llama(args.llama_bin, model_arg, prompt, args.num_ctx, args.num_predict)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.runner_label}-{args.engine}-{args.model}-{args.case}".replace("/", "_").replace(":", "_")
    report = args.out_dir / f"{stem}.md"
    report.write_text(
        f"## {args.runner_label} · {args.engine} · {args.model} · {args.case}\n\n"
        f"*prefill {r['prefill_tok_s']} tok/s · gen {r['gen_tok_s']} tok/s · "
        f"{r['wall_s']}s wall · {r['peak_mb']} MB peak · {r['out_tokens']} out-tok*\n\n"
        f"{r['report']}\n"
    )

    csv_path = args.out_dir / "metrics.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["runner", "engine", "model", "case", "prefill_tok_s",
                        "gen_tok_s", "wall_s", "peak_mb", "out_tokens"])
        w.writerow([args.runner_label, args.engine, args.model, args.case,
                    r["prefill_tok_s"], r["gen_tok_s"], r["wall_s"],
                    r["peak_mb"], r["out_tokens"]])
    print(f"[{stem}] {r['gen_tok_s']} gen tok/s, {r['wall_s']}s, {r['peak_mb']} MB -> {report}")


if __name__ == "__main__":
    main()
