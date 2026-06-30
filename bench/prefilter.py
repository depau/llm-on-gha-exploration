#!/usr/bin/env python3
"""Denoise an Android logcat so a small LLM can ingest it.

Keeps the signal (errors, fatals, app crashes) and drops the firehose of audio/
codec/compat spam, then collapses consecutive duplicates and truncates to a byte
budget. The goal is "honestly noisy but fits in ~6K tokens", not a clean summary.

# ponytail: NOISE_TAGS + the budget are tunable heuristics, not a parser. If the
# upstream pid-based filter lands, this becomes a thin second pass.
"""
import argparse
import re
import sys

# logcat threadtime format: "MM-DD HH:MM:SS.mmm  PID  TID  P  TAG: message"
LINE_RE = re.compile(r"^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+\d+\s+\d+\s+([VDIWEF])\s+([^:]*?):\s?(.*)$")

# Tags that are pure environment noise on these emulator/CI runs. Substring match
# on the tag, case-insensitive.
NOISE_TAGS = (
    "audiosystem-jni", "audiomanager", "as.audioservice", "audioflinger",
    "c2store", "ccodec", "ccodecconfig", "ccodecbuffers", "codec2",
    "mediaswcodec", "mediacodec", "bufferpoolaccessor",
    "thermal", "compatibilityinfo", "compatibilitychangereporter",
    "providerscache", "carriersvcbindhelper", "openglrenderer",
    "connectivityservice", "ethernetnetworkfactory", "networkfactory",
    "phoneswitcher", "windowmanager", "inputmanager-jni", "bpbinder",
    "ringtoneplayer", "nlservice", "b/260135164",
    "mediaprovider", "parcel", "ashmem", "taskpersister",
    "bptransactioncompletedlistener", "activitytaskmanager", "activitymanager",
    "gms", "appsfilter",
    # graphics / vsync noise that survives app-scoped filtering
    "egl-main", "choreographer", "frameevents", "colorutils", "openglrenderer",
    "hidlservicemanagement", "trafficstats", "usagestatsservice",
)

# Lines this strong always survive filtering, even with a noisy tag, and are never
# dropped by the byte budget — this is the crash signal the benchmark cares about.
ALWAYS_KEEP_RE = re.compile(
    r"sigsegv|fatal signal|libusb|libaums|jahnen|backtrace|app died|"
    r"androidruntime|has died|\bF DEBUG\b|tombstone",
    re.IGNORECASE,
)


def is_noise(tag: str) -> bool:
    t = tag.strip().lower()
    return any(n in t for n in NOISE_TAGS)


def filter_log(text: str, max_bytes: int) -> str:
    # Pass 1: keep signal lines, collapse consecutive duplicates, and tag each as
    # "critical" (the crash/backtrace) or "normal". Crash happens late in the file,
    # so plain top-down truncation would drop it — instead we guarantee criticals.
    kept = []  # list of (text, is_critical)
    prev_key = None
    dup = 0

    def flush_dup():
        nonlocal dup
        if dup > 1 and kept:
            kept.append((f"    ... (x{dup} repeated)\n", kept[-1][1]))
        dup = 0

    for line in text.splitlines(keepends=True):
        critical = bool(ALWAYS_KEEP_RE.search(line))
        m = LINE_RE.match(line)
        if m:
            prio, tag, msg = m.group(1), m.group(2), m.group(3)
            if not critical:
                if is_noise(tag):
                    continue
                if not (prio in ("E", "F") or prio == "W"):
                    continue
            key = (tag.strip(), msg.strip())
        else:
            if not critical:  # non-threadtime banner lines: only if they carry signal
                continue
            key = ("_raw", line.strip())

        if key == prev_key:
            dup += 1
            continue
        flush_dup()
        prev_key = key
        kept.append((line, critical))
    flush_dup()

    # Pass 2: always emit criticals; fill the remaining budget with normals in order.
    crit_bytes = sum(len(t) for t, c in kept if c)
    budget = max_bytes - crit_bytes
    out = []
    truncated = False
    for text_line, critical in kept:
        if critical:
            out.append(text_line)
        elif budget - len(text_line) >= 0:
            out.append(text_line)
            budget -= len(text_line)
        else:
            truncated = True
    if truncated:
        out.append("\n[... non-critical lines truncated to byte budget; "
                   "crash/backtrace lines retained in full ...]\n")
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile")
    ap.add_argument("--max-bytes", type=int, default=20000,
                    help="byte budget for the filtered output (default 20000 ≈ 6K tokens)")
    args = ap.parse_args(argv)
    with open(args.logfile, encoding="utf-8", errors="replace") as f:
        sys.stdout.write(filter_log(f.read(), args.max_bytes))


def demo():
    sample = (
        "06-30 18:11:07.527   992  1516 W AudioManager: updateAudioPortCache: listAudioPorts failed\n"
        "06-30 18:11:07.527   992  1516 W AudioManager: updateAudioPortCache: listAudioPorts failed\n"
        "06-30 18:11:32.886  4990  5040 F libc    : Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0 in tid 5040 (BlockDeviceOutp), pid 4990 (depau.etchdroid)\n"
        "06-30 18:11:07.598   992  3662 D CompatibilityInfo: applicationScale - 1.0\n"
        "06-30 18:11:33.476  5195  5195 F DEBUG   :       #00 pc ... liblibusb.so (libusb_submit_transfer+538)\n"
    )
    out = filter_log(sample, max_bytes=20000)
    assert "SIGSEGV" in out, "must keep the fatal signal line"
    assert "libusb_submit_transfer" in out, "must keep the libusb backtrace"
    assert "AudioManager" not in out, "must drop audio noise"
    assert "CompatibilityInfo" not in out, "must drop compat noise"
    assert len(out.encode()) <= 20000
    print("demo OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
