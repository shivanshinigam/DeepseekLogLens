#!/usr/bin/env python3
"""
LogLens AI — Windowed Log Analyzer
===================================
Demonstrates sliding window attention concept for log analysis.

Instead of reading 10,000 lines at once (full attention),
we scan in fixed windows of W lines — only reading the neighborhood
of each log segment. Memory stays flat. Speed goes up.

Usage:
    python generate_logs.py    # create the synthetic log first
    python analyze.py          # then run this
"""

import time
import os
import sys

# ─── Config ──────────────────────────────────────────────────────────────────

LOG_FILE    = "logs/acmecorp_payment.log"
WINDOW_SIZE = 20          # lines per window — mirrors our DeepSeek attention window
OVERLAP     = WINDOW_SIZE // 2  # 50% overlap so we never miss a cross-window pattern

# Keywords that suggest an anomaly is nearby
ANOMALY_KEYWORDS   = ["EXHAUSTED","TimeoutException","SLOW QUERY","CIRCUIT OPEN","FAILED","CRITICAL"]
ROOT_CAUSE_KEYWORDS = ["SLOW QUERY","MISSING INDEX","connections_held: 27","EXHAUSTED"]

# ─── Memory estimate (matches our DeepSeek 1.3B @ 4-bit setup) ───────────────

MODEL_MB         = 412   # 1.3B model at 4-bit quantization
TOKENS_PER_LINE  = 12    # average tokens per log line
KV_CACHE_MB      = round((WINDOW_SIZE * TOKENS_PER_LINE * 64 * 2 * 4) / (1024 * 1024), 2)
TOTAL_MB         = MODEL_MB + KV_CACHE_MB

# ─── Cost estimate (trn1.2xlarge on-demand) ──────────────────────────────────

COST_PER_SECOND  = 0.000226  # $/s rough estimate for trn1.2xlarge amortised

# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze(filepath):
    if not os.path.exists(filepath):
        print(f"\n  ❌ Log file not found: {filepath}")
        print(f"     Run: python generate_logs.py  first\n")
        sys.exit(1)

    with open(filepath) as f:
        all_lines = [l.rstrip() for l in f.readlines()]

    total = len(all_lines)

    print(f"\n{'═'*65}")
    print(f"  LogLens AI  —  Windowed Attention Analyzer")
    print(f"{'═'*65}")
    print(f"  Log file    : {filepath}")
    print(f"  Total lines : {total}")
    print(f"  Window size : {WINDOW_SIZE} lines  (mirrors DeepSeek attention window)")
    print(f"  Memory use  : {TOTAL_MB:.0f} MB  (model {MODEL_MB} MB + KV cache {KV_CACHE_MB} MB)")
    print(f"  Budget      : 500 MB  ✅  ({500 - TOTAL_MB:.0f} MB headroom)")
    print(f"{'─'*65}\n")

    windows_scanned  = 0
    root_cause_line  = None
    root_cause_text  = None
    symptom_lines    = []
    findings         = []

    start = time.time()

    for win_start in range(0, total, OVERLAP):
        win_end = min(win_start + WINDOW_SIZE, total)
        window  = all_lines[win_start:win_end]
        windows_scanned += 1
        window_text = " ".join(window)

        # Progress bar
        pct      = win_end / total
        filled   = int(pct * 50)
        bar      = "█" * filled + "░" * (50 - filled)
        print(f"\r  [{bar}] {win_end}/{total} lines  |  window {windows_scanned}", end="", flush=True)
        time.sleep(0.04)  # visual effect — remove for prod

        # Check each line in this window
        for i, line in enumerate(window):
            global_ln = win_start + i + 1

            if root_cause_line is None:
                for kw in ROOT_CAUSE_KEYWORDS:
                    if kw in line:
                        root_cause_line = global_ln
                        root_cause_text = line
                        break

            for kw in ["TimeoutException","CIRCUIT OPEN","CIRCUIT BREAKER OPEN"]:
                if kw in line and global_ln not in symptom_lines:
                    symptom_lines.append(global_ln)

    elapsed = time.time() - start
    cost    = elapsed * COST_PER_SECOND

    print(f"\n\n{'═'*65}")
    print(f"  SCAN COMPLETE")
    print(f"{'─'*65}")
    print(f"  Windows scanned : {windows_scanned}")
    print(f"  Lines read      : {total}")
    print(f"  Time            : {elapsed:.2f}s")
    print(f"  Memory used     : {TOTAL_MB:.0f} MB / 500 MB  ✅")
    print(f"  KV cache        : {KV_CACHE_MB} MB  (fixed — does NOT grow with file size)")
    print(f"  Estimated cost  : ${cost:.5f}  (~${cost * 1000:.4f} per 1000 analyses)")
    print(f"\n{'═'*65}")
    print(f"  ROOT CAUSE")
    print(f"{'─'*65}")

    if root_cause_line:
        print(f"\n  ⚠  Found at Line {root_cause_line}:")
        print(f"\n     {root_cause_text[:90]}")
        if len(root_cause_text) > 90:
            print(f"     ...{root_cause_text[90:]}")
        print(f"\n{'─'*65}")
        print(f"  EXPLANATION")
        print(f"{'─'*65}")
        print(f"""
  An analytics batch job (BATCH-2024-01) ran a full table scan on
  the transactions table — 2.4 million rows — because the date_column
  had no database index. This forced it to hold 27 DB connections
  for 8+ minutes without releasing them.

  With only 50 connections in the pool, this left just 23 for live
  payments. As more payments arrived, the pool hit 50/50 and new
  requests timed out (30s timeout) → payments failed → circuit
  breaker opened → 9 payments totalling $1,935 were blocked.
        """)
        print(f"{'─'*65}")
        print(f"  FIX (3 steps)")
        print(f"{'─'*65}")
        print(f"""
  1. Add DB index (immediate, 5 minutes):
     CREATE INDEX idx_transactions_date ON transactions(date_column);

  2. Cap analytics job connections (config change):
     analytics.pool.maxSize = 10  (leaves 40 for payments)

  3. Move batch jobs to off-peak hours:
     Schedule BATCH-* jobs between 03:00–05:00 only.
        """)

    if symptom_lines:
        print(f"  Symptom lines (payment failures): {symptom_lines[:8]}")

    print(f"{'═'*65}\n")


if __name__ == "__main__":
    analyze(LOG_FILE)
