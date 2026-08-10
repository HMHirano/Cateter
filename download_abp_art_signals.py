"""
Download ABP/ART signals from MIMIC-III records listed in matching_records.csv.

This script:
  1. Reads matching_records.csv (columns: dir, seg_name, length).
  2. For each record, checks the WFDB header (no signal data downloaded yet)
     to see if it contains an 'ABP' or 'ART' channel.
  3. If found, downloads ONLY that channel's data and saves it as a local
     .csv file, so you never need to call rdrecord over the network again.
  4. Writes a summary CSV listing what was downloaded / skipped / failed.
     Already-downloaded records are skipped on re-runs, so it's safe to
     interrupt and resume.

Requirements:
    pip install wfdb pandas numpy tqdm

Usage:
    python download_abp_art_signals.py
"""

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock

import numpy as np
import pandas as pd
import wfdb

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional, fall back to no-op
    def tqdm(iterable=None, **kwargs):
        class _NoOpBar:
            def update(self, n=1): pass
            def close(self): pass
        return _NoOpBar()


# ----------------------------- CONFIG ---------------------------------- #

INPUT_CSV = "matching_records.csv"          # path to matching_records.csv
OUTPUT_DIR = "downloaded_signals"           # where per-record CSVs go
SUMMARY_CSV = "download_summary.csv"        # log of what happened
TARGET_SIGNALS = {"ABP", "ART"}             # signal names to look for
MAX_RETRIES = 2                             # network retry attempts per record
RETRY_DELAY_SEC = 2                         # base delay between retries
MAX_WORKERS = 1                            # concurrent downloads (raise/lower to taste)
TIMEOUT_SEC = 300                           # ignore/skip a record if it takes longer than this

# ------------------------------------------------------------------------ #

_summary_lock = Lock()

# Tracks when each record's execution ACTUALLY started (set by the worker
# thread itself, the instant it begins running — not when it was merely
# queued). This is what the timeout check must compare against; using the
# submission/queue time instead would falsely flag tasks that are still
# waiting for a free worker as "timed out".
_start_lock = Lock()
_start_times = {}


def find_target_channels(sig_names):
    """Return indices of channels whose name matches ABP or ART (case-insensitive)."""
    return [i for i, name in enumerate(sig_names) if name.strip().upper() in TARGET_SIGNALS]


def download_record_signal(pn_dir, seg_name, channels):
    """Download only the requested channels for a record, with retries."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            record = wfdb.rdrecord(seg_name, pn_dir=pn_dir, channels=channels)
            return record
        except Exception as e:  # network hiccups, missing data, etc.
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
    raise last_err


def save_record_to_csv(record, out_path):
    """Save a wfdb Record's signal(s) to a local CSV with a time column."""
    n_samples = record.p_signal.shape[0]
    fs = record.fs
    time_s = np.arange(n_samples) / fs

    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    df.insert(0, "time_s", time_s)
    df.to_csv(out_path, index=False)


def process_record(pn_dir, seg_name):
    """Process a single record: check header, download target channels if present."""
    t0 = time.time()
    with _start_lock:
        _start_times[seg_name] = t0  # mark true execution start, for the watchdog
    out_path = os.path.join(OUTPUT_DIR, f"{seg_name}.csv")
    result = {
        "dir": pn_dir,
        "seg_name": seg_name,
        "status": None,
        "matched_signals": None,
        "fs": None,
        "n_samples": None,
        "output_path": None,
        "elapsed_sec": None,
        "error": None,
    }

    try:
        # Lightweight check: read header only, no signal data yet
        header = wfdb.rdheader(seg_name, pn_dir=pn_dir)
        channels = find_target_channels(header.sig_name)

        if not channels:
            result["status"] = "skipped_no_target_signal"
        else:
            record = download_record_signal(pn_dir, seg_name, channels)
            save_record_to_csv(record, out_path)

            result["status"] = "downloaded"
            result["matched_signals"] = ",".join(record.sig_name)
            result["fs"] = record.fs
            result["n_samples"] = record.p_signal.shape[0]
            result["output_path"] = out_path

    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records_df = pd.read_csv(INPUT_CSV)
    required_cols = {"dir", "seg_name"}
    if not required_cols.issubset(records_df.columns):
        raise ValueError(f"{INPUT_CSV} must contain columns: {required_cols}")

    # Resume support: load existing summary if present
    if os.path.exists(SUMMARY_CSV):
        summary_df = pd.read_csv(SUMMARY_CSV)
        done_segs = set(summary_df["seg_name"])
        results = summary_df.to_dict("records")
    else:
        done_segs = set()
        results = []

    todo = [
        (row["dir"], row["seg_name"])
        for _, row in records_df.iterrows()
        if row["seg_name"] not in done_segs
    ]
    print(f"{len(done_segs)} already done, {len(todo)} remaining, "
          f"running with {MAX_WORKERS} concurrent workers...")

    def flush():
        with _summary_lock:
            pd.DataFrame(results).to_csv(SUMMARY_CSV, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # future -> (pn_dir, seg_name)  — note: no submit_time here anymore;
        # actual start time is tracked separately in _start_times, set by
        # the worker thread itself once it truly begins executing.
        pending = {
            executor.submit(process_record, pn_dir, seg_name): (pn_dir, seg_name)
            for pn_dir, seg_name in todo
        }

        pbar = tqdm(total=len(pending), desc="Processing records")
        completed_since_flush = 0

        while pending:
            # Poll every 5s so we can also check for records stuck past TIMEOUT_SEC
            done, _ = wait(pending.keys(), timeout=5, return_when=FIRST_COMPLETED)
            now = time.time()

            # 1) Handle futures that actually finished
            for fut in done:
                pn_dir, seg_name = pending.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "dir": pn_dir, "seg_name": seg_name, "status": "failed",
                        "matched_signals": None, "fs": None, "n_samples": None,
                        "output_path": None,
                        "elapsed_sec": round(now - _start_times.get(seg_name, now), 1),
                        "error": f"{type(e).__name__}: {e}",
                    }
                with _summary_lock:
                    results.append(result)
                pbar.update(1)
                completed_since_flush += 1

            # 2) Handle futures still running but past the timeout: ignore/skip them.
            #    Only tasks that have ACTUALLY STARTED (present in _start_times)
            #    are eligible — a task still waiting in the pool's internal
            #    queue for a free worker has not started yet, so it is never
            #    falsely flagged as timed out no matter how long it's been queued.
            #    Note: we stop WAITING on a timed-out future, but the underlying
            #    thread keeps running in the background until it naturally
            #    finishes (Python can't force-kill a thread mid network call) —
            #    its eventual result is simply discarded once it completes.
            timed_out = []
            for fut, (pn_dir, seg_name) in pending.items():
                start = _start_times.get(seg_name)
                if start is not None and (now - start) > TIMEOUT_SEC:
                    timed_out.append(fut)

            for fut in timed_out:
                pn_dir, seg_name = pending.pop(fut)
                result = {
                    "dir": pn_dir, "seg_name": seg_name, "status": "timeout_skipped",
                    "matched_signals": None, "fs": None, "n_samples": None,
                    "output_path": None, "elapsed_sec": ">300",
                    "error": f"exceeded {TIMEOUT_SEC}s timeout, ignored",
                }
                with _summary_lock:
                    results.append(result)
                pbar.update(1)
                completed_since_flush += 1

            if completed_since_flush >= 20:
                flush()
                completed_since_flush = 0

        pbar.close()
        flush()  # final flush

    summary_df = pd.DataFrame(results)
    n_downloaded = (summary_df["status"] == "downloaded").sum()
    n_skipped = (summary_df["status"] == "skipped_no_target_signal").sum()
    n_failed = (summary_df["status"] == "failed").sum()
    n_timeout = (summary_df["status"] == "timeout_skipped").sum()

    print("\nDone.")
    print(f"  Downloaded: {n_downloaded}")
    print(f"  Skipped (no ABP/ART): {n_skipped}")
    print(f"  Ignored (>{TIMEOUT_SEC}s timeout): {n_timeout}")
    print(f"  Failed: {n_failed}")
    print(f"  Signals saved to: {OUTPUT_DIR}/")
    print(f"  Summary log: {SUMMARY_CSV}")
    if n_failed:
        print(f"  ({n_failed} failed records logged in {SUMMARY_CSV} with error details "
              f"— rerun the script to retry just those)")
    if n_timeout:
        print(f"  ({n_timeout} records exceeded the {TIMEOUT_SEC}s timeout and were skipped — "
              f"marked 'timeout_skipped' with elapsed_sec='>300' in {SUMMARY_CSV})")


if __name__ == "__main__":
    main()
