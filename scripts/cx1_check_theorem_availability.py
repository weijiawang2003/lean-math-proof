"""CX1 Stage 3 — availability check for discovered theorems.

Strategy. Probing every one of the ~1843 fresh-with-tactic-proof
candidates against LeanDojo would cost hours. Instead this script:

  1. Builds a small per-file sample (default 3 theorems per source
     file) drawn from project/discovered_theorems_cx1.json.
  2. For each sample, opens `Dojo(theorem)` and verifies that the
     initial tactic state can be obtained. Classifies outcome as
       - available     : Dojo entered, initial state obtained
       - import_error  : Dojo refused — file not in traced repo
       - name_collision: full_name not found in file
       - timeout       : Dojo entry exceeded the per-theorem watchdog
       - unavailable   : any other failure
  3. Treats every theorem in a file where ≥1 sample is `available`
     as PRESUMED AVAILABLE for downstream theorem-set construction.
     Files with 0 available samples are marked UNAVAILABLE and
     their theorems are excluded.

Writes:
  - project/data/cx1_available_theorems.json
  - project/evolve/reports/cx1_availability_report.md

This is intentionally a coarse probe — it is meant to gate at the
file level. The fine-grained per-theorem availability filter happens
later when eval_rollout_all.py runs the actual evals.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

DISCOVERED = Path("project/discovered_theorems_cx1.json")
AVAIL_OUT = Path("project/data/cx1_available_theorems.json")
MD_OUT = Path("project/evolve/reports/cx1_availability_report.md")

DEFAULT_SAMPLES_PER_FILE = 3
DEFAULT_TIMEOUT_PER_PROBE = 90  # seconds


class _Timeout(Exception):
    pass


def _handler(_signum, _frame):
    raise _Timeout()


def _probe_one(theorem_cfg: dict, timeout_s: int) -> tuple[str, str | None]:
    """Try to enter the theorem in Dojo. Returns (status, reason)."""
    # Lazy import so the catalog-audit stage 1 / 2 stages don't pull
    # LeanDojo in.
    try:
        from lean_dojo import Dojo, Theorem
        from env import make_repo
    except Exception as exc:  # noqa: BLE001
        return ("unavailable", f"import error: {exc}")

    repo = make_repo()
    thm = Theorem(repo=repo, file_path=theorem_cfg["file_path"],
                  full_name=theorem_cfg["full_name"])
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        with Dojo(thm) as (dojo, state):
            # If we get here, the theorem entered and we have an
            # initial state.
            _ = state.pp  # touch it
            return ("available", None)
    except _Timeout:
        return ("timeout", f">{timeout_s}s")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "import" in low or "no such file" in low:
            return ("import_error", msg[:140])
        if "not found" in low or "collision" in low:
            return ("name_collision", msg[:140])
        return ("unavailable", msg[:140])
    finally:
        signal.alarm(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-file", type=int,
                    default=DEFAULT_SAMPLES_PER_FILE)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_PER_PROBE)
    ap.add_argument("--max-files", type=int, default=0,
                    help="Limit probe to first N files (0 = all)")
    args = ap.parse_args()

    data = json.loads(DISCOVERED.read_text(encoding="utf-8"))
    thms = data["theorems"]
    by_file: dict[str, list[dict]] = defaultdict(list)
    for t in thms:
        if not t.get("has_tactic_proof"):
            continue
        by_file[t["file_path"]].append(t)
    files = sorted(by_file)
    if args.max_files:
        files = files[:args.max_files]

    print(f"probing {len(files)} files × {args.samples_per_file} "
          f"samples (timeout {args.timeout}s/probe)")

    file_status: dict[str, dict] = {}
    sample_results: list[dict] = []
    t_start = time.time()
    for i, fp in enumerate(files, 1):
        cand = by_file[fp][:args.samples_per_file]
        statuses: Counter[str] = Counter()
        for c in cand:
            t0 = time.time()
            status, reason = _probe_one(c, args.timeout)
            dt = time.time() - t0
            sample_results.append({
                "file_path": fp,
                "full_name": c["full_name"],
                "status": status,
                "reason": reason,
                "elapsed_s": round(dt, 1),
            })
            statuses[status] += 1
            print(f"  [{i}/{len(files)}] {c['full_name']}: {status} ({dt:.0f}s)")
        any_avail = statuses["available"] > 0
        file_status[fp] = {
            "samples": args.samples_per_file,
            "available": statuses.get("available", 0),
            "import_error": statuses.get("import_error", 0),
            "name_collision": statuses.get("name_collision", 0),
            "timeout": statuses.get("timeout", 0),
            "unavailable": statuses.get("unavailable", 0),
            "verdict": "PRESUMED_AVAILABLE" if any_avail else "UNAVAILABLE",
        }
    elapsed = time.time() - t_start

    # Build the available-theorems list = every theorem from files
    # marked PRESUMED_AVAILABLE.
    available_thms: list[dict] = []
    for fp, info in file_status.items():
        if info["verdict"] == "PRESUMED_AVAILABLE":
            available_thms.extend(by_file[fp])
    avail_count_by_ns: Counter[str] = Counter()
    for t in available_thms:
        ns = (t["full_name"].split(".", 1)[0] if "." in t["full_name"]
              else "_no_ns_")
        avail_count_by_ns[ns] += 1

    out = {
        "samples_per_file": args.samples_per_file,
        "timeout_per_probe": args.timeout,
        "files_probed": len(file_status),
        "elapsed_s": round(elapsed, 1),
        "file_status": file_status,
        "sample_results": sample_results,
        "available_theorem_count": len(available_thms),
        "available_by_namespace": dict(avail_count_by_ns),
        "theorems": available_thms,
    }
    AVAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    AVAIL_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown report.
    lines: list[str] = []
    lines.append("# CX1 — availability report\n")
    lines.append(f"Probed **{len(file_status)} files × {args.samples_per_file} "
                 f"theorems** = {len(sample_results)} samples in "
                 f"{elapsed:.0f}s.\n")
    lines.append("## Per-file verdicts\n")
    lines.append("| file | available | import_err | timeout | unavail | verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for fp, info in sorted(file_status.items()):
        lines.append(
            f"| `{fp}` | {info['available']} | {info['import_error']} | "
            f"{info['timeout']} | {info['unavailable']} | {info['verdict']} |"
        )
    lines.append("")
    lines.append("## Available theorem count by namespace\n")
    lines.append("| namespace | count |")
    lines.append("|---|---:|")
    for ns, c in avail_count_by_ns.most_common():
        lines.append(f"| `{ns}` | {c} |")
    lines.append("")
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {AVAIL_OUT}")
    print(f"wrote {MD_OUT}")
    print(f"total available theorems (presumed): {len(available_thms)}")


if __name__ == "__main__":
    main()
