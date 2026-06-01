#!/usr/bin/env python3
"""SF2 Part 1 — build structured failure_cases.json from the real RC1 smoke run.

Authoritative source of solved/failed is the run's metrics.json `per_theorem`
(`finished` field). NOTE: SF1's eval_matrix_results.json mislabels solved status
because its parser keyed on `proof_finished`/`solved` while metrics.json uses
`finished` — this script reads metrics.json directly and records the discrepancy.

Per-theorem final goal / error are recovered from traces.jsonl; never fabricated.

Output:
  project/evolve/experiments/sf2/out/multiset_seed/failure_cases.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

EVAL_RESULTS = "project/evolve/experiments/sf1/out/real/eval_matrix_results.json"
OUT = "project/evolve/experiments/sf2/out/multiset_seed/failure_cases.json"


def _read_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _trace_final(run_dir, full_name):
    """Return (all_tactics, last_error, last_state) from traces for full_name."""
    tactics, errors, states = [], [], []
    for tp in sorted(glob.glob(os.path.join(run_dir or "", "**", "*.jsonl"), recursive=True)):
        for rec in _read_jsonl(tp):
            if not isinstance(rec, dict):
                continue
            if (rec.get("full_name") or rec.get("theorem")) != full_name:
                continue
            if rec.get("tactic"):
                tactics.append(rec["tactic"])
            em = rec.get("error_message")
            if em:
                errors.append(em)
            sp = rec.get("state_pp")
            if sp:
                states.append(sp)
    return tactics, (errors[-1] if errors else None), (states[-1] if states else None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF2: build failure_cases.json")
    p.add_argument("--eval-results", default=EVAL_RESULTS)
    p.add_argument("--out", default=OUT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    er = json.load(open(args.eval_results))
    cases = []
    discrepancies = []
    for res in er.get("results", []):
        run_dir = res.get("run_dir")
        metrics_path = res.get("metrics_path")
        metrics = {}
        if metrics_path and os.path.isfile(metrics_path):
            metrics = json.load(open(metrics_path))
        by_name = {t.get("full_name"): t for t in (metrics.get("per_theorem") or [])}
        sf1_by_name = {t.get("full_name"): t for t in (res.get("per_theorem") or [])}

        for fn, mt in by_name.items():
            finished = bool(mt.get("finished"))
            tactics_used = mt.get("tactics_used") or []
            win = mt.get("winning_tactic")
            win_origin = mt.get("winning_tactic_origin")
            win_template = mt.get("winning_tactic_template_source")
            tr_tactics, tr_err, tr_state = _trace_final(run_dir, fn)
            wx3_fired = any("Multiset.induction_on" in (x or "")
                            for x in (tactics_used + tr_tactics))
            wx3_closed = bool(finished and win and "Multiset.induction_on" in win)
            final_error = mt.get("error_message") or tr_err
            final_goal = tr_state
            level = ("full_trace" if tr_state else
                     "error_only" if final_error else
                     "tactics_only" if (tactics_used or tr_tactics) else "none")
            # SF1-vs-metrics discrepancy check
            sf1_solved = bool((sf1_by_name.get(fn) or {}).get("solved"))
            if sf1_solved != finished:
                discrepancies.append({"full_name": fn, "sf1_eval_matrix_solved": sf1_solved,
                                      "metrics_finished": finished})
            notes = []
            if not finished:
                notes.append("GENUINE RC1 failure (metrics.finished=false)")
                if wx3_fired and not wx3_closed:
                    notes.append("WX3 Multiset.induction_on fired but did NOT close -> real post-oracle failure")
            else:
                notes.append(f"RC1 PROVED via `{win}` (origin={win_origin}"
                             + (f", template={win_template}" if win_template else "") + ")")
                if wx3_closed:
                    notes.append("closed by the WX3 Multiset induction oracle")
            if sf1_solved != finished:
                notes.append("NOTE: SF1 eval_matrix_results.json mislabels this as "
                             f"solved={sf1_solved} (parser bug: used 'proof_finished'/'solved' "
                             "instead of metrics key 'finished'); metrics.json is authoritative")
            cases.append({
                "full_name": fn,
                "file_path": mt.get("file_path"),
                "namespace": fn.split(".")[0] if "." in fn else None,
                "rc1_solved": finished,
                "rc1_winning_tactic": win,
                "rc1_winning_tactic_origin": win_origin,
                "wx3_fired": wx3_fired,
                "wx3_closed": wx3_closed,
                "last_tried_tactics": tactics_used,
                "final_error": final_error,
                "final_goal": final_goal,
                "trace_paths": sorted(glob.glob(os.path.join(run_dir or "", "**", "*.jsonl"),
                                                recursive=True)) if run_dir else [],
                "metrics_path": metrics_path,
                "available_context_level": level,
                "notes": notes,
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"source_run": (er.get("results", [{}]) or [{}])[0].get("run_dir"),
               "rc1_solved_count": sum(1 for c in cases if c["rc1_solved"]),
               "rc1_failed_count": sum(1 for c in cases if not c["rc1_solved"]),
               "sf1_metrics_discrepancies": discrepancies,
               "cases": cases}, open(args.out, "w"), ensure_ascii=False, indent=2)
    nfail = sum(1 for c in cases if not c["rc1_solved"])
    print(f"[sf2:cases] wrote {len(cases)} cases (RC1 solved "
          f"{sum(1 for c in cases if c['rc1_solved'])}/{len(cases)}, "
          f"genuine failures={nfail}) -> {args.out}")
    if discrepancies:
        print(f"[sf2:cases] SF1 metrics-parse discrepancies: {discrepancies}")
    for c in cases:
        print(f"  {c['full_name']}: rc1_solved={c['rc1_solved']} win={c['rc1_winning_tactic']} "
              f"wx3_fired={c['wx3_fired']} wx3_closed={c['wx3_closed']} ctx={c['available_context_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
