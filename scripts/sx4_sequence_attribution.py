#!/usr/bin/env python3
"""SX4 — sequence attribution harness.

Credits a depth-k sequence candidate (A <;> B [<;> ...]) ONLY when it produces a
genuine new win over LITERAL PRODUCTION search. Fixes the depth-(k-1)-controls
over-credit bug that mis-credited SX3_SET_ITE_AESOP (see sx4_methodology.md).

Inputs (all are existing result JSONs — no live Lean is run):
  --baseline-results   literal production results (rc3_run_literal_validation format:
                       {per_theorem:[{full_name,finished,tactics_used,winning_tactic,...}], trace_path})
  --candidate-results  candidate (production ⊕ sequence) results, same format
  --sequence-results   custom sequence-runner attribution (e.g. sx3_minimal_attribution.json:
                       {per_theorem:[{full_name,attribution,winning_sequence,controls_solved,...}]})
  --cases              cases/manifest JSON (context only; theorem list derived from results)
  --sequence           the grouped tactic, e.g. "simp [Set.ite] <;> aesop"
  --candidate-id       e.g. SX3_SET_ITE_AESOP
  --baseline-policy / --candidate-policy   labels

Classification (ordered, see sequence_attribution_schema.json):
  PRODUCTION_SUBSUMED  baseline already solves (the RC3 bug class)
  FAILED_SEQUENCE      candidate does not solve
  DEPTH1_DUPLICATE     A alone or B alone solves
  ROUTING_DUPLICATE    baseline (not finished) closed by an equivalent generic family
  TRUE_SEQUENCE_DELTA  baseline fails, candidate solves, controls fail, no equivalent prod continuation
  TRACE_INSUFFICIENT   cannot distinguish
  NEEDS_REVIEW         otherwise
Only TRUE_SEQUENCE_DELTA carries credit=true.
"""
from __future__ import annotations

import argparse
import json
import os

GENERIC_FAMILIES = {"simp", "simp_all", "aesop", "classical <;> aesop", "tauto"}


def _split_sequence(seq):
    return [p.strip() for p in seq.split("<;>") if p.strip()]


def _index(results):
    per = results.get("per_theorem", results if isinstance(results, list) else [])
    return {r["full_name"]: r for r in per}


def _load_trace(trace_path):
    rows_by = {}
    if not trace_path or not os.path.isfile(trace_path):
        return rows_by
    for line in open(trace_path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rows_by.setdefault(rec.get("full_name"), []).append(rec)
    return rows_by


def _trace_analysis(full_name, A, B, baseline_row, trace_rows):
    """Detect whether the literal-production baseline already reaches an A-advanced
    state and applies B there (the subsumption mechanism)."""
    used = (baseline_row or {}).get("tactics_used") or []
    reaches_A = applies_B = equiv = None
    confidence = "insufficient"
    winning_path = used

    rows = trace_rows.get(full_name)
    if rows:
        confidence = "full"
        a_advanced = any(
            r.get("tactic") == A and not r.get("proof_finished")
            and (r.get("result_kind") in ("TacticState",) or r.get("num_goals_after") not in (None,))
            and r.get("result_kind") != "LeanError"
            for r in rows)
        b_closed = any(r.get("tactic") == B and r.get("proof_finished") for r in rows)
        # B applied at a step strictly after some advancing A
        a_steps = [r.get("step") for r in rows if r.get("tactic") == A and not r.get("proof_finished")
                   and r.get("result_kind") == "TacticState"]
        b_close_steps = [r.get("step") for r in rows if r.get("tactic") == B and r.get("proof_finished")]
        b_after_a = bool(a_steps and b_close_steps and min(a_steps) < max(b_close_steps))
        reaches_A = bool(a_advanced)
        applies_B = bool(b_closed)
        equiv = bool(a_advanced and b_closed and b_after_a)
    elif used:
        # fall back to winning path: did the baseline win via [..., A, ..., B] ?
        confidence = "partial"
        try:
            ia, ib = used.index(A), used.index(B)
            reaches_A = True
            applies_B = True
            equiv = ia < ib
        except ValueError:
            reaches_A = A in used
            applies_B = B in used
            equiv = (A in used and B in used)
    return {
        "baseline_reaches_A_equivalent_state": reaches_A,
        "baseline_applies_B_after_A_state": applies_B,
        "equivalent_sequence_observed": equiv,
        "baseline_winning_path": winning_path,
        "trace_confidence": confidence,
    }


def _seq_runner(seq_row, full_sequence_tactic):
    """Normalize a custom-runner record into (solved, controls_solved, proxy_credited)."""
    if not seq_row:
        return None, [], False
    controls = list(seq_row.get("controls_solved") or [])
    attribution = seq_row.get("attribution")
    win_seq = seq_row.get("winning_sequence")
    solved = (win_seq == full_sequence_tactic) or attribution in (
        "TRUE_DEPTH2_SEQUENCE_WIN", "SINGLE_STEP_DUPLICATE", "new_depth2_win")
    proxy_credited = attribution in ("TRUE_DEPTH2_SEQUENCE_WIN", "new_depth2_win")
    return solved, controls, proxy_credited


def classify(full_name, A, B, full_seq, baseline_row, cand_row, seq_row, trace_rows):
    baseline_finished = bool((baseline_row or {}).get("finished"))
    candidate_finished = bool((cand_row or {}).get("finished"))
    seq_solved, controls_solved, proxy_credited = _seq_runner(seq_row, full_seq)

    A_initial = "solved" if A in controls_solved else ("failed" if seq_row else "unknown")
    B_initial = "solved" if B in controls_solved else ("failed" if seq_row else "unknown")
    A_then_B = "solved" if (seq_solved or candidate_finished) else ("failed" if (seq_row or cand_row) else "unknown")
    generic_hit = sorted(set(controls_solved) & GENERIC_FAMILIES)

    ta = _trace_analysis(full_name, A, B, baseline_row, trace_rows)

    notes = []
    if baseline_finished:
        cls, credit = "PRODUCTION_SUBSUMED", False
        if ta["equivalent_sequence_observed"]:
            notes.append(f"literal baseline solves via equivalent {A} -> {B} continuation "
                         f"(path={ta['baseline_winning_path']})")
        else:
            notes.append(f"literal baseline already solves (path={ta['baseline_winning_path']})")
    elif not candidate_finished and not seq_solved:
        cls, credit = "FAILED_SEQUENCE", False
        notes.append("candidate sequence does not solve the theorem")
    elif A_initial == "solved" or B_initial == "solved":
        cls, credit = "DEPTH1_DUPLICATE", False
        notes.append(f"depth-1 control solves: A_initial={A_initial}, B_initial={B_initial} "
                     f"(controls_solved={controls_solved})")
    elif generic_hit:
        cls, credit = "ROUTING_DUPLICATE", False
        notes.append(f"baseline closed by equivalent generic family: {generic_hit}")
    elif (not baseline_finished and (candidate_finished or seq_solved)
          and A_then_B == "solved" and A_initial != "solved" and B_initial != "solved"
          and not ta["equivalent_sequence_observed"] and ta["trace_confidence"] != "insufficient"):
        cls, credit = "TRUE_SEQUENCE_DELTA", True
        notes.append("baseline fails; candidate solves; depth-1 controls fail; "
                     "no equivalent production A->B continuation observed")
    elif ta["trace_confidence"] == "insufficient":
        cls, credit = "TRACE_INSUFFICIENT", False
        notes.append("baseline trace does not expose enough state to distinguish subsumption")
    else:
        cls, credit = "NEEDS_REVIEW", False
        notes.append("no rule cleanly applies")

    return {
        "candidate_id": None,  # filled by caller
        "sequence": [A, B] if len(_split_sequence(full_seq)) == 2 else _split_sequence(full_seq),
        "full_sequence_tactic": full_seq,
        "theorem": full_name,
        "baseline_finished": baseline_finished,
        "candidate_finished": candidate_finished,
        "sequence_runner_solved": seq_solved if seq_row else None,
        "proxy_runner_credited": proxy_credited,
        "controls": {"A_initial": A_initial, "B_initial": B_initial, "A_then_B": A_then_B,
                     "baseline_controls_solved": controls_solved},
        "production_trace_analysis": ta,
        "classification": cls,
        "credit": credit,
        "notes": notes,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-results", required=True)
    p.add_argument("--candidate-results", required=True)
    p.add_argument("--sequence-results", default=None)
    p.add_argument("--cases", default=None)
    p.add_argument("--sequence", default="simp [Set.ite] <;> aesop")
    p.add_argument("--candidate-id", default="SX3_SET_ITE_AESOP")
    p.add_argument("--baseline-policy", default="RC2")
    p.add_argument("--candidate-policy", default="RC3_candidate")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    parts = _split_sequence(args.sequence)
    A, B = parts[0], parts[-1]
    base = json.load(open(args.baseline_results))
    cand = json.load(open(args.candidate_results))
    seq = json.load(open(args.sequence_results)) if args.sequence_results else {}
    bidx, cidx, sidx = _index(base), _index(cand), _index(seq)
    trace_rows = _load_trace(base.get("trace_path"))

    names = sorted(set(bidx) | set(cidx) | set(sidx))
    records = []
    for fn in names:
        rec = classify(fn, A, B, args.sequence, bidx.get(fn), cidx.get(fn), sidx.get(fn), trace_rows)
        rec["candidate_id"] = args.candidate_id
        rec["literal_baseline_policy"] = args.baseline_policy
        rec["candidate_policy"] = args.candidate_policy
        records.append(rec)

    hist = {}
    for r in records:
        hist[r["classification"]] = hist.get(r["classification"], 0) + 1
    credited = sorted(r["theorem"] for r in records if r["credit"])
    subsumed = sorted(r["theorem"] for r in records if r["classification"] == "PRODUCTION_SUBSUMED")
    # over-credit caught: proxy runner credited it, SX4 does not
    proxy_credited = sorted(r["theorem"] for r in records if r.get("proxy_runner_credited"))
    over_credit = sorted(r["theorem"] for r in records if r.get("proxy_runner_credited") and not r["credit"])

    out = {
        "candidate_id": args.candidate_id,
        "full_sequence_tactic": args.sequence,
        "literal_baseline_policy": args.baseline_policy,
        "candidate_policy": args.candidate_policy,
        "inputs": {"baseline": args.baseline_results, "candidate": args.candidate_results,
                   "sequence": args.sequence_results, "cases": args.cases,
                   "baseline_trace": base.get("trace_path")},
        "num_theorems": len(records),
        "classification_histogram": hist,
        "credited_true_sequence_deltas": credited,
        "num_credited": len(credited),
        "production_subsumed": subsumed,
        "proxy_runner_credited": proxy_credited,
        "num_proxy_credited": len(proxy_credited),
        "over_credit_caught_theorems": over_credit,
        "over_credit_caught": len(over_credit) > 0,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    _write_md(out, args.out_md)
    print(f"[sx4] {args.candidate_id}: credited={len(credited)} "
          f"subsumed={len(subsumed)} proxy_credited={len(proxy_credited)} "
          f"over_credit_caught={out['over_credit_caught']}")
    return 0


def _write_md(out, path):
    L = [f"# SX4 sequence attribution — `{out['candidate_id']}`", "",
         f"- sequence: `{out['full_sequence_tactic']}`",
         f"- baseline policy (literal production): **{out['literal_baseline_policy']}**",
         f"- theorems analyzed: **{out['num_theorems']}**",
         f"- **credited TRUE_SEQUENCE_DELTA: {out['num_credited']}** {out['credited_true_sequence_deltas']}",
         f"- proxy runner credited: {out['num_proxy_credited']} {out['proxy_runner_credited']}",
         f"- **over-credit caught: {out['over_credit_caught']}** "
         f"({len(out['over_credit_caught_theorems'])} theorems proxy-credited but SX4 declines): "
         f"{out['over_credit_caught_theorems']}",
         "", "## Classification histogram", ""]
    for k, v in sorted(out["classification_histogram"].items()):
        L.append(f"- `{k}`: {v}")
    L += ["", "## Per-theorem", "",
          "| theorem | baseline | candidate | proxy credit | SX4 class | credit |",
          "|---|---|---|---|---|---|"]
    for r in out["records"]:
        L.append(f"| `{r['theorem']}` | {'✓' if r['baseline_finished'] else '—'} | "
                 f"{'✓' if r['candidate_finished'] else '—'} | "
                 f"{'✓' if r.get('proxy_runner_credited') else '—'} | "
                 f"**{r['classification']}** | {'✅' if r['credit'] else '—'} |")
    L += ["", "## Subsumption evidence (PRODUCTION_SUBSUMED)", ""]
    for r in out["records"]:
        if r["classification"] == "PRODUCTION_SUBSUMED":
            ta = r["production_trace_analysis"]
            L.append(f"- `{r['theorem']}`: equivalent_sequence_observed="
                     f"**{ta['equivalent_sequence_observed']}** "
                     f"(conf={ta['trace_confidence']}, path={ta['baseline_winning_path']})")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
