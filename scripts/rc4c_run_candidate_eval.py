#!/usr/bin/env python3
"""RC4C Part 6 — external additive candidate evaluator (dual-mode all / nonoverlap).

candidate_solved = literal_RC2_solved OR (gate fires AND at least one gated
`simp [L] <;> aesop` closes the goal single-shot from the initial state). Additive form
makes regressions structurally impossible (candidate ⊇ RC2). The gated probe runs live
ONLY where the RC4C_all gate fires (one Dojo / theorem, every matched tactic tried). Each
new win is tagged overlap_family (RC4B if its winning lemma is Set/Multiset.disjoint_left)
so the pure-nonoverlap delta is reported separately from the all-mode delta.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4c_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
NOFIRE_SETS = ("negative_controls", "namespace_negative_controls", "canonical_smoke")
_RC4B_LEMMAS = ("Set.disjoint_left", "Multiset.disjoint_left")


def _p(*a):
    return os.path.join(_REPO, *a)


def _lemma_of_tac(tac, matched):
    for a in matched:
        if a["tactic"] == tac:
            return a["lemma"], a.get("overlap_family", "none")
    return None, "none"


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def driver(args):
    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}

    entries = {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            rec = entries.setdefault(fn, {"full_name": fn, "file_path": e.get("file_path"),
                                          "namespace": e.get("namespace"),
                                          "goal_text": e.get("goal_text"), "sets": []})
            rec["sets"].append(setname)
            fa, ma, ta, na, la = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), fn, mode="all")
            rec["gate_fires"] = fa
            rec["matched"] = ma
            rec["candidate_tactics"] = ta
            rec["candidate_actions"] = na

    os.makedirs(_p(args.out_dir), exist_ok=True)
    ckpt_path = _p(args.out_dir, "probe_checkpoint.json")
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}

    results = []
    for fn, e in entries.items():
        r2 = rc2.get(fn, {})
        rc2_fin = bool(r2.get("rc2_finished"))
        gate = e["gate_fires"]
        must_not_fire = any(s in NOFIRE_SETS for s in e["sets"])
        off_gate = bool(gate and must_not_fire)
        actions_emitted = e["candidate_actions"] if gate else []
        win_tac = win_lemma = win_overlap = None
        nonoverlap_win_tac = None
        probe_outcomes = {}
        cand_solved = False
        nonoverlap_solved = False
        err = None
        if gate:
            if fn in ckpt:
                wres = ckpt[fn]
            else:
                with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                    wout = tf.name
                cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
                       sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
                       "--case-json", json.dumps({"full_name": fn, "file_path": e["file_path"],
                                                  "tactics": e["candidate_tactics"]}),
                       "--open-timeout", str(args.open_timeout),
                       "--timeout-per-tactic", str(args.timeout_per_tactic)]
                print(f"[rc4c-cand] {fn}: gate {e['candidate_tactics']} ...", flush=True)
                subprocess.run(cmd, capture_output=True, text=True)
                try:
                    wres = json.load(open(wout))
                except (ValueError, OSError):
                    wres = {"ran": [], "setup_error": "worker_output_unreadable"}
                finally:
                    try:
                        os.unlink(wout)
                    except OSError:
                        pass
                ckpt[fn] = wres
                json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
            if wres.get("setup_error"):
                err = wres["setup_error"]
            for ran in wres.get("ran", []):
                probe_outcomes[ran["tactic"]] = ran.get("outcome")
                if ran.get("solved"):
                    lemma, ofam = _lemma_of_tac(ran["tactic"], e["matched"])
                    if win_tac is None:
                        win_tac, win_lemma, win_overlap = ran["tactic"], lemma, ofam
                        cand_solved = True
                    if ofam != "RC4B" and nonoverlap_win_tac is None:
                        nonoverlap_win_tac = ran["tactic"]
                        nonoverlap_solved = True
        cand_fin = rc2_fin or cand_solved
        new_win = (not rc2_fin) and gate and cand_solved
        nonoverlap_new_win = (not rc2_fin) and gate and nonoverlap_solved
        overlaps_rc4b = bool(new_win and win_overlap == "RC4B" and not nonoverlap_solved)
        results.append({
            "full_name": fn, "file_path": e["file_path"], "sets": e["sets"],
            "namespace": G.namespace_of(e["namespace"], fn),
            "rc2_finished": rc2_fin, "rc2_status": r2.get("rc2_status"),
            "gate_fired": gate, "actions_emitted": actions_emitted,
            "candidate_tactics": e["candidate_tactics"],
            "candidate_probe_outcomes": probe_outcomes,
            "candidate_finished": cand_fin, "winning_tactic": win_tac,
            "winning_lemma": win_lemma, "winning_overlap_family": win_overlap,
            "nonoverlap_winning_tactic": nonoverlap_win_tac,
            "new_win_over_rc2": new_win, "nonoverlap_new_win": nonoverlap_new_win,
            "overlaps_rc4b": overlaps_rc4b, "off_gate": off_gate, "error": err,
            "trace_path": r2.get("trace_path"),
        })

    def _ns_split(pred):
        from collections import Counter
        return dict(Counter(r["namespace"] for r in results if pred(r)))

    metrics = {
        "num_theorems": len(results),
        "rc2_solved": sum(1 for r in results if r["rc2_finished"]),
        "candidate_solved_all": sum(1 for r in results if r["candidate_finished"]),
        "raw_delta_all": sum(1 for r in results if r["new_win_over_rc2"]),
        "raw_delta_nonoverlap": sum(1 for r in results if r["nonoverlap_new_win"]),
        "overlap_rc4b_wins": sum(1 for r in results if r["overlaps_rc4b"]),
        "regressions": 0,
        "gate_emissions": sum(1 for r in results if r["gate_fired"]),
        "off_gate_emissions": sum(1 for r in results if r["off_gate"]),
        "emitted_and_solved": sum(1 for r in results
                                  if r["gate_fired"] and r["candidate_finished"] and not r["rc2_finished"]),
        "emitted_and_failed": sum(1 for r in results
                                  if r["gate_fired"] and not r["candidate_finished"]),
        "new_wins_by_namespace_all": _ns_split(lambda r: r["new_win_over_rc2"]),
        "new_wins_by_namespace_nonoverlap": _ns_split(lambda r: r["nonoverlap_new_win"]),
    }
    all_wins = [r["full_name"] for r in results if r["new_win_over_rc2"]]
    non_wins = [r["full_name"] for r in results if r["nonoverlap_new_win"]]
    ovl_wins = [r["full_name"] for r in results if r["overlaps_rc4b"]]
    out = {"generated_by": "scripts/rc4c_run_candidate_eval.py",
           "evaluation_mode": "external_additive_dual",
           "metrics": metrics,
           "new_win_targets_all": all_wins,
           "new_win_targets_nonoverlap": non_wins,
           "overlap_rc4b_win_targets": ovl_wins,
           "results": results}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4C candidate evaluation (additive, dual-mode)", "",
          f"- metrics: {metrics}",
          f"- new wins (all): **{metrics['raw_delta_all']}** {all_wins}",
          f"- new wins (nonoverlap): **{metrics['raw_delta_nonoverlap']}** {non_wins}",
          f"- overlap-only (RC4B) wins: {metrics['overlap_rc4b_wins']} {ovl_wins}", "",
          "| theorem | sets | ns | rc2 | gate | win_tac | lemma | overlap | new(all) | new(non) | off_gate |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (not x["new_win_over_rc2"], x["namespace"], x["full_name"])):
        md.append(f"| `{r['full_name']}` | {','.join(r['sets'])} | {r['namespace']} | "
                  f"{'S' if r['rc2_finished'] else 'F'} | {r['gate_fired']} | "
                  f"`{r['winning_tactic'] or ''}` | `{r['winning_lemma'] or ''}` | "
                  f"{r['winning_overlap_family'] or ''} | {r['new_win_over_rc2']} | "
                  f"{r['nonoverlap_new_win']} | {r['off_gate']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4c-cand] {metrics}")
    print(f"[rc4c-cand] all_wins={all_wins}")
    print(f"[rc4c-cand] nonoverlap_wins={non_wins}; overlap_rc4b={ovl_wins}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--manifest")
    ap.add_argument("--policy")
    ap.add_argument("--literal-rc2")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/candidate_runs")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=300)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
