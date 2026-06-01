#!/usr/bin/env python3
"""RC4D Part 6 — external additive composition evaluator (ordered attribution).

candidate_solved = literal_RC2_solved OR (a gated RC4D component tactic closes the goal
single-shot). The candidate ⊇ RC2, so regressions are structurally impossible. For each
gate-firing theorem we collect the ORDERED RC4D emissions [RC4A..RC4B..RC4C_residue] and
find the FIRST emitted tactic that closes the goal — winning_component is that tactic's
component. Ordering is the composition de-duplication: a Multiset disjoint theorem that both
RC4B (`disjoint_left`) and RC4C_residue (`disjoint_right`) close is credited to RC4B.

Reuse-first: per-(theorem,tactic) solve outcomes already executed by the three components'
live probes are reused from their caches; only theorems with an un-cached emitted tactic are
probed live (one Dojo, every emitted tactic tried, RC4A/B/C harness).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
NOFIRE_SETS = ("negative_controls", "namespace_negative_controls", "canonical_smoke")


def _p(*a):
    return os.path.join(_REPO, *a)


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
    reuse = G.build_reuse_map()

    # collect entries + emissions
    entries = {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            rec = entries.setdefault(fn, {"full_name": fn, "file_path": e.get("file_path"),
                                          "namespace": e.get("namespace"),
                                          "goal_text": e.get("goal_text"), "sets": []})
            rec["sets"].append(setname)
            fires, em = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), fn)
            rec["gate_fires"] = fires
            rec["emissions"] = em
            rec["tactics"] = G.tactics_of(em)

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
        actions_emitted = [x["action"] for x in e["emissions"]] if gate else []
        tac_solved = {}
        err = None
        if gate:
            # which emitted tactics are not yet known from reuse cache?
            missing = [t for t in e["tactics"] if (fn, t) not in reuse]
            if not missing:
                for t in e["tactics"]:
                    tac_solved[t] = bool(reuse.get((fn, t)))
            else:
                if fn in ckpt:
                    wres = ckpt[fn]
                else:
                    with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                        wout = tf.name
                    cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
                           sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
                           "--case-json", json.dumps({"full_name": fn, "file_path": e["file_path"],
                                                      "tactics": e["tactics"]}),
                           "--open-timeout", str(args.open_timeout),
                           "--timeout-per-tactic", str(args.timeout_per_tactic)]
                    print(f"[rc4d-cand] {fn}: {e['tactics']} ...", flush=True)
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
                ran = {x["tactic"]: x for x in wres.get("ran", [])}
                for t in e["tactics"]:
                    if (fn, t) in reuse:
                        tac_solved[t] = bool(reuse[(fn, t)])
                    else:
                        tac_solved[t] = bool(ran.get(t, {}).get("solved"))

        # ordered attribution: first EMITTED tactic (in emission order) that solves
        win_tac = win_comp = win_action = None
        win_lemma = None
        if gate:
            for x in e["emissions"]:
                if tac_solved.get(x["tactic"]):
                    win_tac, win_comp, win_action = x["tactic"], x["component"], x["action"]
                    win_lemma = x["lemma_or_defs"]
                    break
        cand_solved = win_tac is not None
        cand_fin = rc2_fin or cand_solved
        new_win = (not rc2_fin) and gate and cand_solved
        results.append({
            "full_name": fn, "file_path": e["file_path"], "sets": e["sets"],
            "namespace": G.namespace_of(e["namespace"], fn),
            "rc2_finished": rc2_fin, "rc2_status": r2.get("rc2_status"),
            "gate_fired": gate, "actions_emitted": actions_emitted,
            "candidate_tactics": e["tactics"],
            "components_firing": G.components_firing(e["emissions"]) if gate else [],
            "tactic_solved": tac_solved,
            "candidate_finished": cand_fin,
            "winning_tactic": win_tac, "winning_component": win_comp,
            "winning_action": win_action, "winning_lemma": win_lemma,
            "new_win_over_rc2": new_win, "off_gate": off_gate,
            "regression": False, "error": err,
        })

    def _by(pred, key):
        return dict(Counter(r[key] for r in results if pred(r)))

    new_wins = [r for r in results if r["new_win_over_rc2"]]
    delta_by_comp = Counter(r["winning_component"] for r in new_wins)
    # overlap eliminated = theorems where RC4C_residue also fired but RC4B (earlier) won
    overlap_elim = [r["full_name"] for r in new_wins
                    if r["winning_component"] == "RC4B" and "RC4C_residue" in r["components_firing"]]
    metrics = {
        "num_theorems": len(results),
        "rc2_solved": sum(1 for r in results if r["rc2_finished"]),
        "candidate_solved": sum(1 for r in results if r["candidate_finished"]),
        "raw_delta": len(new_wins),
        "delta_by_component": dict(delta_by_comp),
        "overlap_eliminated_count": len(overlap_elim),
        "overlap_eliminated_targets": overlap_elim,
        "regressions": 0,
        "gate_emissions": sum(1 for r in results if r["gate_fired"]),
        "off_gate_emissions": sum(1 for r in results if r["off_gate"]),
        "emitted_and_solved": sum(1 for r in results
                                  if r["gate_fired"] and r["candidate_finished"] and not r["rc2_finished"]),
        "emitted_and_failed": sum(1 for r in results
                                  if r["gate_fired"] and not r["candidate_finished"]),
        "new_wins_by_namespace": _by(lambda r: r["new_win_over_rc2"], "namespace"),
    }
    win_targets = {c: [r["full_name"] for r in new_wins if r["winning_component"] == c]
                   for c in ("RC4A", "RC4B", "RC4C_residue")}
    out = {"generated_by": "scripts/rc4d_run_additive_eval.py",
           "evaluation_mode": "external_additive_ordered",
           "metrics": metrics, "new_win_targets_by_component": win_targets,
           "new_win_targets": [r["full_name"] for r in new_wins], "results": results}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4D additive composition evaluation (ordered)", "",
          f"- raw delta over RC2: **{metrics['raw_delta']}**",
          f"- delta by component: {metrics['delta_by_component']}",
          f"- overlap eliminated (RC4C→RC4B): {metrics['overlap_eliminated_count']} {overlap_elim}",
          f"- off-gate emissions: {metrics['off_gate_emissions']} | regressions: 0",
          f"- emitted&solved {metrics['emitted_and_solved']} / emitted&failed {metrics['emitted_and_failed']}", "",
          "| theorem | sets | ns | rc2 | comps_fire | win_comp | win_tac | new |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (not x["new_win_over_rc2"], x["namespace"], x["full_name"])):
        if not r["gate_fired"] and not r["new_win_over_rc2"]:
            continue
        md.append(f"| `{r['full_name']}` | {','.join(r['sets'])} | {r['namespace']} | "
                  f"{'S' if r['rc2_finished'] else 'F'} | {','.join(r['components_firing'])} | "
                  f"{r['winning_component'] or ''} | `{r['winning_tactic'] or ''}` | "
                  f"{r['new_win_over_rc2']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-cand] {metrics}")
    print(f"[rc4d-cand] win_targets={win_targets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--manifest", "--validation-manifest", dest="manifest")
    ap.add_argument("--policy")
    ap.add_argument("--literal-rc2")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/candidate_runs")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=300)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
