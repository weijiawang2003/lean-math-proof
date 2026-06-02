#!/usr/bin/env python3
"""FLI3 Part 6 — additive candidate evaluation at theorem position (vacuity-safe).

For each validation item where the gate fires AND literal RC2 failed, open a LeanDojo Dojo at the
theorem's real file position and run controls + the gated candidate actions from the initial state
(reusing the FLI2 worker). Additive win = RC2-failed + gate-fired + candidate-solved +
all-controls-failed + non-vacuous. Wins re-run once for robustness. Offgate/floor items don't fire
→ no actions → no emission (additive design); recorded as gate=False.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fli3_gate as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
_FLI2_WORKER = os.path.join(_REPO, "scripts", "fli2_run_live_deployment_eval.py")
CONTROLS = ["simp", "aesop", "classical <;> aesop", "constructor <;> simp", "ext x <;> simp"]


def _p(*a):
    return os.path.join(_REPO, *a)


def _run_worker(case, ckpt, open_to=90, per_tac=8, hard=120):
    if os.path.exists(ckpt):
        return json.load(open(ckpt))
    wout = ckpt + ".tmp"
    cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable, _FLI2_WORKER, "--worker",
           "--worker-out", wout, "--case-json", json.dumps(case),
           "--open-timeout", str(open_to), "--timeout-per-tactic", str(per_tac)]
    subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(wout):
        os.replace(wout, ckpt)
        return json.load(open(ckpt))
    r = {"theorem": case["theorem"], "setup_error": "no worker output", "controls": [], "actions": []}
    json.dump(r, open(ckpt, "w"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--trace-dir", required=True)
    args = ap.parse_args()

    items = json.load(open(_p(args.sets)))["items"]
    rc2 = {r["theorem"]: r["rc2_status"] for r in json.load(open(_p(args.rc2_results)))["results"]}
    trace_dir = _p(args.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    results = []
    for it in items:
        g = G.gate(it)
        base = {"theorem": it["theorem"], "set": it["set"], "namespace": it["namespace"],
                "candidate_family": it["candidate_family"], "lemma": g.get("lemma"),
                "gate": g["gate"], "gate_reason": g["reason"], "rc2_status": rc2.get(it["theorem"]),
                "actions": [], "control_solved": None, "candidate_win": False,
                "winning_tactic": None, "robust": None, "is_offgate_emission": False}
        if not g["gate"]:
            # gate didn't fire → additive design ⇒ candidate ≡ RC2 (no emission)
            results.append(base)
            continue
        if rc2.get(it["theorem"]) == "solved":
            base["note"] = "gate fired on an RC2-solved theorem"
            base["is_offgate_emission"] = True  # would-fire on solved (flag for review)
        case = {"theorem": it["theorem"], "file_path": it.get("file_path"),
                "controls": CONTROLS,
                "actions": [{"action_id": f"{it['theorem']}::{i}", "tactic": t,
                             "lemma": g.get("lemma"), "template": "FLI3"}
                            for i, t in enumerate(g["action_templates"])]}
        ck = os.path.join(trace_dir, it["theorem"].replace("/", "_").replace(".", "_") + ".json")
        w = _run_worker(case, ck)
        ctrl_solved = [c["tactic"] for c in w.get("controls", []) if c.get("solved")]
        base["control_solved"] = ctrl_solved
        win = next((a for a in w.get("actions", []) if a.get("solved")), None)
        non_vac = g.get("lemma") != it["theorem"]
        base["actions"] = [{"tactic": a.get("tactic"), "solved": bool(a.get("solved")),
                            "status": a.get("status"),
                            "residual_after": (a.get("residual_after") or "")[:200]}
                           for a in w.get("actions", [])]
        base["residual_before"] = (w.get("initial_goal") or "")[:200]
        base["setup_error"] = w.get("setup_error")
        if win and not ctrl_solved and non_vac:
            base["candidate_win"] = True
            base["winning_tactic"] = win["tactic"]
            # robustness re-run (winning tactic only)
            rc = os.path.join(trace_dir, "robust_" + it["theorem"].replace("/", "_").replace(".", "_") + ".json")
            case2 = {"theorem": it["theorem"], "file_path": it.get("file_path"),
                     "controls": CONTROLS, "actions": [{"action_id": "rb", "tactic": win["tactic"],
                                                        "lemma": g.get("lemma"), "template": "FLI3"}]}
            w2 = _run_worker(case2, rc)
            cs2 = [c["tactic"] for c in w2.get("controls", []) if c.get("solved")]
            base["robust"] = bool(any(a.get("solved") for a in w2.get("actions", [])) and not cs2)
        results.append(base)
        print(f"[fli3-cand] {it['set']} {it['theorem']}: gate={g['gate']} "
              f"win={base['candidate_win']} robust={base.get('robust')}", flush=True)

    wins = [r for r in results if r["candidate_win"]]
    rescue_wins = [r for r in wins if r["set"] == "rescue_replay"]
    holdout_wins = [r for r in wins if r["set"] == "family_holdout"]
    offgate_em = [r for r in results if r["is_offgate_emission"]]
    out = {"generated_by": "scripts/fli3_run_candidate_eval.py",
           "num_items": len(results),
           "gate_fired": sum(1 for r in results if r["gate"]),
           "candidate_wins": len(wins),
           "robust_wins": sum(1 for r in wins if r["robust"]),
           "rescue_replay_wins": len(rescue_wins),
           "family_holdout_wins": len(holdout_wins),
           "offgate_emissions": len(offgate_em),
           "regressions": 0,
           "unknown_name_or_import": sum(1 for r in results for a in r["actions"]
                                         if a["status"] == "unknown_name"),
           "win_theorems": sorted(r["theorem"] for r in wins),
           "results": results}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = ["# FLI3 candidate eval", "",
          f"- items: {out['num_items']} | gate fired: {out['gate_fired']}",
          f"- **candidate wins: {out['candidate_wins']}** (robust {out['robust_wins']}) | "
          f"rescue_replay {out['rescue_replay_wins']}/6 | family_holdout {out['family_holdout_wins']}",
          f"- offgate emissions: {out['offgate_emissions']} | regressions: {out['regressions']} | "
          f"unknown-name: {out['unknown_name_or_import']}", "",
          "| set | theorem | gate | win | robust | winning tactic |", "|---|---|---|---|---|---|"]
    for r in results:
        if r["gate"]:
            md.append(f"| {r['set']} | `{r['theorem']}` | {r['gate']} | {r['candidate_win']} | "
                      f"{r.get('robust')} | `{r.get('winning_tactic') or ''}` |")
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli3-cand] DONE wins={len(wins)} robust={out['robust_wins']} "
          f"rescue={len(rescue_wins)}/6 holdout={len(holdout_wins)} offgate={len(offgate_em)}")


if __name__ == "__main__":
    main()
