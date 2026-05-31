#!/usr/bin/env python3
"""RC4A Part 7 — minimal attribution of each new candidate win.

For every new_win_over_rc2 theorem, run the bare controls (simp / simp_all / aesop /
classical <;> aesop) and the candidate `simp [defs]` live, and classify:
  TRUE_DEF_UNFOLD_SIMP_WIN  RC2 failed, controls fail, gated `simp [defs]` closes
  BASELINE_DUPLICATE        a simpler bare control already closes
  RC2_ALREADY_SOLVED        stale baseline (rc2 actually solved)
  SOURCE_SPECIFIC           closing needs more than the gated def unfold
  HETEROGENEOUS_MECHANISM   winning defs not in the allowlist
  NEEDS_REVIEW
Only TRUE_DEF_UNFOLD_SIMP_WIN counts.
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
import rc4a_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def driver(args):
    cand = json.load(open(_p(args.candidate_results)))
    policy = G.load_policy(args.policy)
    allow = set(policy["validated_def_allowlist"])
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    new_wins = [r for r in cand["results"] if r["new_win_over_rc2"]]

    records = []
    for r in new_wins:
        fn = r["full_name"]
        cand_tac = r["candidate_tactic"]
        tactics = CONTROLS + [cand_tac]
        with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": r.get("file_path") or
                                          rc2.get(fn, {}).get("file_path"), "tactics": tactics}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        # file_path may be absent in candidate record; pull from rc2 baseline
        print(f"[rc4a-attrib] {fn}: controls + candidate ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = {x["tactic"]: x for x in wres.get("ran", [])}
        control_solved = [c for c in CONTROLS if ran.get(c, {}).get("solved")]
        cand_solved = bool(ran.get(cand_tac, {}).get("solved"))
        defs_in_allow = all(d in allow for d in r.get("matched_defs", []))
        rc2_fin = bool(rc2.get(fn, {}).get("rc2_finished"))

        if rc2_fin:
            cls = "RC2_ALREADY_SOLVED"
        elif not defs_in_allow:
            cls = "HETEROGENEOUS_MECHANISM"
        elif control_solved:
            cls = "BASELINE_DUPLICATE"
        elif cand_solved:
            cls = "TRUE_DEF_UNFOLD_SIMP_WIN"
        elif wres.get("setup_error"):
            cls = "NEEDS_REVIEW"
        else:
            cls = "SOURCE_SPECIFIC"

        records.append({
            "full_name": fn, "candidate_tactic": cand_tac,
            "matched_defs": r.get("matched_defs", []),
            "controls_solved": control_solved, "candidate_solved": cand_solved,
            "classification": cls, "credited": cls == "TRUE_DEF_UNFOLD_SIMP_WIN",
            "setup_error": wres.get("setup_error"),
        })

    from collections import Counter
    hist = Counter(r["classification"] for r in records)
    credited = [r["full_name"] for r in records if r["credited"]]
    out = {"generated_by": "scripts/rc4a_minimal_attribution.py",
           "num_new_wins": len(new_wins), "classification_histogram": dict(hist),
           "num_true_def_unfold_wins": len(credited),
           "true_def_unfold_win_targets": credited, "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4A minimal attribution", "",
          f"- new wins examined: {len(new_wins)}",
          f"- classifications: {dict(hist)}",
          f"- **TRUE_DEF_UNFOLD_SIMP_WIN: {len(credited)}** {credited}", "",
          "| theorem | candidate | controls_solved | cand_solved | class |",
          "|---|---|---|---|---|"]
    for r in records:
        md.append(f"| `{r['full_name']}` | `{r['candidate_tactic']}` | "
                  f"{r['controls_solved']} | {r['candidate_solved']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4a-attrib] {dict(hist)}; TRUE_DEF_UNFOLD_SIMP_WIN={len(credited)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--candidate-results")
    ap.add_argument("--literal-rc2")
    ap.add_argument("--policy")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=300)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
