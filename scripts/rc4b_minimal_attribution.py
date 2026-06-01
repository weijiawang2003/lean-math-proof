#!/usr/bin/env python3
"""RC4B Part 7 — minimal attribution of each new candidate win.

For every new_win_over_rc2 theorem, run bare controls (simp / simp_all / aesop /
classical <;> aesop), the lemma-direct controls (`exact <NS>.disjoint_left` /
`simpa using <NS>.disjoint_left`), and the gated bridge tactics, then classify:

  TRUE_DISJOINT_LEFT_BRIDGE_WIN  RC2 failed, bare controls fail, a policy bridge tactic
                                 closes it, and its lemma is in the matched namespace.
  BASELINE_DUPLICATE             a bare control already closes it.
  RC2_ALREADY_SOLVED             stale baseline (RC2 actually solved).
  WRONG_NAMESPACE                the solving bridge lemma is from the wrong namespace.
  SOURCE_SPECIFIC                closes only with more than the gated bridge tactic.
  NEEDS_REVIEW                   setup error / ambiguous.
Only TRUE_DISJOINT_LEFT_BRIDGE_WIN counts.
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
import rc4b_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
BARE_CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]
POLICY_TACTICS = {  # all four policy bridge tactics, by namespace
    "Set": ["simp [Set.disjoint_left]", "simp [Set.disjoint_left] <;> aesop"],
    "Multiset": ["simp [Multiset.disjoint_left]", "simp [Multiset.disjoint_left] <;> aesop"],
}


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
    policy_set = {t for a in policy["actions"] for t in [a["tactic"]]}
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    new_wins = [r for r in cand["results"] if r["new_win_over_rc2"]]

    records = []
    for r in new_wins:
        fn = r["full_name"]
        ns = r["namespace"]
        lemma = r.get("bridge_lemma")
        # controls + lemma-direct + all policy bridge tactics for this namespace
        lemma_direct = []
        if lemma:
            lemma_direct = [f"exact {lemma}", f"simpa using {lemma}"]
        tactics = BARE_CONTROLS + lemma_direct + POLICY_TACTICS.get(ns, [])
        # uniquify preserving order
        seen, ordered = set(), []
        for t in tactics:
            if t not in seen:
                seen.add(t); ordered.append(t)
        fp = r.get("file_path") or rc2.get(fn, {}).get("file_path")
        with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": fp, "tactics": ordered}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[rc4b-attrib] {fn}: controls + bridge ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = {x["tactic"]: x for x in wres.get("ran", [])}
        bare_solved = [c for c in BARE_CONTROLS if ran.get(c, {}).get("solved")]
        lemma_direct_solved = [c for c in lemma_direct if ran.get(c, {}).get("solved")]
        bridge_solved = [t for t in POLICY_TACTICS.get(ns, []) if ran.get(t, {}).get("solved")]
        rc2_fin = bool(rc2.get(fn, {}).get("rc2_finished"))
        # namespace check: the solving bridge tactic's lemma must match the namespace
        wrong_ns = bool(bridge_solved) and any(
            (".disjoint_left" in t and not t.split("[")[1].split(".")[0].strip() == ns)
            for t in bridge_solved)

        if rc2_fin:
            cls = "RC2_ALREADY_SOLVED"
        elif bare_solved:
            cls = "BASELINE_DUPLICATE"
        elif wrong_ns:
            cls = "WRONG_NAMESPACE"
        elif bridge_solved:
            cls = "TRUE_DISJOINT_LEFT_BRIDGE_WIN"
        elif wres.get("setup_error"):
            cls = "NEEDS_REVIEW"
        else:
            cls = "SOURCE_SPECIFIC"

        records.append({
            "full_name": fn, "namespace": ns, "sets": r["sets"],
            "bridge_lemma": lemma, "candidate_winning_tactic": r.get("winning_tactic"),
            "bare_controls_solved": bare_solved,
            "lemma_direct_solved": lemma_direct_solved,
            "bridge_tactics_solved": bridge_solved,
            "in_policy": bool(bridge_solved) and all(t in policy_set or "<;>" in t for t in bridge_solved),
            "classification": cls, "credited": cls == "TRUE_DISJOINT_LEFT_BRIDGE_WIN",
            "fresh": any(s.startswith("fresh_holdout") for s in r["sets"]),
            "known_reproduction": "known_wins" in r["sets"],
            "setup_error": wres.get("setup_error"),
        })

    from collections import Counter
    hist = Counter(r["classification"] for r in records)
    credited = [r for r in records if r["credited"]]
    cred_names = [r["full_name"] for r in credited]
    split = {
        "Set_true_wins": sorted(r["full_name"] for r in credited if r["namespace"] == "Set"),
        "Multiset_true_wins": sorted(r["full_name"] for r in credited if r["namespace"] == "Multiset"),
        "fresh_holdout_true_wins": sorted(r["full_name"] for r in credited if r["fresh"]),
        "known_reproduction_true_wins": sorted(r["full_name"] for r in credited if r["known_reproduction"]),
    }
    out = {"generated_by": "scripts/rc4b_minimal_attribution.py",
           "num_new_wins": len(new_wins), "classification_histogram": dict(hist),
           "num_true_bridge_wins": len(credited),
           "true_bridge_win_targets": cred_names, "split": split, "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4B minimal attribution", "",
          f"- new wins examined: {len(new_wins)}",
          f"- classifications: {dict(hist)}",
          f"- **TRUE_DISJOINT_LEFT_BRIDGE_WIN: {len(credited)}** {cred_names}",
          f"- Set true wins: {len(split['Set_true_wins'])} | Multiset true wins: {len(split['Multiset_true_wins'])}",
          f"- fresh-holdout true wins: {len(split['fresh_holdout_true_wins'])} | "
          f"known-reproduction true wins: {len(split['known_reproduction_true_wins'])}", "",
          "| theorem | ns | bare_solved | bridge_solved | class |",
          "|---|---|---|---|---|"]
    for r in records:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['bare_controls_solved']} | "
                  f"{r['bridge_tactics_solved']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4b-attrib] {dict(hist)}; TRUE_DISJOINT_LEFT_BRIDGE_WIN={len(credited)}; split={split}")


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
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=400)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
