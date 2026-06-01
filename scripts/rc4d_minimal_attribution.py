#!/usr/bin/env python3
"""RC4D Part 7 — minimal attribution of each new composition win.

For every new_win_over_rc2 theorem we re-probe (one Dojo): the bare controls
(simp / simp_all / aesop / classical <;> aesop) and the component-specific controls for the
winning component — RC4A's `simp [defs]`; RC4B's `simp [<NS>.disjoint_left]` (+`<;> aesop`);
RC4C_residue's `simp [L]` (+`<;> aesop`). A win is credited to its component only if:

  TRUE_RC4A_WIN          RC2 failed, bare controls fail, `simp [defs]` closes.
  TRUE_RC4B_WIN          RC2 failed, bare fail, a `simp [<NS>.disjoint_left]`(+aesop) closes
                         (and it is the earliest-component winner).
  TRUE_RC4C_RESIDUE_WIN  RC2 failed, bare fail, NO RC4A/RC4B mechanism applies, and a residue
                         `simp [L] <;> aesop` closes with `simp [L]` alone NOT closing
                         (genuine depth-2; else it would be depth-1/RC4A territory).
  OVERLAP_DUPLICATE      win is attributed to RC4B but an RC4C_residue tactic also closes it
                         (the de-dup case — RC4B keeps credit, residue does not).
  BASELINE_DUPLICATE     a bare control already closes it.
  RC2_ALREADY_SOLVED     stale baseline.
  SCHEMA_UNSTABLE / SOURCE_SPECIFIC / NEEDS_REVIEW   as in RC4C.

Only TRUE_RC4A_WIN / TRUE_RC4B_WIN / TRUE_RC4C_RESIDUE_WIN count toward credited delta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
BARE = ["simp", "simp_all", "aesop", "classical <;> aesop"]


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _probe(fn, fp, tactics, args):
    import subprocess
    seen, ordered = set(), []
    for t in tactics:
        if t not in seen:
            seen.add(t); ordered.append(t)
    with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        wout = tf.name
    cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
           sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
           "--case-json", json.dumps({"full_name": fn, "file_path": fp, "tactics": ordered}),
           "--open-timeout", str(args.open_timeout),
           "--timeout-per-tactic", str(args.timeout_per_tactic)]
    print(f"[rc4d-attrib] {fn}: {ordered} ...", flush=True)
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
    return wres


def driver(args):
    cand = json.load(open(_p(args.candidate_results)))
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    new_wins = [r for r in cand["results"] if r["new_win_over_rc2"]]

    records = []
    for r in new_wins:
        fn = r["full_name"]
        comp = r["winning_component"]
        win_tac = r["winning_tactic"]
        win_lemma = r["winning_lemma"]
        # build control tactic list: bare + the winning tactic + (for residue/bridge) the
        # bare simp[L] depth-1 to test genuineness, + lemma-direct.
        tactics = list(BARE)
        tactics.append(win_tac)
        L = None
        if isinstance(win_lemma, list):
            L = ", ".join(win_lemma)  # RC4A defs
            tactics.append(f"simp [{L}]")
        elif win_lemma:
            L = win_lemma
            tactics += [f"simp [{L}]", f"simp [{L}] <;> aesop",
                        f"exact {L}", f"simpa using {L}"]
        fp = r.get("file_path") or rc2.get(fn, {}).get("file_path")
        wres = _probe(fn, fp, tactics, args)
        ran = {x["tactic"]: x for x in wres.get("ran", [])}

        def solved(t):
            return bool(ran.get(t, {}).get("solved"))

        bare_solved = [c for c in BARE if solved(c)]
        rc2_fin = bool(rc2.get(fn, {}).get("rc2_finished"))
        simp_l = solved(f"simp [{L}]") if L else False
        win_solved = solved(win_tac)
        genuine_d2 = bool(win_solved and ("<;> aesop" in win_tac) and not simp_l)
        also_residue = "RC4C_residue" in r.get("components_firing", []) and comp == "RC4B"

        if rc2_fin:
            cls = "RC2_ALREADY_SOLVED"
        elif bare_solved:
            cls = "BASELINE_DUPLICATE"
        elif comp == "RC4A":
            cls = "TRUE_RC4A_WIN" if win_solved else "NEEDS_REVIEW"
        elif comp == "RC4B":
            cls = "TRUE_RC4B_WIN" if win_solved else "NEEDS_REVIEW"
            # note overlap with residue but RC4B keeps credit (de-dup)
        elif comp == "RC4C_residue":
            if not win_solved:
                cls = "NEEDS_REVIEW"
            elif "<;> aesop" not in win_tac:
                # bare simp[L] won -> depth-1, RC4A-ish, not a genuine d2 residue win
                cls = "SIMP_ONLY_DUPLICATE"
            elif genuine_d2:
                cls = "TRUE_RC4C_RESIDUE_WIN"
            else:
                cls = "SIMP_ONLY_DUPLICATE"
        elif wres.get("setup_error"):
            cls = "NEEDS_REVIEW"
        else:
            cls = "SOURCE_SPECIFIC"

        credited = cls in ("TRUE_RC4A_WIN", "TRUE_RC4B_WIN", "TRUE_RC4C_RESIDUE_WIN")
        records.append({
            "full_name": fn, "namespace": r["namespace"], "sets": r["sets"],
            "winning_component": comp, "winning_tactic": win_tac, "winning_lemma": win_lemma,
            "bare_controls_solved": bare_solved, "simp_l_alone_solved": simp_l,
            "winning_tactic_solved": win_solved, "genuine_depth2": genuine_d2,
            "residue_also_fires_but_rc4b_won": also_residue,
            "classification": cls, "credited": credited,
            "fresh": any(s == "composition_fresh_holdout" for s in r["sets"]),
            "setup_error": wres.get("setup_error"),
        })

    hist = Counter(r["classification"] for r in records)
    credited = [r for r in records if r["credited"]]
    by_comp = Counter(r["winning_component"] for r in credited)
    fresh = [r["full_name"] for r in credited if r["fresh"]]
    repro = [r["full_name"] for r in credited if not r["fresh"]]
    overlap_removed = [r["full_name"] for r in records if r["residue_also_fires_but_rc4b_won"]]

    out = {
        "generated_by": "scripts/rc4d_minimal_attribution.py",
        "num_new_wins": len(new_wins),
        "classification_histogram": dict(hist),
        "credited_delta_total": len(credited),
        "credited_delta_by_component": dict(by_comp),
        "credited_fresh": fresh, "credited_reproductions": repro,
        "overlap_removed_rc4c_to_rc4b": overlap_removed,
        "credited_targets": [r["full_name"] for r in credited],
        "records": records,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4D minimal attribution", "",
          f"- new wins examined: {len(new_wins)}",
          f"- classifications: {dict(hist)}",
          f"- **credited delta total: {len(credited)}**",
          f"- credited by component: {dict(by_comp)}",
          f"- credited fresh: {len(fresh)} {fresh}",
          f"- credited reproductions: {len(repro)}",
          f"- overlap removed (RC4C_residue→RC4B): {len(overlap_removed)} {overlap_removed}", "",
          "| theorem | ns | win_comp | win_tac | bare | simp[L] | genuine_d2 | class |",
          "|---|---|---|---|---|---|---|---|"]
    for r in records:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['winning_component']} | "
                  f"`{r['winning_tactic']}` | {r['bare_controls_solved']} | "
                  f"{r['simp_l_alone_solved']} | {r['genuine_depth2']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-attrib] {dict(hist)}")
    print(f"[rc4d-attrib] credited={len(credited)} by_comp={dict(by_comp)} "
          f"fresh={fresh} overlap_removed={overlap_removed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--candidate-results")
    ap.add_argument("--literal-rc2")
    ap.add_argument("--policy")
    ap.add_argument("--manifest")
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
