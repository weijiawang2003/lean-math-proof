#!/usr/bin/env python3
"""RC4C Part 7 — minimal attribution of each new candidate win.

For every new_win_over_rc2 theorem, run bare controls (simp / simp_all / aesop /
classical <;> aesop), and for each matched allowlist lemma L: the lemma-direct controls
(`exact L` / `simpa using L`), the depth-1 `simp [L]`, and the depth-2 `simp [L] <;> aesop`.
The depth-2 mechanism is GENUINE for L only if `simp [L] <;> aesop` closes the goal AND
`simp [L]` alone does NOT (otherwise it is depth-1 / RC4A territory). Classify:

  TRUE_D2_SIMP_AESOP_WIN          RC2 failed, bare fail, a NON-overlap lemma's genuine
                                  depth-2 closes it (simp[L] alone fails).  [pure RC4C credit]
  TRUE_D2_SIMP_AESOP_OVERLAP_RC4B genuine depth-2 win but only via a disjoint_left lemma
                                  (Set/Multiset.disjoint_left) — already RC4B. [composition only]
  SIMP_ONLY_DUPLICATE             `simp [L]` alone already closes it (not true depth-2).
  BASELINE_DUPLICATE              a bare control already closes it.
  RC2_ALREADY_SOLVED              stale baseline.
  SOURCE_SPECIFIC                 closes only with more than a single allowlisted d2 tactic.
  HETEROGENEOUS_MECHANISM         solving lemma is outside the allowlist.
  NEEDS_REVIEW                    setup error / ambiguous.

Only TRUE_D2_SIMP_AESOP_WIN counts as pure RC4C credit; overlap wins count for the
eventual RC4 composition but not for pure nonoverlap RC4C.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4c_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
BARE_CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]
_RC4B_LEMMAS = ("Set.disjoint_left", "Multiset.disjoint_left")


def _p(*a):
    return os.path.join(_REPO, *a)


def _lemmas_from_tactics(tactics):
    out = []
    for t in tactics:
        m = re.search(r"simp \[([^\]]+)\]", t)
        if m:
            L = m.group(1).split(",")[0].strip()
            if L not in out:
                out.append(L)
    return out


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def driver(args):
    cand = json.load(open(_p(args.candidate_results)))
    policy = G.load_policy(args.policy)
    allow = set(policy["allowlist_lemmas"])
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]}
    new_wins = [r for r in cand["results"] if r["new_win_over_rc2"]]

    records = []
    for r in new_wins:
        fn = r["full_name"]
        ns = r["namespace"]
        lemmas = _lemmas_from_tactics(r["candidate_tactics"])
        tactics = list(BARE_CONTROLS)
        for L in lemmas:
            tactics += [f"simp [{L}]", f"simp [{L}] <;> aesop",
                        f"exact {L}", f"simpa using {L}"]
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
        print(f"[rc4c-attrib] {fn}: controls + d2 over {lemmas} ...", flush=True)
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
        ran = {x["tactic"]: x for x in wres.get("ran", [])}

        def solved(t):
            return bool(ran.get(t, {}).get("solved"))

        bare_solved = [c for c in BARE_CONTROLS if solved(c)]
        per_lemma = {}
        for L in lemmas:
            d1 = solved(f"simp [{L}]")
            d2 = solved(f"simp [{L}] <;> aesop")
            ld = [c for c in (f"exact {L}", f"simpa using {L}") if solved(c)]
            per_lemma[L] = {"simp_only": d1, "d2": d2, "lemma_direct_solved": ld,
                            "genuine_d2": bool(d2 and not d1),
                            "overlap": L in _RC4B_LEMMAS, "in_allowlist": L in allow}
        rc2_fin = bool(rc2.get(fn, {}).get("rc2_finished"))
        nonoverlap_genuine = [L for L, d in per_lemma.items()
                              if d["genuine_d2"] and not d["overlap"] and d["in_allowlist"]]
        overlap_genuine = [L for L, d in per_lemma.items()
                           if d["genuine_d2"] and d["overlap"]]
        any_simp_only = [L for L, d in per_lemma.items() if d["simp_only"]]
        any_d2 = [L for L, d in per_lemma.items() if d["d2"]]
        out_of_allowlist_win = [L for L, d in per_lemma.items() if d["d2"] and not d["in_allowlist"]]

        if rc2_fin:
            cls = "RC2_ALREADY_SOLVED"
        elif bare_solved:
            cls = "BASELINE_DUPLICATE"
        elif nonoverlap_genuine:
            cls = "TRUE_D2_SIMP_AESOP_WIN"
        elif overlap_genuine:
            cls = "TRUE_D2_SIMP_AESOP_OVERLAP_RC4B"
        elif any_simp_only:
            cls = "SIMP_ONLY_DUPLICATE"
        elif out_of_allowlist_win:
            cls = "HETEROGENEOUS_MECHANISM"
        elif wres.get("setup_error"):
            cls = "NEEDS_REVIEW"
        elif any_d2:
            # d2 solved but with simp[L] alone also solving for the same lemma already
            # handled; this branch = d2 solved but not genuine and not simp-only -> review
            cls = "SOURCE_SPECIFIC"
        else:
            cls = "SOURCE_SPECIFIC"

        records.append({
            "full_name": fn, "namespace": ns, "sets": r["sets"],
            "candidate_winning_tactic": r.get("winning_tactic"),
            "candidate_winning_lemma": r.get("winning_lemma"),
            "lemmas_tested": lemmas, "per_lemma": per_lemma,
            "bare_controls_solved": bare_solved,
            "nonoverlap_genuine_d2_lemmas": nonoverlap_genuine,
            "overlap_genuine_d2_lemmas": overlap_genuine,
            "simp_only_lemmas": any_simp_only,
            "classification": cls,
            "credited_pure_rc4c": cls == "TRUE_D2_SIMP_AESOP_WIN",
            "credited_composition": cls in ("TRUE_D2_SIMP_AESOP_WIN", "TRUE_D2_SIMP_AESOP_OVERLAP_RC4B"),
            "fresh": any(s.startswith("fresh_holdout") for s in r["sets"]),
            "known_reproduction": any(s.startswith("known_wins") for s in r["sets"]),
            "setup_error": wres.get("setup_error"),
        })

    hist = Counter(r["classification"] for r in records)
    pure = [r for r in records if r["credited_pure_rc4c"]]
    overlap = [r for r in records if r["classification"] == "TRUE_D2_SIMP_AESOP_OVERLAP_RC4B"]
    simp_only = [r for r in records if r["classification"] == "SIMP_ONLY_DUPLICATE"]
    pure_names = [r["full_name"] for r in pure]
    split = {
        "pure_rc4c_true_wins": pure_names,
        "pure_rc4c_by_namespace": dict(Counter(r["namespace"] for r in pure)),
        "pure_rc4c_fresh": sorted(r["full_name"] for r in pure if r["fresh"]),
        "pure_rc4c_reproduction": sorted(r["full_name"] for r in pure if r["known_reproduction"]),
        "overlap_rc4b_true_wins": sorted(r["full_name"] for r in overlap),
        "simp_only_duplicates": sorted(r["full_name"] for r in simp_only),
    }
    out = {"generated_by": "scripts/rc4c_minimal_attribution.py",
           "num_new_wins": len(new_wins), "classification_histogram": dict(hist),
           "num_pure_rc4c_true_wins": len(pure),
           "num_overlap_rc4b_true_wins": len(overlap),
           "num_simp_only_duplicates": len(simp_only),
           "num_composition_credited": sum(1 for r in records if r["credited_composition"]),
           "pure_rc4c_true_win_targets": pure_names, "split": split, "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4C minimal attribution", "",
          f"- new wins examined: {len(new_wins)}",
          f"- classifications: {dict(hist)}",
          f"- **TRUE_D2_SIMP_AESOP_WIN (pure RC4C): {len(pure)}** {pure_names}",
          f"- TRUE_D2_SIMP_AESOP_OVERLAP_RC4B: {len(overlap)} {split['overlap_rc4b_true_wins']}",
          f"- SIMP_ONLY_DUPLICATE: {len(simp_only)} {split['simp_only_duplicates']}",
          f"- pure RC4C by namespace: {split['pure_rc4c_by_namespace']} | "
          f"fresh: {len(split['pure_rc4c_fresh'])} repro: {len(split['pure_rc4c_reproduction'])}", "",
          "| theorem | ns | bare | genuine_d2(non) | genuine_d2(ovl) | simp_only | class |",
          "|---|---|---|---|---|---|---|"]
    for r in records:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['bare_controls_solved']} | "
                  f"{r['nonoverlap_genuine_d2_lemmas']} | {r['overlap_genuine_d2_lemmas']} | "
                  f"{r['simp_only_lemmas']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4c-attrib] {dict(hist)}")
    print(f"[rc4c-attrib] pure_rc4c={len(pure)} {pure_names}; overlap_rc4b={len(overlap)}; "
          f"simp_only={len(simp_only)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--candidate-results")
    ap.add_argument("--literal-rc2")
    ap.add_argument("--policy")
    ap.add_argument("--rc4b-attribution")
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
