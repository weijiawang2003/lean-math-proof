#!/usr/bin/env python3
"""RC4D Part 10 — schema-native composition wrapper construction + smoke.

Builds `rc4d_candidate_wrapper.json` = a functional copy of the frozen RC2 wrapper with the
RC4A/RC4B/RC4C_residue tactics added the narrowest schema-native way: prepended to
`priority_templates["any"]` (the SET_ITE_SIMP / RC2 / RC4B precedent) and gated by
`theorem_name_tactic_gates`. KEY LESSON FROM RC4C: deploy every depth-2 lemma RC4B-style as
BOTH the bare `simp [L]` enabling action AND the `simp [L] <;> aesop` combinator, because the
best-first search applies `<;>` differently than a single-shot transition — RC4B reproduced
10/11 only because it also added the bare simp. RC4A is added as its exact winning
`simp [defs]` tactics. Then smokes known wins + fresh + negatives + canonical floors through
the real eval_rollout_all search and compares to literal RC2. NOT a release validation; if the
search misses many additive wins → SCHEMA_INTEGRATION_BLOCKER.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
RC2 = "project/evolve/experiments/rc2_release/rc2_production_wrapper.json"
WRAPPER_OUT = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_candidate_wrapper.json"
# Smoke scope: all 23 credited-win-bearing sets (reproduction) + de-dup overlap controls +
# negatives (no-regression) + canonical floors. composition_fresh_holdout is excluded: those
# are dominated by emitted-and-failed heavy `<;> aesop` goals whose FULL best-first search is
# hours of wall-clock and adds no reproduction signal (the additive evaluator already covers
# them). Wrapper reproduction of the credited wins is the decision-critical measurement here.
SMOKE_SETS = ("rc4a_known_wins", "rc4b_known_wins", "rc4c_residue_known_wins",
              "component_overlap_controls",
              "negative_controls", "namespace_negative_controls", "canonical_smoke")

# RC4A: exact winning def-unfold tactics, gated by NAME PREFIXES.
# RC2's gate semantics is `full_name.startswith(prefix)` (evolve/strategy_wrapper.py:757),
# so prefixes MUST include the namespace (e.g. "Set.monotoneOn", not "monotoneOn").
RC4A_GATES = {
    "simp [Finset.disjUnion]": ["Finset.mem_disjUnion", "Finset.disjUnion",
                                "Finset.coe_disjUnion", "Finset.disjiUnion"],
    "simp [Monotone, MonotoneOn]": ["Set.monotoneOn", "Set.not_monotoneOn"],
    "simp [Antitone, AntitoneOn]": ["Set.antitoneOn", "Set.not_antitoneOn"],
    "simp [StrictMono, StrictMonoOn]": ["Set.strictMonoOn"],
    "simp [StrictAnti, StrictAntiOn]": ["Set.strictAntiOn"],
}
# RC4B: namespace-parametric disjoint bridge (verbatim from the validated RC4B wrapper).
RC4B_GATES = {
    "simp [Set.disjoint_left]": ["Set.disjoint", "Set._root_.Disjoint"],
    "simp [Set.disjoint_left] <;> aesop": ["Set.disjoint", "Set._root_.Disjoint"],
    "simp [Multiset.disjoint_left]": ["Multiset.disjoint", "Multiset.singleton_disjoint",
                                      "Multiset.zero_disjoint"],
    "simp [Multiset.disjoint_left] <;> aesop": ["Multiset.disjoint", "Multiset.singleton_disjoint",
                                                "Multiset.zero_disjoint"],
}
# RC4C_residue: depth-2 residue lemmas, deployed RC4B-style (bare + combinator).
RC4C_GATES = {
    "simp [Multiset.disjoint_right]": ["Multiset.disjoint", "Multiset.singleton_disjoint",
                                       "Multiset.zero_disjoint"],
    "simp [Multiset.disjoint_right] <;> aesop": ["Multiset.disjoint", "Multiset.singleton_disjoint",
                                                 "Multiset.zero_disjoint"],
    "simp [Set.subset_pair_iff_eq]": ["subset_pair", "Set.Nonempty.subset_pair"],
    "simp [Set.subset_pair_iff_eq] <;> aesop": ["subset_pair", "Set.Nonempty.subset_pair"],
    "simp [List.forall_iff_forall_mem]": ["List.Forall", "List.forall"],
    "simp [List.forall_iff_forall_mem] <;> aesop": ["List.Forall", "List.forall"],
}
ALL_GATES = {**RC4A_GATES, **RC4B_GATES, **RC4C_GATES}
# component label per tactic, for reporting which component reproduced
COMP_OF = ({t: "RC4A" for t in RC4A_GATES} | {t: "RC4B" for t in RC4B_GATES}
           | {t: "RC4C_residue" for t in RC4C_GATES})


def _p(*a):
    return os.path.join(_REPO, *a)


def build_wrapper():
    w = copy.deepcopy(json.load(open(_p(RC2))))
    pri = w.setdefault("priority_templates", {})
    anyl = list(pri.get("any", []))
    added = list(ALL_GATES.keys())
    for t in reversed(added):
        if t not in anyl:
            anyl.insert(0, t)
    pri["any"] = anyl
    gates = w.setdefault("theorem_name_tactic_gates", {})
    for t, subs in ALL_GATES.items():
        gates[t] = subs
    w["_rc4d_candidate_metadata"] = {
        "base": "RC2", "base_wrapper": RC2,
        "added_component": "RC4D composition = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue "
                           "(off-by-default, NOT released)",
        "added_tactics_priority_any_prepended": added,
        "theorem_name_tactic_gates_added": ALL_GATES,
        "rc4c_residue_deployed_rc4b_style": "bare simp [L] + simp [L] <;> aesop (RC4C lesson)",
        "promotion_allowed": False, "rc1_wrapper_untouched": True,
        "rc2_wrapper_untouched": True, "ns24_router_untouched": True,
        "note": "Name-substring gates approximate the external namespace+token gate; the "
                "external additive evaluator remains the authority.",
    }
    os.makedirs(os.path.dirname(_p(WRAPPER_OUT)), exist_ok=True)
    json.dump(w, open(_p(WRAPPER_OUT), "w"), ensure_ascii=False, indent=2)
    return WRAPPER_OUT


def worker(args):
    cases = json.loads(args.cases_json)
    out = {}
    try:
        sys.path.insert(0, _REPO)
        from core_types import TheoremConfig
        import eval_rollout_all as E
        fields = {f.name for f in dataclasses.fields(TheoremConfig)}
        tcs, names = [], {}
        for c in cases:
            if not c.get("file_path"):
                out[c["full_name"]] = {"status": "path_error"}
                continue
            tcs.append(TheoremConfig(**{k: v for k, v in
                                        {"file_path": c["file_path"], "full_name": c["full_name"]}.items()
                                        if k in fields}))
            names[c["full_name"]] = c
        if tcs:
            nm = "rc4d_schema_smoke"
            _get, _list = E.get_theorems, E.list_theorem_sets
            E.list_theorem_sets = lambda: (list(_list()) + [nm]) if nm not in _list() else list(_list())
            E.get_theorems = lambda n: tcs if n == nm else _get(n)
            os.makedirs(args.out_dir, exist_ok=True)
            sys.argv = ["eval_rollout_all.py", "--theorem-set", nm,
                        "--policy-type", args.policy_type, "--route-config", args.route_config,
                        "--strategy-config", args.wrapper, "--top-k", str(args.top_k),
                        "--max-steps", str(args.max_steps), "--out-dir", args.out_dir]
            try:
                E.main()
            except SystemExit:
                pass
            mfiles = sorted(glob.glob(os.path.join(args.out_dir, "eval-*", "metrics.json")),
                            key=os.path.getmtime)
            if mfiles:
                m = json.load(open(mfiles[-1]))
                seen = {r["full_name"]: r for r in m.get("per_theorem", [])}
                for fn in names:
                    r = seen.get(fn)
                    if r is None:
                        out[fn] = {"status": "path_error"}
                    elif not r.get("available"):
                        out[fn] = {"status": "open_flake"}
                    else:
                        out[fn] = {"status": "solved" if r.get("finished") else "failed",
                                   "winning_tactic": r.get("winning_tactic"),
                                   "num_steps": r.get("num_steps")}
            else:
                for fn in names:
                    out[fn] = {"status": "trace_insufficient"}
    except Exception as e:
        for c in cases:
            out.setdefault(c["full_name"], {"status": "trace_insufficient", "error": str(e)[:120]})
    json.dump(out, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def driver(args):
    wrapper = build_wrapper()
    manifest = json.load(open(_p(args.manifest)))
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.literal_rc2)))["results"]} \
        if os.path.exists(_p(args.literal_rc2)) else {}
    cand = {}
    if args.candidate_results and os.path.exists(_p(args.candidate_results)):
        cand = {r["full_name"]: r for r in json.load(open(_p(args.candidate_results)))["results"]}

    cases, seen, membership = [], set(), {}
    for setname in SMOKE_SETS:
        rel = manifest["set_files"].get(setname)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            membership.setdefault(fn, []).append(setname)
            if fn not in seen and e.get("file_path"):
                seen.add(fn)
                cases.append({"full_name": fn, "file_path": e["file_path"]})

    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    chunks = [cases[i:i + args.chunk_size] for i in range(0, len(cases), args.chunk_size)]
    results = {}
    for ci, chunk in enumerate(chunks):
        wout = os.path.join(out_dir, f"chunk_{ci}.json")
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--cases-json", json.dumps(chunk), "--out-dir", os.path.join(out_dir, "runs"),
               "--wrapper", wrapper, "--route-config", args.route_config,
               "--policy-type", args.policy_type, "--top-k", str(args.top_k),
               "--max-steps", str(args.max_steps)]
        print(f"[rc4d-smoke] chunk {ci+1}/{len(chunks)} ({len(chunk)}) ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            results.update(json.load(open(wout)))
        except Exception:
            for c in chunk:
                results[c["full_name"]] = {"status": "trace_insufficient"}

    known_sets = ("rc4a_known_wins", "rc4b_known_wins", "rc4c_residue_known_wins")
    recs = []
    for fn in seen:
        st = results.get(fn, {}).get("status")
        wt = results.get(fn, {}).get("winning_tactic")
        r2 = rc2.get(fn, {})
        cr = cand.get(fn, {})
        rc2_fin = bool(r2.get("rc2_finished"))
        wrap_fin = st == "solved"
        recs.append({"full_name": fn, "sets": membership[fn], "rc2_finished": rc2_fin,
                     "additive_new_win": bool(cr.get("new_win_over_rc2")),
                     "additive_component": cr.get("winning_component"),
                     "wrapper_status": st, "wrapper_finished": wrap_fin, "winning_tactic": wt,
                     "wrapper_component": COMP_OF.get(wt),
                     "regression": rc2_fin and not wrap_fin,
                     "new_win": (not rc2_fin) and wrap_fin,
                     "is_known_win": any(s in known_sets for s in membership[fn])})
    known = [r for r in recs if r["is_known_win"]]
    known_solved = sum(1 for r in known if r["wrapper_finished"])
    regressions = [r["full_name"] for r in recs if r["regression"]]
    wrapper_new = [r["full_name"] for r in recs if r["new_win"]]
    # additive wins the wrapper missed
    missed = [r["full_name"] for r in recs if r["additive_new_win"] and not r["wrapper_finished"]]
    from collections import Counter
    wrap_by_comp = Counter(r["wrapper_component"] for r in recs if r["new_win"] and r["wrapper_component"])

    blocker = len(missed) > max(2, 0.5 * len([r for r in recs if r["additive_new_win"]]))
    summary = {
        "generated_by": "scripts/rc4d_schema_wrapper_smoke.py", "wrapper": wrapper,
        "num_smoke": len(recs), "known_wins_total": len(known),
        "known_wins_solved_by_wrapper": known_solved,
        "wrapper_new_wins": len(wrapper_new), "wrapper_new_wins_by_component": dict(wrap_by_comp),
        "additive_wins_total": sum(1 for r in recs if r["additive_new_win"]),
        "additive_wins_missed_by_wrapper": missed,
        "regressions": regressions, "no_regression": not regressions,
        "verdict": "SCHEMA_INTEGRATION_BLOCKER" if blocker else "SCHEMA_REPRODUCES",
        "note": "Smoke only; name-substring gates approximate the external gate. RC4C_residue "
                "deployed RC4B-style (bare simp + combinator).",
        "results": recs,
    }
    json.dump(summary, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4D schema-native wrapper smoke", "",
          f"- wrapper: `{wrapper}`",
          f"- known wins solved by wrapper: **{known_solved}/{len(known)}**",
          f"- wrapper new wins: {len(wrapper_new)} by_comp={dict(wrap_by_comp)}",
          f"- additive wins missed by wrapper: {len(missed)} {missed}",
          f"- regressions: {len(regressions)} {regressions}",
          f"- verdict: **{summary['verdict']}**", "",
          "| theorem | sets | rc2 | add_comp | wrapper | win_tac |", "|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: (x["sets"], x["full_name"])):
        md.append(f"| `{r['full_name']}` | {','.join(r['sets'])} | "
                  f"{'S' if r['rc2_finished'] else 'F'} | {r['additive_component'] or ''} | "
                  f"{'S' if r['wrapper_finished'] else 'F'} | `{r['winning_tactic'] or ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-smoke] known_solved={known_solved}/{len(known)} new={len(wrapper_new)} "
          f"missed={len(missed)} regressions={regressions} verdict={summary['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--manifest", "--validation-manifest", dest="manifest")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/schema_smoke")
    ap.add_argument("--wrapper")
    ap.add_argument("--candidate-results",
                    default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/additive_candidate_results.json")
    ap.add_argument("--literal-rc2",
                    default="project/evolve/experiments/rc4_candidates/composition_rc4d/out/literal_rc2_results.json")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--policy-type", default="hybrid_evolved")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--hard-timeout", type=int, default=1800)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))
    driver(args)


if __name__ == "__main__":
    main()
