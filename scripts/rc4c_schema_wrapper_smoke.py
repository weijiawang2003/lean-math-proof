#!/usr/bin/env python3
"""RC4C Part 10 — optional schema-native wrapper smoke.

Builds `rc4c_candidate_wrapper.json` as a functional copy of the frozen RC2 wrapper with
the six allowlisted depth-2 `simp [L] <;> aesop` tactics added in the narrowest schema-
native way: prepended to `priority_templates["any"]` (the SET_ITE_SIMP / RC2 / RC4B
precedent) and gated by `theorem_name_tactic_gates` so each only emits on its lemma's
namespace/name shape. Then runs a small smoke over known_wins_all + the negative controls
through the real eval_rollout_all harness and compares to literal RC2:
  - known wins still solve through the wrapper,
  - no regressions on negative controls,
  - emitted gate behaviour matches the external evaluator as closely as possible.
NOT a release validation.
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
WRAPPER_OUT = "project/evolve/experiments/rc4_candidates/d2_simp_aesop/rc4c_candidate_wrapper.json"
SMOKE_SETS = ("known_wins_all", "negative_controls", "namespace_negative_controls")

# narrow name gates (substring match, RC2 theorem_name_tactic_gates semantics)
GATES = {
    "simp [Set.disjoint_left] <;> aesop": ["Set.disjoint", "Set.Disjoint"],
    "simp [Multiset.disjoint_left] <;> aesop": ["Multiset.disjoint", "Multiset.Disjoint"],
    "simp [Multiset.disjoint_right] <;> aesop": ["Multiset.disjoint", "Multiset.Disjoint"],
    "simp [Set.subset_pair_iff_eq] <;> aesop": ["subset_pair"],
    "simp [Finset.biUnion_subset] <;> aesop": ["Finset.biUnion", "Finset.bunion"],
    "simp [List.forall_iff_forall_mem] <;> aesop": ["List.Forall", "List.forall"],
}


def _p(*a):
    return os.path.join(_REPO, *a)


def build_wrapper():
    w = copy.deepcopy(json.load(open(_p(RC2))))
    pri = w.setdefault("priority_templates", {})
    anyl = list(pri.get("any", []))
    bridge = list(GATES.keys())
    for t in reversed(bridge):
        if t not in anyl:
            anyl.insert(0, t)
    pri["any"] = anyl
    gates = w.setdefault("theorem_name_tactic_gates", {})
    for t, subs in GATES.items():
        gates[t] = subs
    w["_rc4c_candidate_metadata"] = {
        "base": "RC2", "base_wrapper": RC2,
        "added_component": "d2_simp_aesop allowlist (RC4C candidate, off-by-default, NOT released)",
        "added_tactics_priority_any_prepended": bridge,
        "theorem_name_tactic_gates_added": GATES,
        "promotion_allowed": False,
        "rc1_wrapper_untouched": True, "rc2_wrapper_untouched": True,
        "ns24_router_untouched": True,
        "note": "Smoke-only schema-native copy. Name-substring gates approximate the external "
                "namespace+token gate; the external additive evaluator remains the authority. "
                "Two of these tactics (Set/Multiset.disjoint_left) overlap RC4B.",
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
            nm = "rc4c_schema_smoke"
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
        print(f"[rc4c-smoke] chunk {ci+1}/{len(chunks)} ({len(chunk)}) ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            results.update(json.load(open(wout)))
        except Exception:
            for c in chunk:
                results[c["full_name"]] = {"status": "trace_insufficient"}

    recs = []
    for fn in seen:
        st = results.get(fn, {}).get("status")
        r2 = rc2.get(fn, {})
        rc2_fin = bool(r2.get("rc2_finished"))
        wrap_fin = st == "solved"
        recs.append({"full_name": fn, "sets": membership[fn], "rc2_finished": rc2_fin,
                     "wrapper_status": st, "wrapper_finished": wrap_fin,
                     "winning_tactic": results.get(fn, {}).get("winning_tactic"),
                     "regression": rc2_fin and not wrap_fin,
                     "new_win": (not rc2_fin) and wrap_fin})
    known = [r for r in recs if "known_wins_all" in r["sets"]]
    known_solved = sum(1 for r in known if r["wrapper_finished"])
    regressions = [r["full_name"] for r in recs if r["regression"]]
    new_wins = [r["full_name"] for r in recs if r["new_win"]]
    summary = {
        "generated_by": "scripts/rc4c_schema_wrapper_smoke.py",
        "wrapper": wrapper, "num_smoke": len(recs),
        "known_wins_total": len(known), "known_wins_solved_by_wrapper": known_solved,
        "regressions": regressions, "new_wins": new_wins, "no_regression": not regressions,
        "note": "Name-substring gates approximate the external namespace+token gate; wrapper search "
                "may differ slightly from single-shot probe. Smoke only.",
        "results": recs,
    }
    json.dump(summary, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4C schema-native wrapper smoke", "",
          f"- wrapper: `{wrapper}`",
          f"- known wins solved by wrapper: **{known_solved}/{len(known)}**",
          f"- regressions on negative controls: {len(regressions)} {regressions}",
          f"- new wins observed: {len(new_wins)} {new_wins}", "",
          "| theorem | sets | rc2 | wrapper | win_tac |", "|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: (x["sets"], x["full_name"])):
        md.append(f"| `{r['full_name']}` | {','.join(r['sets'])} | "
                  f"{'S' if r['rc2_finished'] else 'F'} | "
                  f"{'S' if r['wrapper_finished'] else 'F'} | `{r['winning_tactic'] or ''}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4c-smoke] known_solved={known_solved}/{len(known)} regressions={regressions} "
          f"new_wins={new_wins}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--cases-json")
    ap.add_argument("--manifest")
    ap.add_argument("--policy")  # accepted for CLI symmetry; wrapper built from RC2
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--out-dir", default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/schema_smoke")
    ap.add_argument("--wrapper")  # used by the worker subprocess (built by the driver)
    ap.add_argument("--literal-rc2",
                    default="project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/literal_rc2_results.json")
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
