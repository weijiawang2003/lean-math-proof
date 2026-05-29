"""SX1 Stage 1 — collect multi-step symbolic-assisted cases from mined traces.

There is no live Lean in this stage: the authoritative record of what a
symbolic action did to a proof state is the already-mined trace corpus under
project/evolve/eval_runs/. Each trace record carries the pre-state (`state_pp`),
the tactic, its origin, `result_kind` (LeanError / TacticState / ProofFinished),
and the state hashes that link a tactic's output to the next step's inputs.

We scan every symbolic-capable wrapper run (WX3 oracle `*_wx3ind_*` for
Multiset, AX1-symbolic `*_ax1sym_*` for Option/List) and reconstruct, per
theorem, the fate of each symbolic-action firing:

  * single_shot   — symbolic action => ProofFinished in one step.
  * advanced      — symbolic action => TacticState (goals remain, no error).
  * multistep     — symbolic action advanced AND a later step closed the proof
                    from the resulting state (a genuine depth>=2 symbolic-
                    assisted proof captured by the search).

For each multistep case we record the theorem, namespace, arc, the first
symbolic action id + instantiated tactic, the resulting-state hash, the closing
tactic(s) and their origin, and cross-referenced raw/NS9 win status from the
sibling `*_raw_*` / `*_ns9_*` runs. This is the empirical inventory the SX1
sequence schema and evaluator are built against.

Outputs:
  project/data/sx1_multistep_symbolic_cases_meta.json
  project/evolve/reports/sx1_multistep_symbolic_cases_inventory.md
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_META = ROOT / "project/data/sx1_multistep_symbolic_cases_meta.json"
OUT_MD = ROOT / "project/evolve/reports/sx1_multistep_symbolic_cases_inventory.md"

SYM = "wrapper_symbolic_action"

# Symbolic-capable wrapper run families -> (arc tag, sibling raw/ns9 prefixes).
# We glob each run dir's traces, then find the matching raw/ns9 sibling by
# swapping the wrapper tag for the run set suffix.
WRAPPER_GLOBS = [
    ("AX4", "ax4_wx3ind"),
    ("AX3", "ax3_wx3ind"),
    ("AX2", "ax2_ax1sym"),
]
RAW_TAG = {"ax4_wx3ind": "ax4_raw", "ax3_wx3ind": "ax3_raw",
           "ax2_ax1sym": "ax2_raw"}
NS9_TAG = {"ax4_wx3ind": "ax4_ns9", "ax3_wx3ind": "ax3_ns9",
           "ax2_ax1sym": "ax2_ns9"}


def _namespace_of(full_name: str) -> str:
    return full_name.split(".", 1)[0] if full_name else ""


def load_episodes(run_glob: str):
    """{full_name: [records...]} from all traces under matching run dirs."""
    eps = defaultdict(list)
    for tf in glob.glob(run_glob):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            fn = o.get("full_name")
            if fn:
                eps[fn].append(o)
    return eps


def won_set(run_glob: str) -> set:
    won = set()
    for tf in glob.glob(run_glob):
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("proof_finished") and o.get("full_name"):
                won.add(o["full_name"])
    return won


def main() -> None:
    cases = []            # multistep symbolic-assisted
    advanced_only = []    # symbolic advanced but search never closed from it
    single_shot = []      # symbolic closed in one step
    totals = defaultdict(int)

    for arc, wtag in WRAPPER_GLOBS:
        run_dirs = sorted({Path(p).parents[1]
                           for p in glob.glob(
                               f"project/evolve/eval_runs/{wtag}_*/eval-*/traces.jsonl")})
        for rd in run_dirs:
            set_name = rd.name[len(wtag) + 1:]
            wglob = f"{rd}/eval-*/traces.jsonl"
            raw_glob = f"project/evolve/eval_runs/{RAW_TAG[wtag]}_{set_name}/eval-*/traces.jsonl"
            ns9_glob = f"project/evolve/eval_runs/{NS9_TAG[wtag]}_{set_name}/eval-*/traces.jsonl"
            raw_won = won_set(raw_glob)
            ns9_won = won_set(ns9_glob)

            eps = load_episodes(wglob)
            for fn, recs in eps.items():
                ns = _namespace_of(fn)
                finished_from = defaultdict(list)  # hash_before -> closers
                for r in recs:
                    if r.get("result_kind") == "ProofFinished":
                        finished_from[r.get("state_hash_before")].append(r)
                for r in recs:
                    if r.get("tactic_origin") != SYM:
                        continue
                    totals["symbolic_firings"] += 1
                    rk = r.get("result_kind")
                    init_state = next(
                        (x.get("state_pp") for x in recs
                         if x.get("step") == 1 and x.get("state_pp")), "")
                    base = {
                        "theorem": fn, "namespace": ns, "source_arc": arc,
                        "eval_set": set_name,
                        "first_symbolic_action": r.get("tactic_family_source"),
                        "first_tactic": r.get("tactic"),
                        "first_step": r.get("step"),
                        "initial_state_snippet": (init_state or "")[:240],
                        "initial_state_hash": r.get("state_hash_before"),
                        "raw_or_ns9_solved": (fn in raw_won) or (fn in ns9_won),
                        "raw_solved": fn in raw_won,
                        "ns9_solved": fn in ns9_won,
                    }
                    if rk == "ProofFinished":
                        totals["single_shot"] += 1
                        single_shot.append(base)
                    elif rk == "TacticState":
                        sh = r.get("state_hash_after")
                        closers = finished_from.get(sh, [])
                        if closers:
                            totals["multistep"] += 1
                            c = closers[0]
                            cc = {**base,
                                  "resulting_state_hash": sh,
                                  "closing_tactic": c.get("tactic"),
                                  "closing_tactic_origin": c.get("tactic_origin"),
                                  "closing_step": c.get("step"),
                                  "n_closers": len(closers)}
                            # de-dupe by (theorem, first_tactic, closing_tactic)
                            cases.append(cc)
                        else:
                            totals["advanced_no_close"] += 1
                            advanced_only.append(base)

    # de-dupe multistep cases by (theorem, first_tactic, closing_tactic)
    seen = set()
    uniq_cases = []
    for c in cases:
        k = (c["theorem"], c["first_tactic"], c["closing_tactic"])
        if k in seen:
            continue
        seen.add(k)
        uniq_cases.append(c)

    by_arc = defaultdict(int)
    for c in uniq_cases:
        by_arc[c["source_arc"]] += 1

    meta = {
        "description": "SX1 Stage 1 — multi-step symbolic-assisted cases mined "
                       "from existing oracle/symbolic trace corpus (offline).",
        "totals": dict(totals),
        "unique_multistep_cases": len(uniq_cases),
        "multistep_by_arc": dict(by_arc),
        "n_advanced_no_close_states": len(advanced_only),
        "n_single_shot": len(single_shot),
        "key_finding": (
            "The existing NS9/WX3 best-first search already explores follow-up "
            "tactics from advanced symbolic states; every multistep case below "
            "was already CLOSED by that search. Sequence mode does not add these "
            "as new wins — it makes the two-step shape explicit/learnable."),
        "multistep_cases": uniq_cases,
        "advanced_no_close_sample": advanced_only[:25],
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ---- inventory markdown -----------------------------------------
    lines = []
    lines.append("# SX1 multi-step symbolic-assisted case inventory\n")
    lines.append("Mined offline from the existing oracle/symbolic trace corpus "
                 "(`*_wx3ind_*` Multiset, `*_ax1sym_*` Option/List). No live "
                 "Lean was run in this stage.\n")
    lines.append("## Totals\n")
    lines.append(f"- symbolic-action firings scanned: **{totals['symbolic_firings']}**")
    lines.append(f"- single-shot closes (symbolic => ProofFinished): "
                 f"**{totals['single_shot']}**")
    lines.append(f"- advanced (symbolic => TacticState): "
                 f"**{totals['advanced_no_close'] + totals['multistep']}** "
                 f"(of which {totals['multistep']} were later closed by the search)")
    lines.append(f"- **unique multistep symbolic-assisted cases: "
                 f"{len(uniq_cases)}** ({dict(by_arc)})\n")
    lines.append("## Key finding\n")
    lines.append("> " + meta["key_finding"] + "\n")
    lines.append("## Multistep cases\n")
    lines.append("| theorem | ns | arc | first symbolic tactic | closing tactic "
                 "| closer origin | raw/ns9 solved? |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in uniq_cases:
        lines.append(
            f"| `{c['theorem']}` | {c['namespace']} | {c['source_arc']} "
            f"| `{c['first_tactic']}` | `{c['closing_tactic']}` "
            f"| {c['closing_tactic_origin']} | {c['raw_or_ns9_solved']} |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("Every multistep close above was produced by the existing "
                 "best-first search (the closing tactic appears in the trace). "
                 "The symbolic first action *advances* the state; the search "
                 "then finds the closer (base-model `aesop`, a re-applied "
                 "symbolic action, etc.). The SX1 sequence schema turns this "
                 "implicit two-step behaviour into an explicit, namespace-gated, "
                 "depth-2 object — its value is **selectivity / learnability**, "
                 "not new raw search reach (see decision gate in the report).")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"firings={totals['symbolic_firings']} single_shot={totals['single_shot']} "
          f"advanced={totals['advanced_no_close']+totals['multistep']} "
          f"multistep={len(uniq_cases)} by_arc={dict(by_arc)}")


if __name__ == "__main__":
    main()
