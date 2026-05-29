"""MX1 Stage 5 — LIVE strict minimal relabeling of new wins beyond production.

For every theorem won by the extended symbolic wrapper (E) but NOT by the
production wrapper (B) — read from project/data/mx1_live_mining_probe_meta.json —
re-open a live Dojo session and try a strict ordered battery from the INITIAL
state. The first tactic that closes the proof determines the minimal label, so a
symbolic action is only credited when no simpler tactic closes it.

Battery (in order):
  assumption, rfl, decide, simp, simp_all, aesop,
  ext x <;> simp, ext x <;> simp_all,
  cases {v} <;> simp, cases {v} <;> simp_all,
  induction {v} using Multiset.induction_on <;> simp[_all]   (Multiset only)
where {v} is read from the live state (vars_of_type), falling back to the
wrapper's winning tactic.

Classification:
  over_attributed_raw         — a plain tactic (assumption..aesop) closes alone.
  clean_single_shot_symbolic  — a single ext/cases/induction one-liner closes.
  sequence_needed             — only a symbolic-first + follow-up closes (rare;
                                detected when the single symbolic action leaves
                                a TacticState that a battery follow-up then closes).
  flaky / dropped             — nothing in the battery reproduces the close.

Outputs:
  project/data/mx1_minimal_symbolic_frontier_labels.json
  project/data/mx1_symbolic_family_pools_meta.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "project/data/mx1_live_mining_probe_meta.json"
OUT_LABELS = ROOT / "project/data/mx1_minimal_symbolic_frontier_labels.json"
OUT_POOLS = ROOT / "project/data/mx1_symbolic_family_pools_meta.json"

KNOWN_BAD = {"Multiset.eq_of_mem_map_const"}
RAW_BATTERY = ["assumption", "rfl", "decide", "simp", "simp_all", "aesop"]


def action_id_for(tactic: str, ns: str) -> str | None:
    """Map a minimal symbolic tactic string to a stable action id."""
    if tactic.startswith("ext x <;> "):
        mode = tactic.split("<;>", 1)[1].strip()
        pref = {"Set": "SET_EXT_SIMP", "Finset": "FINSET_EXT_SIMP",
                "Multiset": "EXT_SIMP"}.get(ns, "EXT_SIMP")
        return f"{pref}[{ns},{mode}]"
    if "using Multiset.induction_on <;> " in tactic:
        mode = tactic.rsplit("<;>", 1)[1].strip()
        return f"MULTISET_INDUCTION_SIMP[Multiset,{mode}]"
    if tactic.split()[0] == "cases" and "<;>" in tactic:
        mode = tactic.split("<;>", 1)[1].strip()
        pref = "FINSET_CASES_SIMP" if ns == "Finset" else "CASES_SIMP"
        return f"{pref}[{ns},{mode}]"
    return None


def main() -> None:
    if not PROBE.exists():
        print("no probe meta; run mx1_collect_live_mining.py first")
        return
    probe = json.loads(PROBE.read_text())
    new_wins = [r for r in probe.get("new_win_records", [])
                if r["theorem"] not in KNOWN_BAD]

    labels = []
    if new_wins:
        from env import make_repo, make_theorem
        from core_types import TheoremConfig
        from lean_dojo import Dojo, ProofFinished, TacticState
        from project.evolve.state_vars import vars_of_type
        repo = make_repo()

        # need file_path per theorem — pull from the MX1 theorem sets
        sets = json.loads(
            (ROOT / "project/evolve/routing/mx1_theorem_sets.json").read_text())
        fp = {}
        for items in sets.values():
            for it in items:
                fp[it["full_name"]] = it["file_path"]

        for rec in new_wins:
            fn = rec["theorem"]
            ns = rec["namespace"]
            file_path = fp.get(fn)
            if not file_path:
                labels.append({**rec, "classification": "dropped",
                               "reason": "no file_path"})
                continue
            thm = make_theorem(repo, TheoremConfig(file_path=file_path,
                                                   full_name=fn))
            cls = "flaky"
            minimal = None
            try:
                with Dojo(thm) as (dojo, state):
                    sp = state.pp
                    # build the var-dependent battery from the live state
                    battery = list(RAW_BATTERY)
                    for mode in ("simp", "simp_all"):
                        battery.append(f"ext x <;> {mode}")
                    cvars = (vars_of_type(sp, ns, max_vars=1)
                             if ns in ("Finset", "Multiset", "List", "Option")
                             else [])
                    for v in cvars:
                        for mode in ("simp", "simp_all"):
                            battery.append(f"cases {v} <;> {mode}")
                    if ns == "Multiset":
                        for v in vars_of_type(sp, "Multiset", max_vars=1):
                            for mode in ("simp", "simp_all"):
                                battery.append(
                                    f"induction {v} using Multiset.induction_on "
                                    f"<;> {mode}")
                    # always include the wrapper's winning tactic as a fallback
                    wt = rec.get("winning_symbolic_tactic")
                    if wt and wt not in battery:
                        battery.append(wt)

                    advanced_then_closed = None
                    for tac in battery:
                        try:
                            res = dojo.run_tac(state, tac)
                        except Exception:
                            continue
                        if isinstance(res, ProofFinished):
                            minimal = tac
                            if tac in RAW_BATTERY:
                                cls = "over_attributed_raw"
                            else:
                                cls = "clean_single_shot_symbolic"
                            break
                        # depth-2 detection: a symbolic-first advance, then a
                        # raw follow-up closes
                        if isinstance(res, TacticState) and tac not in RAW_BATTERY:
                            for fu in ("simp_all", "aesop", "simp"):
                                try:
                                    r2 = dojo.run_tac(res, fu)
                                except Exception:
                                    continue
                                if isinstance(r2, ProofFinished):
                                    advanced_then_closed = (tac, fu)
                                    break
                            if advanced_then_closed:
                                break
                    if minimal is None and advanced_then_closed:
                        cls = "sequence_needed"
                        minimal = " then ".join(advanced_then_closed)
            except Exception as e:
                cls = "dropped"
                minimal = f"dojo_error:{type(e).__name__}"

            aid = action_id_for(minimal or "", ns) \
                if cls in ("clean_single_shot_symbolic", "sequence_needed") \
                else None
            labels.append({**rec, "classification": cls,
                           "minimal_closer": minimal, "action_id": aid})
            print(f"  {fn:40s} {cls:28s} <- {minimal}")

    by_cls = defaultdict(int)
    for r in labels:
        by_cls[r["classification"]] += 1

    # family pools: clean single-shot symbolic labels by action id
    pools = defaultdict(list)
    for r in labels:
        if r["classification"] == "clean_single_shot_symbolic" and r.get("action_id"):
            pools[r["action_id"]].append({
                "theorem": r["theorem"], "minimal_tactic": r["minimal_closer"]})

    OUT_LABELS.write_text(json.dumps({
        "description": "MX1 Stage 5 — LIVE strict minimal relabel of new wins "
                       "beyond the production wrapper.",
        "n_new_wins": len(new_wins),
        "by_classification": dict(by_cls),
        "labels": labels,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT_POOLS.write_text(json.dumps({
        "description": "MX1 clean single-shot symbolic-label family pools "
                       "(new this arc).",
        "gate_unique_required": 20,
        "n_clean_labels": sum(len(v) for v in pools.values()),
        "biggest_pool": max((len(v) for v in pools.values()), default=0),
        "pools": {k: {"unique_count": len(v), "labels": v}
                  for k, v in pools.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {OUT_LABELS.relative_to(ROOT)}")
    print(f"wrote {OUT_POOLS.relative_to(ROOT)}")
    print(f"new_wins={len(new_wins)} by_class={dict(by_cls)} "
          f"clean_pools={ {k: len(v) for k, v in pools.items()} }")


if __name__ == "__main__":
    main()
