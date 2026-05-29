"""SX1 Stage 5 — carve sequence-search theorem sets from the trace corpus.

Candidates are the theorems where a symbolic first action is relevant: the
multistep symbolic-assisted cases (Stage 1), the *advanced-but-not-closed*
states (symbolic action made progress but the oracle's single action did not
finish — the prime target for a deliberate depth-2 sequence), plus a negative
control of theorems where the symbolic first action does not even instantiate
or should not help.

We read the mined trace corpus directly (state + result_kind + origin per
record) and the Stage-1 meta, bin by namespace and fate, prioritise
advanced-not-closed, and write disjoint sets (40-100 total).

Output: project/evolve/routing/sx1_theorem_sets.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "project/evolve/routing/sx1_theorem_sets.json"
SYM = "wrapper_symbolic_action"

WRAPPER = {"AX4": "ax4_wx3ind", "AX3": "ax3_wx3ind", "AX2": "ax2_ax1sym"}
NEG_CONTROL_GLOB = "project/evolve/eval_runs/ax4_wx3ind_ax4_multiset_negative_control/eval-*/traces.jsonl"


def _ns(fn):
    return fn.split(".", 1)[0] if fn else ""


def scan():
    """Return per-theorem fate: file_path, namespace, fate in
    {multistep, advanced, single_shot, no_symbolic}."""
    info = {}  # full_name -> dict
    for arc, wtag in WRAPPER.items():
        for tf in glob.glob(f"project/evolve/eval_runs/{wtag}_*/eval-*/traces.jsonl"):
            recs = defaultdict(list)
            for line in open(tf):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("full_name"):
                    recs[o["full_name"]].append(o)
            for fn, rs in recs.items():
                fp = next((r.get("file_path") for r in rs if r.get("file_path")), "")
                finished_from = defaultdict(list)
                for r in rs:
                    if r.get("result_kind") == "ProofFinished":
                        finished_from[r.get("state_hash_before")].append(r)
                fate = "no_symbolic"
                for r in rs:
                    if r.get("tactic_origin") != SYM:
                        continue
                    rk = r.get("result_kind")
                    if rk == "ProofFinished":
                        fate = "single_shot" if fate == "no_symbolic" else fate
                    elif rk == "TacticState":
                        if finished_from.get(r.get("state_hash_after")):
                            fate = "multistep"
                        elif fate in ("no_symbolic", "single_shot"):
                            fate = "advanced"
                prev = info.get(fn)
                rank = {"no_symbolic": 0, "single_shot": 1, "advanced": 2,
                        "multistep": 3}
                if prev is None or rank[fate] > rank[prev["fate"]]:
                    info[fn] = {"full_name": fn, "file_path": fp,
                                "namespace": _ns(fn), "fate": fate, "arc": arc}
    return info


def entry(d):
    return {"file_path": d["file_path"], "full_name": d["full_name"]}


def main() -> None:
    info = scan()

    def pick(ns, fates, cap):
        # priority order: multistep > advanced > single_shot
        order = {"multistep": 0, "advanced": 1, "single_shot": 2}
        cands = [d for d in info.values()
                 if d["namespace"] == ns and d["fate"] in fates]
        cands.sort(key=lambda d: (order.get(d["fate"], 9), d["full_name"]))
        return cands[:cap]

    used = set()

    def take(cands):
        out = []
        for d in cands:
            if d["full_name"] in used:
                continue
            used.add(d["full_name"])
            out.append(entry(d))
        return out

    # Reserve the mixed-eval slice FIRST (a fresh disjoint cross-namespace
    # slice) so the per-namespace sets don't starve it.
    mixed = []
    for ns in ("Multiset", "List"):
        mixed += take(pick(ns, {"advanced", "single_shot"}, 8))
    mixed = mixed[:20]

    ms = take(pick("Multiset", {"multistep", "advanced", "single_shot"}, 30))
    li = take(pick("List", {"multistep", "advanced", "single_shot"}, 20))
    # Option: the symbolic corpus carries no Option symbolic-assisted cases
    # (CX3 negative); this set is intentionally empty pending an Option mine.
    op = take(pick("Option", {"multistep", "advanced", "single_shot"}, 20))

    # negative control: Multiset theorems where the symbolic action did NOT
    # advance (no_symbolic / failed) — sequence first action should not help.
    neg = []
    for d in sorted(info.values(), key=lambda d: d["full_name"]):
        if d["namespace"] == "Multiset" and d["fate"] == "no_symbolic" \
                and d["full_name"] not in used:
            used.add(d["full_name"])
            neg.append(entry(d))
        if len(neg) >= 20:
            break

    sets = {
        "sx1_multiset_multistep_candidates": ms,
        "sx1_list_multistep_candidates": li,
        "sx1_option_multistep_candidates": op,
        "sx1_mixed_symbolic_sequence_eval": mixed,
        "sx1_negative_control": neg,
    }
    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    print(f"wrote {OUT.relative_to(ROOT)}  (total {total} theorems)")
    for k, v in sets.items():
        print(f"  {k:42s} {len(v)}")


if __name__ == "__main__":
    main()
