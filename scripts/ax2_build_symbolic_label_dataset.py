"""AX2 Stage 4 — build the symbolic-label dataset from symbolic-only wins.

For each theorem the AX1 symbolic wrapper solves that NS9 did NOT (the
symbolic-only-beyond-NS9 wins from the Stage 3 probe), pair the live proof
state with the *state-independent* symbolic action label the wrapper used
(e.g. CASES_SIMP[List,simp]) — the AX2 training target. Merge with the AX1
27-example prototype dataset.

Each example records: theorem, set, namespace, proof-state snippet + hash,
extracted local variables, symbolic action label (+dict), instantiated
tactic, variable used, whether NS9 solved (False by construction), whether
raw solved, and — if the Stage 5 minimal relabel has run — whether the
symbolic action is truly needed / minimal.

Outputs:
  project/data/ax2_symbolic_label_dataset_meta.json
  project/data/ax2_symbolic_label_dataset.jsonl  (gitignored; tiny)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from project.evolve.symbolic_actions import SymbolicAction  # noqa: E402
from project.evolve.state_vars import extract_state_variables  # noqa: E402

PROBE = ROOT / "project/data/ax2_symbolic_mining_probe_meta.json"
AX1_DS = ROOT / "project/data/ax1_symbolic_label_dataset_meta.json"
MINIMAL = ROOT / "project/data/ax2_minimal_symbolic_labels.json"
OUT_META = ROOT / "project/data/ax2_symbolic_label_dataset_meta.json"
OUT_JSONL = ROOT / "project/data/ax2_symbolic_label_dataset.jsonl"

_NS_VARTYPE = {"List": "List", "Option": "Option", "Bool": "Bool"}
_TAC_RE = re.compile(
    r"^\s*(cases|induction)\s+(\S+)\s+<;>\s+(simp_all|simp|decide)\s*$")


def parse_symbolic_label(tactic: str, namespace: str):
    m = _TAC_RE.match(tactic or "")
    if not m:
        return None, None
    head, var, mode = m.group(1), m.group(2), m.group(3)
    var_type = _NS_VARTYPE.get(namespace)
    if var_type is None:
        return None, var
    action_type = "CASES_SIMP" if head == "cases" else "INDUCTION_SIMP"
    return SymbolicAction(
        action_type=action_type, var_type=var_type, simp_mode=mode,
        namespace_gate=var_type, max_vars=2, priority=40), var


def load_initial_states(globs):
    init = {}
    for g in globs:
        for tf in glob.glob(str(ROOT / g)):
            for line in open(tf):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fn, st, pp = r.get("full_name"), r.get("step"), r.get("state_pp")
                if not fn or not pp or st is None:
                    continue
                if fn not in init or st < init[fn][0]:
                    init[fn] = (st, pp)
    return {k: v[1] for k, v in init.items()}


def load_minimal():
    """Optional Stage 5 output: {full_name: result}."""
    if not MINIMAL.exists():
        return {}
    try:
        data = json.load(open(MINIMAL))
    except Exception:
        return {}
    return {r["full_name"]: r for r in data.get("relabel_results", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-jsonl", action="store_true",
                    help="also write the tiny JSONL (gitignored)")
    args = ap.parse_args()

    if not PROBE.exists():
        sys.exit(f"missing probe meta {PROBE}; run Stage 3 first")
    probe = json.load(open(PROBE))
    sym_wins = probe["symbolic_only_beyond_ns9_theorems"]

    states = load_initial_states([
        "project/evolve/eval_runs/ax2_ax1sym_*/eval-*/traces.jsonl",
    ])
    minimal = load_minimal()

    new_examples = []
    for w in sym_wins:
        fn = w["full_name"]
        ns = w["namespace"]
        tac = w.get("winning_tactic") or ""
        action, var = parse_symbolic_label(tac, ns)
        st = states.get(fn, "")
        goal = next((ln.strip() for ln in st.splitlines()
                     if ln.lstrip().startswith("⊢")), "")
        svars = [v.to_dict() for v in extract_state_variables(st)] if st else []
        mrec = minimal.get(fn, {})
        new_examples.append({
            "arc": "AX2",
            "theorem": fn,
            "theorem_set": w.get("set"),
            "namespace": ns,
            "raw_winning_tactic": tac,
            "symbolic_label": action.action_id if action else w.get("symbolic_action_id"),
            "symbolic_label_dict": action.to_dict() if action else None,
            "instantiated_tactic": tac if action else None,
            "variable_used": var or w.get("variable_selected"),
            "family_source": (action.default_family_source()
                              if action else w.get("winning_family")),
            "raw_tactic_is_variable_dependent": bool(action),
            "ns9_solved": False,            # by construction (beyond-NS9)
            "raw_solved": not w.get("also_beyond_raw", True),
            "state_hash": hashlib.sha1(st.encode()).hexdigest()[:12] if st else None,
            "goal_snippet": goal[:160],
            "local_variables": svars,
            # Stage-5 minimal-relabel fields (None until Stage 5 runs):
            "minimal_tactic": mrec.get("minimal_tactic"),
            "symbolic_action_needed": mrec.get("symbolic_action_needed"),
            "final_training_label": mrec.get("final_training_label"),
            "minimal_unique_or_needed": mrec.get("symbolic_action_needed"),
        })

    # merge AX1 prototype examples (already minimal-relabeled & trusted)
    ax1_examples = []
    if AX1_DS.exists():
        for e in json.load(open(AX1_DS)).get("examples", []):
            if not e.get("symbolic_label"):
                continue
            ax1_examples.append({
                "arc": e.get("arc", "AX1"),
                "theorem": e["theorem"],
                "theorem_set": None,
                "namespace": e.get("namespace"),
                "raw_winning_tactic": e.get("raw_winning_tactic"),
                "symbolic_label": e.get("symbolic_label"),
                "symbolic_label_dict": e.get("symbolic_label_dict"),
                "instantiated_tactic": e.get("instantiated_tactic"),
                "variable_used": e.get("variable_used"),
                "family_source": e.get("family_source"),
                "raw_tactic_is_variable_dependent":
                    e.get("raw_tactic_is_variable_dependent"),
                "ns9_solved": False,
                "raw_solved": None,
                "state_hash": e.get("state_hash"),
                "goal_snippet": e.get("goal_snippet"),
                "local_variables": [],
                "minimal_tactic": e.get("raw_winning_tactic"),
                "symbolic_action_needed": True,   # AX1 set passed WX relabel
                "final_training_label": e.get("symbolic_label"),
                "minimal_unique_or_needed": True,
            })

    # de-dup by theorem (AX2 fresh sets are disjoint from AX1 by construction,
    # but guard anyway: prefer the AX2 entry if a collision occurs)
    seen = set()
    merged = []
    for e in new_examples + ax1_examples:
        if e["theorem"] in seen:
            continue
        seen.add(e["theorem"])
        merged.append(e)

    labelled = [e for e in merged if e["symbolic_label"]]
    by_label = Counter(e["symbolic_label"] for e in labelled)
    by_arc = Counter(e["arc"] for e in labelled)
    by_ns = Counter(e["namespace"] for e in labelled)

    out = {
        "purpose": ("AX2 symbolic-label dataset: state -> state-independent "
                    "symbolic action id (CASES_SIMP[List,simp], ...). New AX2 "
                    "examples come from fresh-List symbolic-only-beyond-NS9 "
                    "wins; merged with the AX1 27-example prototype. "
                    "final_training_label / symbolic_action_needed are filled "
                    "by Stage 5 minimal relabeling (NS23 over-attribution "
                    "discipline)."),
        "minimal_relabel_applied": bool(minimal),
        "counts": {
            "ax2_new_examples": len(new_examples),
            "ax1_merged_examples": len(ax1_examples),
            "total_examples": len(merged),
            "labelled_examples": len(labelled),
        },
        "by_symbolic_label": dict(by_label.most_common()),
        "by_arc": dict(by_arc.most_common()),
        "by_namespace": dict(by_ns.most_common()),
        "examples": merged,
    }
    OUT_META.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"ax2_new={len(new_examples)} ax1_merged={len(ax1_examples)} "
          f"total={len(merged)} labelled={len(labelled)}")
    print(f"by_symbolic_label: {dict(by_label)}")
    print(f"by_namespace: {dict(by_ns)}")

    if args.write_jsonl:
        with open(OUT_JSONL, "w") as fh:
            for e in labelled:
                fh.write(json.dumps({
                    "theorem": e["theorem"],
                    "goal_snippet": e["goal_snippet"],
                    "state_hash": e["state_hash"],
                    "label": e["final_training_label"] or e["symbolic_label"],
                }, ensure_ascii=False) + "\n")
        print(f"wrote {OUT_JSONL.relative_to(ROOT)} ({len(labelled)} rows)")


if __name__ == "__main__":
    main()
