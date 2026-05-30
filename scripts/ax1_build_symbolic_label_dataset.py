"""AX1 Stage 7 — symbolic-label dataset prototype (no training).

Builds a small metadata dataset from the WX1 (Option) and WX2 (List)
wrapper wins. Each example pairs the variable-dependent raw tactic with
its *state-independent symbolic label* — the target a future AX2 model
would learn:

    not  `cases xs <;> simp_all`
    but  `CASES_SIMP[List,simp_all]`   (+ var read from state at apply time)

Reads the WX1/WX2 minimal-tactic relabel outputs (which carry the minimal
tactic, variable, and namespace) and the initial proof state from eval
traces (for a short snippet/hash). Writes metadata only.

Output: project/data/ax1_symbolic_label_dataset_meta.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project.evolve.symbolic_actions import SymbolicAction

_NS_VARTYPE = {"List": "List", "Option": "Option", "Bool": "Bool"}


def parse_symbolic_label(minimal_tactic: str, namespace: str):
    """Map a minimal tactic + namespace to a SymbolicAction, or None."""
    t = (minimal_tactic or "").strip()
    m = re.match(r"^(cases|induction)\s+(\S+)\s+<;>\s+(simp_all|simp|decide)\s*$", t)
    if not m:
        return None, None
    head, var, mode = m.group(1), m.group(2), m.group(3)
    var_type = _NS_VARTYPE.get(namespace)
    if var_type is None:
        return None, var
    action_type = "CASES_SIMP" if head == "cases" else "INDUCTION_SIMP"
    return SymbolicAction(
        action_type=action_type, var_type=var_type, simp_mode=mode,
        namespace_gate=var_type, max_vars=2, priority=40,
    ), var


def load_initial_states(globs):
    init = {}
    for g in globs:
        for tf in glob.glob(g):
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


def main() -> None:
    states = load_initial_states([
        "project/evolve/eval_runs/wx1_wx2gen_wx2_*/*/traces.jsonl",
        "project/evolve/eval_runs/cx3_*/*/traces.jsonl",
        "project/evolve/eval_runs/wx1_wx1_*/*/traces.jsonl",
    ])

    examples = []
    for arc, labels_path in [
        ("WX1", "project/data/wx1_minimal_tactic_labels.json"),
        ("WX2", "project/data/wx2_minimal_tactic_labels.json"),
    ]:
        if not Path(labels_path).exists():
            continue
        for r in json.load(open(labels_path)).get("relabel_results", []):
            mt = r.get("minimal_tactic")
            ns = r.get("namespace", "")
            if not mt:
                continue
            action, var = parse_symbolic_label(mt, ns)
            var_dependent = bool(re.match(r"^(cases|induction)\s", mt.strip()))
            st = states.get(r["full_name"], "")
            goal = next((ln.strip() for ln in st.splitlines()
                         if ln.lstrip().startswith("⊢")), "")
            examples.append({
                "arc": arc,
                "theorem": r["full_name"],
                "namespace": ns,
                "raw_winning_tactic": mt,
                "symbolic_label": action.action_id if action else None,
                "symbolic_label_dict": action.to_dict() if action else None,
                "instantiated_tactic": mt if action else None,
                "variable_used": var or r.get("minimal_var"),
                "family_source": (action.default_family_source()
                                  if action else r.get("minimal_family")),
                "raw_tactic_is_variable_dependent": var_dependent,
                "state_hash": hashlib.sha1(st.encode()).hexdigest()[:12] if st else None,
                "goal_snippet": goal[:160],
            })

    labelled = [e for e in examples if e["symbolic_label"]]
    by_label = Counter(e["symbolic_label"] for e in labelled)
    by_arc = Counter(e["arc"] for e in labelled)
    var_dep = sum(1 for e in examples if e["raw_tactic_is_variable_dependent"])

    out = {
        "purpose": ("Prototype dataset showing the AX2 training target: a "
                    "state-independent symbolic action label (e.g. "
                    "CASES_SIMP[List,simp_all]) instead of the "
                    "variable-dependent raw tactic (cases xs <;> simp_all). "
                    "No training performed."),
        "total_examples": len(examples),
        "labelled_examples": len(labelled),
        "variable_dependent_raw_tactics": var_dep,
        "by_symbolic_label": dict(by_label.most_common()),
        "by_arc": dict(by_arc.most_common()),
        "examples": examples,
    }
    Path("project/data/ax1_symbolic_label_dataset_meta.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote project/data/ax1_symbolic_label_dataset_meta.json")
    print(f"total={len(examples)} labelled={len(labelled)} "
          f"var_dependent={var_dep}")
    print(f"by_symbolic_label: {dict(by_label)}")
    print(f"by_arc: {dict(by_arc)}")


if __name__ == "__main__":
    main()
