"""WX1 Stage 1 — inventory the CX3 cases_simp|Option headroom pool.

Reads CX3 metadata + eval traces and, for each Option theorem in the
headroom `cases_simp|Option` pool, records: theorem set, raw/wrapper
solve status, wrapper winning tactic, CX3 minimal tactic, the initial
proof-state snippet, and the accessible Option/Bool context variables a
`cases <var> <;> simp_all` skeleton could target (via the same
`_extract_cases_vars` the WX1 wrapper uses).

Outputs:
  project/data/wx1_option_cases_inventory.json
  project/evolve/reports/wx1_option_cases_inventory.md
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evolve.strategy_wrapper import _extract_cases_vars


def load_initial_states() -> dict[str, str]:
    """Min-step state_pp per theorem from all CX3 eval traces."""
    init: dict[str, tuple[int, str]] = {}
    for tf in glob.glob("project/evolve/eval_runs/cx3_*/*/traces.jsonl"):
        with open(tf) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fn, st, pp = (r.get("full_name"), r.get("step"),
                              r.get("state_pp"))
                if not fn or not pp or st is None:
                    continue
                if fn not in init or st < init[fn][0]:
                    init[fn] = (st, pp)
    return {k: v[1] for k, v in init.items()}


def main() -> None:
    pools = json.load(open("project/data/cx3_minimal_family_pools_meta.json"))
    headroom = pools["headroom_pools"].get("cases_simp|Option", {})
    thm_meta = headroom.get("theorems", {})

    probe = json.load(open("project/data/cx3_bool_option_probe_meta.json"))
    rc = {c["full_name"]: c for c in probe["relabel_candidates"]}

    labels = {r["full_name"]: r for r in json.load(
        open("project/data/cx3_minimal_tactic_labels.json"))["relabel_results"]}

    states = load_initial_states()

    items = []
    for name in sorted(thm_meta):
        st = states.get(name, "")
        opt_vars = _extract_cases_vars(st, ["Option"], 3)
        bool_vars = _extract_cases_vars(st, ["Bool"], 3)
        cand = rc.get(name, {})
        lab = labels.get(name, {})
        # first goal line for the snippet
        goal_line = ""
        for ln in st.splitlines():
            if ln.lstrip().startswith("⊢"):
                goal_line = ln.strip()
                break
        items.append({
            "full_name": name,
            "theorem_set": cand.get("first_seen_set"),
            "raw_solved": cand.get("currently_solved_raw"),
            "wrapper_solved": cand.get("currently_solved_wrap"),
            "wrapper_winning_tactic": cand.get("wrapper_tactic") or None,
            "cx3_minimal_tactic": thm_meta[name].get("minimal_tactic"),
            "minimal_family": lab.get("minimal_family"),
            "goal": goal_line,
            "option_vars": opt_vars,
            "bool_vars": bool_vars,
            "cases_simp_likely": bool(opt_vars or bool_vars),
            "state_snippet": st[:500],
        })

    n_with_vars = sum(1 for it in items if it["cases_simp_likely"])
    out = {
        "source_pool": "cases_simp|Option (CX3 headroom)",
        "total": len(items),
        "with_extractable_cases_var": n_with_vars,
        "without_extractable_var": [it["full_name"] for it in items
                                    if not it["cases_simp_likely"]],
        "items": items,
    }
    Path("project/data/wx1_option_cases_inventory.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# WX1 — CX3 Option cases headroom inventory", ""]
    md.append(f"Pool: **cases_simp|Option** (CX3 headroom — theorems the "
              f"routed model fails but `cases <var> <;> simp_all` closes). "
              f"Total **{len(items)}**, with an extractable Option/Bool "
              f"context variable: **{n_with_vars}**.")
    md.append("")
    md.append("| theorem | set | raw | wrap | option vars | cx3 minimal tactic |")
    md.append("|---|---|:---:|:---:|---|---|")
    for it in items:
        md.append(
            f"| `{it['full_name']}` | {it['theorem_set']} | "
            f"{'Y' if it['raw_solved'] else 'n'} | "
            f"{'Y' if it['wrapper_solved'] else 'n'} | "
            f"`{', '.join(it['option_vars']) or '-'}` | "
            f"`{it['cx3_minimal_tactic']}` |")
    md.append("")
    md.append("## Goal snippets")
    md.append("")
    for it in items:
        md.append(f"- `{it['full_name']}` — vars `{it['option_vars']}` — "
                  f"`{it['goal']}`")
    Path("project/evolve/reports/wx1_option_cases_inventory.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("wrote project/data/wx1_option_cases_inventory.json")
    print("wrote project/evolve/reports/wx1_option_cases_inventory.md")
    print(f"\n{len(items)} headroom theorems, "
          f"{n_with_vars} with an extractable cases variable")
    for it in items:
        print(f"  {it['full_name']:34s} vars={it['option_vars']} "
              f"min={it['cx3_minimal_tactic']}")


if __name__ == "__main__":
    main()
