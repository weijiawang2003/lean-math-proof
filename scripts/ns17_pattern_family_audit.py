"""NS17 Stage 1–3 — pattern-family audit of evolved + base supervision.

Walks every available supervised JSONL we have so far (v5 base,
NS11 combined, NS14, NS16 wrapper-only) plus the existing wrapper
trace eval-run trees, classifies every (state, tactic) pair into a
tactic *family*, and emits two outputs:

  - ``project/evolve/reports/ns17_pattern_family_audit.md`` — the
    human-readable report (Stage 3 deliverable).
  - ``project/data/ns17_family_audit.json`` — the machine-readable
    per-family table (used downstream by Stage 6).

Family classifier is greedy: each row is tested against an ordered
list of regex/string predicates; the first match wins.

The script does **not** train, evaluate, or write any large
artifacts; it just reads.

Usage::

    python scripts/ns17_pattern_family_audit.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


# ---- inputs -------------------------------------------------------------

# Each entry: (label, jsonl_path, expected variant tag). v5_base rows
# in NS11 combined are tagged ``_variant: "v5_base"``; evolved
# rows tagged ``medium`` / ``ns14`` / ``ns16``. We use this to split.
JSONL_INPUTS = [
    ("ns11_combined", Path("project/data/ns11_train_combined.jsonl")),
    ("ns14_combined", Path("project/data/ns14_train_combined.jsonl")),
    ("ns16_wrapper_only", Path("project/data/ns16_nat_wrapper_only.jsonl")),
]

# Wrapper trace eval dirs we want to scan for *additional* close
# transitions. Each is a path to a ``traces.jsonl`` glob.
TRACE_GLOBS = [
    # NS15 wrapper on canonical Nat sets
    "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_medium/eval-*/traces.jsonl",
    "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_large_v5/eval-*/traces.jsonl",
    "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_demo_v1/eval-*/traces.jsonl",
    # NS16 wrapper on NS16 sets
    "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_iff_extra/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_div_mod_extra/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_order_extra/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_mixed_extra/eval-*/traces.jsonl",
    # NS14 wrapper across all sets
    "project/evolve/eval_runs/ns14_routed_wrapper_nat/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns14_routed_wrapper_set_finset/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns14_routed_wrapper_mixed_easy/eval-*/traces.jsonl",
    "project/evolve/eval_runs/ns14_routed_wrapper_mixed_medium/eval-*/traces.jsonl",
]


NS11_HELD_OUT = {
    "Nat.AM_GM", "Nat.div_lt_iff_lt_mul'", "Nat.div_lt_one_iff",
    "Nat.div_pos", "Nat.div_pos_iff", "Nat.mul_eq_left",
    "Nat.mul_eq_right", "Nat.dvd_iff_div_mul_eq", "Nat.sqrt_lt",
    "Nat.pow_lt_pow_iff_left",
}


# ---- family classifier --------------------------------------------------

# Family rules are checked in order. Each rule is a (label, predicate)
# where predicate accepts the *tactic* string and returns bool.
#
# This is the homogeneity contract: every row in a family should match
# the same predicate. We deliberately keep rules narrow and obvious.

def _has(t: str, *needles: str) -> bool:
    return all(n in t for n in needles)


def _normalize_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def family_of(tactic: str) -> str:
    t = _normalize_ws(tactic)
    if not t:
        return "empty"

    # 1. iff_omega_pair — the NS14 / NS15 winner pattern.
    if (re.search(r"\b(exact|refine)\b", t)
        and "fun h => by omega" in t
        and t.count("by omega") >= 2):
        return "iff_omega_pair"

    # 2. iff_omega_left_only — one side omega, other side arbitrary.
    if "fun h => by omega" in t and re.search(r"\b(exact|refine)\b", t):
        return "iff_omega_left_only"

    # 3. split_ifs_omega
    if t.startswith("split_ifs") and ("omega" in t or "simp" in t):
        return "split_ifs_omega"

    # 4. constructor_omega — `constructor <;> omega` or similar.
    if re.match(r"constructor\b", t) and "omega" in t:
        return "constructor_omega"

    # 5. fallback_omega — bare omega.
    if t == "omega":
        return "fallback_omega"

    # 6. fallback_aesop
    if t == "aesop":
        return "fallback_aesop"

    # 7. fallback_decide
    if t == "decide":
        return "fallback_decide"

    # 8. fallback_norm_num
    if t == "norm_num":
        return "fallback_norm_num"

    # 9. fallback_rfl
    if t in {"rfl", "rfl'"}:
        return "fallback_rfl"

    # 10. simp_only / simp [Nat.add_mod, …] family.
    if (t.startswith("simp") and re.search(r"\bNat\.(add|mul|sub|mod|div)_", t)):
        return "nat_simp_arith"

    # 11. Set.subset_def emission.
    if "Set.subset_def" in t:
        return "set_subset_simp"

    # 12. Set.ext_iff emission.
    if "Set.ext_iff" in t:
        return "set_ext_simp"

    # 13. Finset.ext / Finset coe.
    if t.startswith("simp") and "Finset" in t and ("ext" in t or "coe" in t):
        return "finset_ext_or_coe_simp"

    # 14. nat_div_rw — rw with Nat.div_/mul_div/.. lemma in args.
    if t.startswith("rw") and re.search(r"Nat\.(div|mod|dvd)_", t):
        return "nat_div_rw"

    # 15. rw_retrieved — any rw of a named lemma.
    if t.startswith("rw"):
        return "rw_named"

    # 16. apply_named — `apply foo.bar.baz`.
    if t.startswith("apply ") and "." in t:
        return "apply_named"

    # 17. cases_omega
    if t.startswith("cases") and "omega" in t:
        return "cases_omega"

    # 18. intro_then_simp
    if t.startswith("intro") and "simp" in t:
        return "intro_simp"

    # 19. simp_only / simp baseline (no Nat args).
    if t.startswith("simp_only") or t == "simp" or t.startswith("simp ["):
        return "simp_baseline"

    # 20. exact_named — `exact foo.bar`.
    if t.startswith("exact ") and "." in t and "fun" not in t:
        return "exact_named"

    # Catch-all.
    return "other"


# ---- ingestion ----------------------------------------------------------

def is_jsonl_path(p: Path) -> bool:
    return p.exists() and p.suffix == ".jsonl"


def domain_of_name(name: str) -> str:
    if not name or "." not in name:
        return "other"
    head = name.split(".", 1)[0]
    return head if head in ("Nat", "Set", "Finset", "List", "Option",
                            "Bool", "Multiset") else "other"


def load_jsonl_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def collect_from_jsonl() -> list[dict]:
    rows: list[dict] = []
    per_source_counts: dict[str, int] = {}
    for label, path in JSONL_INPUTS:
        if not is_jsonl_path(path):
            continue
        n = 0
        for r in load_jsonl_rows(path):
            tactic = r.get("tactic") or r.get("completion") or ""
            thm = r.get("theorem") or ""
            row = {
                "source": label,
                "theorem": thm,
                "theorem_set": r.get("theorem_set") or "",
                "origin": r.get("origin") or "unknown",
                "role": r.get("role") or "unknown",
                "namespace": domain_of_name(thm),
                "tactic": tactic,
                "family": family_of(tactic),
                "variant": r.get("_variant") or "unknown",
                "is_held_out": thm in NS11_HELD_OUT,
                "state_hash": r.get("state_hash") or "",
                "tactic_hash": r.get("tactic_hash") or "",
                "wrapper_only": bool(r.get("wrapper_only", False)),
            }
            rows.append(row)
            n += 1
        per_source_counts[label] = n
    return rows


def is_close_trace(r: dict) -> bool:
    return bool(r.get("proof_finished"))


def collect_from_traces() -> list[dict]:
    """Scan wrapper trace JSONLs for closing transitions we haven't
    already absorbed via the JSONL training data."""
    rows: list[dict] = []
    for pattern in TRACE_GLOBS:
        for path_str in glob.glob(pattern):
            tp = Path(path_str)
            for line in tp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not is_close_trace(r):
                    continue
                tactic = r.get("tactic") or ""
                state_pp = r.get("state_pp") or r.get("state_pp_before") or ""
                full = r.get("full_name") or ""
                if not tactic or not state_pp:
                    continue
                rows.append({
                    "source": f"trace:{tp.parent.parent.name}",
                    "theorem": full,
                    "theorem_set": r.get("theorem_set") or tp.parent.parent.name,
                    "origin": r.get("tactic_origin") or "unknown",
                    "role": "close",
                    "namespace": domain_of_name(full),
                    "tactic": tactic,
                    "family": family_of(tactic),
                    "variant": "trace",
                    "is_held_out": full in NS11_HELD_OUT,
                    "state_hash": hashlib.sha1(state_pp.encode("utf-8")).hexdigest()[:16],
                    "tactic_hash": hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12],
                    "wrapper_only": False,  # set later via raw comparison
                })
    return rows


# ---- aggregation --------------------------------------------------------

def split_evolved(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """v5 base vs. evolved (wrapper-derived) rows."""
    base = [r for r in rows if r["variant"] == "v5_base"]
    evolved = [r for r in rows if r["variant"] != "v5_base"]
    return base, evolved


def family_table(rows: list[dict]) -> dict[str, dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    out = {}
    for fam, items in by_family.items():
        thms = {r["theorem"] for r in items}
        ns_counts: dict[str, int] = defaultdict(int)
        origin_counts: dict[str, int] = defaultdict(int)
        role_counts: dict[str, int] = defaultdict(int)
        set_counts: dict[str, int] = defaultdict(int)
        held_out = 0
        wrapper_only = 0
        for r in items:
            ns_counts[r["namespace"]] += 1
            origin_counts[r["origin"]] += 1
            role_counts[r["role"]] += 1
            if r["theorem_set"]:
                set_counts[r["theorem_set"]] += 1
            if r["is_held_out"]:
                held_out += 1
            if r["wrapper_only"]:
                wrapper_only += 1
        lens = [len(r["tactic"]) for r in items if r["tactic"]]
        out[fam] = {
            "rows": len(items),
            "unique_theorems": len(thms),
            "held_out_rows": held_out,
            "wrapper_only_rows": wrapper_only,
            "by_namespace": dict(ns_counts),
            "by_origin": dict(origin_counts),
            "by_role": dict(role_counts),
            "tactic_len_median": int(statistics.median(lens)) if lens else 0,
            "example_tactics": sorted({r["tactic"] for r in items})[:3],
            "theorem_sets": sorted(set_counts.keys())[:10],
        }
    return out


def write_md(report_path: Path, summary: dict) -> None:
    out: list[str] = []
    out.append("# NS17 — Pattern family audit\n")
    out.append("Inventory of supervised (state, tactic) pairs we've "
               "accumulated through NS16, grouped by *tactic family*. "
               "The hypothesis from NS15→NS16 was that transfer "
               "requires enough rows *per family*, not per dataset; "
               "this audit checks that claim against the data we have.\n")

    out.append("## Inputs\n")
    out.append("| source | rows |")
    out.append("|---|---:|")
    for k, v in summary["input_counts"].items():
        out.append(f"| `{k}` | {v} |")
    out.append("")

    for partition in ("v5_base", "evolved", "traces_close_only"):
        if partition not in summary:
            continue
        block = summary[partition]
        out.append(f"## Partition: `{partition}` "
                   f"({block['total_rows']} rows, "
                   f"{block['unique_theorems']} thms)\n")
        out.append("| family | rows | thms | wrapper-only | held-out | example |")
        out.append("|---|---:|---:|---:|---:|---|")
        # Sort families by row count desc.
        fam_items = sorted(block["families"].items(),
                           key=lambda kv: -kv[1]["rows"])
        for fam, info in fam_items:
            example = info["example_tactics"][0] if info["example_tactics"] else ""
            example = example.replace("\n", "⏎")
            if len(example) > 70:
                example = example[:67] + "…"
            out.append(
                f"| `{fam}` | {info['rows']} | {info['unique_theorems']} | "
                f"{info['wrapper_only_rows']} | {info['held_out_rows']} | "
                f"`{example}` |"
            )
        out.append("")

    out.append("## Family-by-family commentary\n")
    families_of_interest = (
        ("iff_omega_pair", "The NS14 winner. Look for: row count, "
            "namespace breadth, whether trace partition has more rows "
            "than JSONL partition."),
        ("iff_omega_left_only", "Variant of the winner with one side "
            "not omega; smaller pool."),
        ("split_ifs_omega", "Reaches Nat if-then-else theorems."),
        ("nat_simp_arith", "simp [Nat.add_mod, Nat.mul_…] patterns; "
            "broad family with many sub-variants."),
        ("nat_div_rw", "rw [Nat.div_lt_iff_…]; high homogeneity but "
            "narrow coverage."),
        ("set_subset_simp", "demo_v1 retention driver."),
        ("set_ext_simp", "Set.ext_iff emissions."),
        ("fallback_omega", "Bare omega — should already be in raw."),
        ("fallback_aesop", "Heuristic Mathlib closer."),
        ("rw_named", "Any rw of a named lemma."),
        ("apply_named", "Any apply of a named lemma."),
    )
    for fam, blurb in families_of_interest:
        out.append(f"- `{fam}` — {blurb}")
    out.append("")

    out.append("## Headline numbers\n")
    if "evolved" in summary:
        ev = summary["evolved"]["families"]
        out.append(
            f"- Evolved supervision total: "
            f"**{summary['evolved']['total_rows']} rows / "
            f"{summary['evolved']['unique_theorems']} theorems**.\n"
        )
        for fam in ("iff_omega_pair", "fallback_omega", "split_ifs_omega",
                    "nat_div_rw", "nat_simp_arith", "set_subset_simp"):
            info = ev.get(fam) or {}
            n_rows = info.get("rows", 0)
            n_thms = info.get("unique_theorems", 0)
            n_wo = info.get("wrapper_only_rows", 0)
            out.append(f"  - `{fam}`: {n_rows} rows / {n_thms} thms "
                       f"({n_wo} wrapper-only)")
    if "traces_close_only" in summary:
        tr = summary["traces_close_only"]["families"]
        out.append("\nFrom raw wrapper trace closings (deduplicated by "
                   "state+tactic, not yet filtered to wrapper-only):\n")
        for fam in ("iff_omega_pair", "fallback_omega", "split_ifs_omega",
                    "nat_div_rw", "nat_simp_arith", "set_subset_simp",
                    "fallback_aesop"):
            info = tr.get(fam) or {}
            n_rows = info.get("rows", 0)
            n_thms = info.get("unique_theorems", 0)
            out.append(f"  - `{fam}` traces: {n_rows} rows / {n_thms} thms")
    out.append("")

    out.append("## NS18 transfer-readiness gate\n")
    out.append(
        "A family is a strong NS18 training candidate if **all** of:\n"
        "- ≥ 10 wrapper-only rows in the evolved partition, OR\n"
        "- ≥ 20 close transitions across distinct theorems in traces,\n"
        "- consistent tactic surface (small example-tactic count),\n"
        "- a held-out sibling theorem surface exists to evaluate.\n"
    )
    if "evolved" in summary:
        ev = summary["evolved"]["families"]
        out.append("| family | evolved rows | unique thms | wrapper-only | gate |")
        out.append("|---|---:|---:|---:|---|")
        for fam, info in sorted(ev.items(), key=lambda kv: -kv[1]["rows"]):
            wo = info["wrapper_only_rows"]
            n_thms = info["unique_theorems"]
            gate = "PASS" if (wo >= 10 or n_thms >= 10) else "fail"
            out.append(f"| `{fam}` | {info['rows']} | {n_thms} | {wo} | {gate} |")
    out.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    jsonl_rows = collect_from_jsonl()
    trace_rows = collect_from_traces()

    # JSONL rows already include v5_base + ns11/ns14/ns16 evolved.
    # We treat trace rows as a separate partition because they may
    # double-count rows already in the JSONL.
    base, evolved = split_evolved(jsonl_rows)

    # Dedup traces by (theorem, state_hash, tactic_hash) so we
    # don't double-count identical states across runs.
    seen: set[tuple[str, str, str]] = set()
    trace_dedup: list[dict] = []
    for r in trace_rows:
        key = (r["theorem"], r["state_hash"], r["tactic_hash"])
        if key in seen:
            continue
        seen.add(key)
        trace_dedup.append(r)

    summary = {
        "input_counts": {
            **{label: sum(1 for r in jsonl_rows if r["source"] == label)
               for label, _ in JSONL_INPUTS},
            "trace_close_pre_dedup": len(trace_rows),
            "trace_close_post_dedup": len(trace_dedup),
        },
        "v5_base": {
            "total_rows": len(base),
            "unique_theorems": len({r["theorem"] for r in base}),
            "families": family_table(base),
        },
        "evolved": {
            "total_rows": len(evolved),
            "unique_theorems": len({r["theorem"] for r in evolved}),
            "families": family_table(evolved),
        },
        "traces_close_only": {
            "total_rows": len(trace_dedup),
            "unique_theorems": len({r["theorem"] for r in trace_dedup}),
            "families": family_table(trace_dedup),
        },
    }

    out_json = Path("project/data/ns17_family_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = Path("project/evolve/reports/ns17_pattern_family_audit.md")
    write_md(report_path, summary)

    print(f"v5_base    : {summary['v5_base']['total_rows']} rows / "
          f"{summary['v5_base']['unique_theorems']} thms")
    print(f"evolved    : {summary['evolved']['total_rows']} rows / "
          f"{summary['evolved']['unique_theorems']} thms")
    print(f"traces (close, post-dedup) : "
          f"{summary['traces_close_only']['total_rows']} rows / "
          f"{summary['traces_close_only']['unique_theorems']} thms")
    print()
    print("Evolved partition by family (top 15):")
    for fam, info in sorted(summary["evolved"]["families"].items(),
                            key=lambda kv: -kv[1]["rows"])[:15]:
        print(f"  {fam:>24} {info['rows']:>3} rows  "
              f"{info['unique_theorems']:>3} thms  "
              f"WO={info['wrapper_only_rows']:>3}  "
              f"HO={info['held_out_rows']:>3}  "
              f"e.g. {info['example_tactics'][0][:50]!r}")
    print()
    print(f"wrote {out_json}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
