"""CX1 Stage 1 — audit current theorem catalog coverage.

Reads project/discovered_theorems.json and tasks.py THEOREM_SETS,
emits:
  - project/data/cx1_catalog_audit_meta.json
  - project/evolve/reports/cx1_catalog_audit.md

The audit answers: which namespaces are exhausted? Which sets cover
which catalog theorems? What is the remaining unmined surface per
namespace?

NOTE: this is a read-only inventory script. It does not import
LeanDojo or hit Lean.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


CATALOG_PATH = Path("project/discovered_theorems.json")
META_OUT = Path("project/data/cx1_catalog_audit_meta.json")
MD_OUT = Path("project/evolve/reports/cx1_catalog_audit.md")

# Namespaces to feature in the report.
FOCUS_NAMESPACES = (
    "Nat", "Finset", "Set", "List", "Multiset",
    "Bool", "Option", "Int",
)


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    theorems = catalog["theorems"]

    # Per-namespace + per-difficulty counts.
    by_ns: Counter[str] = Counter()
    by_ns_diff: dict[str, Counter[str]] = defaultdict(Counter)
    for t in theorems:
        name = t.get("full_name", "")
        ns = name.split(".", 1)[0] if "." in name else "_no_ns_"
        by_ns[ns] += 1
        by_ns_diff[ns][t.get("difficulty", "?")] += 1

    # Per-source-file counts.
    by_file: Counter[str] = Counter()
    for t in theorems:
        by_file[t.get("file_path", "?")] += 1

    # Which theorems are used in which sets?
    used_by_set: dict[str, list[str]] = {}
    all_used: set[str] = set()
    for set_name, cfgs in THEOREM_SETS.items():
        thms = [c.full_name for c in cfgs]
        used_by_set[set_name] = thms
        all_used.update(thms)

    # Per-namespace usage breakdown.
    catalog_names = {t["full_name"] for t in theorems}
    ns_used_count: Counter[str] = Counter()
    ns_unused_count: Counter[str] = Counter()
    for t in theorems:
        name = t["full_name"]
        ns = name.split(".", 1)[0] if "." in name else "_no_ns_"
        if name in all_used:
            ns_used_count[ns] += 1
        else:
            ns_unused_count[ns] += 1

    # Used-but-not-in-catalog count (theorems used in sets but not
    # in the discovered catalog — e.g. demo_v1 hand-written tasks).
    used_not_in_catalog = sorted(all_used - catalog_names)

    # Exhaustion verdicts.
    verdicts: dict[str, str] = {}
    for ns in FOCUS_NAMESPACES:
        total = by_ns.get(ns, 0)
        unused = ns_unused_count.get(ns, 0)
        if total == 0:
            verdicts[ns] = f"{ns}: ABSENT from catalog"
        elif unused == 0:
            verdicts[ns] = f"{ns}: EXHAUSTED ({total}/{total} used)"
        elif unused < total * 0.1:
            verdicts[ns] = f"{ns}: ~exhausted ({unused}/{total} unused, < 10%)"
        else:
            verdicts[ns] = f"{ns}: HAS REMAINING SURFACE ({unused}/{total} unused)"

    # Set-by-set coverage for the focus namespaces (so the report
    # shows which sets cover which namespaces).
    set_ns_breakdown: dict[str, dict[str, int]] = {}
    for set_name, names in used_by_set.items():
        bd: Counter[str] = Counter()
        for n in names:
            ns = n.split(".", 1)[0] if "." in n else "_no_ns_"
            bd[ns] += 1
        set_ns_breakdown[set_name] = dict(bd)

    meta = {
        "catalog_commit": catalog.get("commit"),
        "catalog_files_scanned": catalog.get("files_scanned", []),
        "catalog_total_theorems": catalog.get("total_extracted", len(theorems)),
        "per_difficulty": catalog.get("per_difficulty", {}),
        "by_namespace": dict(by_ns),
        "by_namespace_difficulty": {k: dict(v) for k, v in by_ns_diff.items()},
        "by_source_file": dict(by_file),
        "theorem_sets_count": len(THEOREM_SETS),
        "all_used_unique": len(all_used),
        "used_not_in_catalog_count": len(used_not_in_catalog),
        "used_not_in_catalog_sample": used_not_in_catalog[:20],
        "used_count_by_namespace": dict(ns_used_count),
        "unused_count_by_namespace": dict(ns_unused_count),
        "exhaustion_verdicts": verdicts,
        "theorem_set_namespace_breakdown": set_ns_breakdown,
    }
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Markdown report.
    lines: list[str] = []
    lines.append("# CX1 — current catalog audit\n")
    lines.append(f"**Catalog:** `{CATALOG_PATH}`")
    lines.append(f"**Mathlib commit:** `{catalog.get('commit', 'unknown')}`")
    lines.append(f"**Total theorems:** **{len(theorems)}**")
    pd = catalog.get("per_difficulty", {})
    diff_s = ", ".join(f"{k}={v}" for k, v in pd.items())
    lines.append(f"**By difficulty:** {diff_s}\n")

    lines.append("## Source files scanned\n")
    lines.append("| file | theorems |")
    lines.append("|---|---:|")
    for f, c in by_file.most_common():
        lines.append(f"| `{f}` | {c} |")
    lines.append("")

    lines.append("## Per-namespace counts\n")
    lines.append("| namespace | total | easy | medium | hard |")
    lines.append("|---|---:|---:|---:|---:|")
    for ns in FOCUS_NAMESPACES:
        total = by_ns.get(ns, 0)
        bd = by_ns_diff.get(ns, Counter())
        lines.append(
            f"| `{ns}` | {total} | {bd.get('easy', 0)} | "
            f"{bd.get('medium', 0)} | {bd.get('hard', 0)} |"
        )
    other_total = sum(c for ns, c in by_ns.items() if ns not in FOCUS_NAMESPACES)
    lines.append(f"| _other_ | {other_total} | – | – | – |")
    lines.append("")

    lines.append("## Theorem-set coverage\n")
    lines.append(f"**Total distinct theorems used across all "
                 f"`THEOREM_SETS`:** {len(all_used)}")
    lines.append(f"**Used but absent from the discovered catalog:** "
                 f"{len(used_not_in_catalog)} (e.g. hand-written tasks)\n")
    lines.append("Examples of used-but-not-in-catalog:")
    for n in used_not_in_catalog[:8]:
        lines.append(f"- `{n}`")
    lines.append("")

    lines.append("## Per-namespace usage / remainder\n")
    lines.append("| namespace | catalog | used | unused | exhaustion |")
    lines.append("|---|---:|---:|---:|---|")
    for ns in FOCUS_NAMESPACES:
        total = by_ns.get(ns, 0)
        used = ns_used_count.get(ns, 0)
        unused = ns_unused_count.get(ns, 0)
        verdict_short = verdicts.get(ns, "?").split(":", 1)[1].strip()
        lines.append(f"| `{ns}` | {total} | {used} | {unused} | {verdict_short} |")
    lines.append("")

    lines.append("## Exhaustion summary\n")
    for ns in FOCUS_NAMESPACES:
        v = verdicts.get(ns, f"{ns}: unknown")
        lines.append(f"- {v}")
    lines.append("")

    lines.append("## CX1 implications\n")
    lines.append(
        "The current catalog was built by scanning only "
        "3 Mathlib source files: `Mathlib/Data/Nat/Defs.lean`, "
        "`Mathlib/Data/Set/Basic.lean`, `Mathlib/Data/Finset/Basic.lean`. "
        "Nat is fully exhausted; Finset is nearly so. Set has remaining "
        "surface but the Set base also looks largely used. List/Multiset "
        "are only 13 thms each — most of the namespace surface is "
        "outside the scanned files. Bool/Option/Int are absent.\n"
    )
    lines.append(
        "**`extract_theorems.py` already lists 16 EXTENDED_FILES that "
        "have not yet been scanned**, including `Finset/Image.lean`, "
        "`Finset/Card.lean`, `Nat/GCD/Basic.lean`, `List/Basic.lean`, "
        "`Bool/Basic.lean`, and `Int/Basic.lean`. Scanning these is the "
        "natural CX1 Stage 2.\n"
    )
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {META_OUT}")
    print(f"wrote {MD_OUT}")
    print()
    print("Exhaustion verdicts:")
    for ns in FOCUS_NAMESPACES:
        print(f"  {verdicts.get(ns, 'unknown')}")


if __name__ == "__main__":
    main()
