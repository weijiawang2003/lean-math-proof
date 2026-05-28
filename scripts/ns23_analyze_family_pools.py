"""NS23 Stage 4 — analyze family pool changes after minimal-tactic relabeling.

Reads:
  - project/data/ns23_wrapper_only_wins_raw_meta.json (original families)
  - project/data/ns23_minimal_tactic_labels.json (battery results)

Outputs:
  - project/data/ns23_minimal_family_pools_meta.json
  - project/evolve/reports/ns23_family_pool_comparison.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    src = json.load(open(
        "project/data/ns23_wrapper_only_wins_raw_meta.json"))
    labels = json.load(open(
        "project/data/ns23_minimal_tactic_labels.json"))

    name_to_orig: dict[str, dict] = {
        t["full_name"]: t for t in src["theorems"]
    }
    name_to_label: dict[str, dict] = {
        r["full_name"]: r for r in labels["relabel_results"]
    }

    rows: list[dict] = []
    relabel_counts = defaultdict(lambda: defaultdict(int))
    minimal_pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    unchanged = 0
    relabeled = 0
    unresolved = 0

    for name, info in name_to_orig.items():
        lab = name_to_label.get(name)
        if not lab:
            continue
        orig_family = info["original_family"]
        min_family = lab.get("minimal_family")
        min_tac = lab.get("minimal_tactic")
        if min_family is None:
            unresolved += 1
        elif min_family == orig_family:
            unchanged += 1
        else:
            relabeled += 1
        rows.append({
            "full_name": name,
            "namespace": info["namespace"],
            "original_family": orig_family,
            "minimal_family": min_family,
            "minimal_tactic": min_tac,
            "changed": (min_family is not None and min_family != orig_family),
            "wrapper_tactic": info.get("wrapper_tactic", ""),
            "first_seen_arc": info["first_seen_arc"],
        })
        # Cross-tab original vs minimal
        relabel_counts[orig_family][min_family or "unresolved"] += 1
        if min_family:
            key = (min_family, info["namespace"])
            minimal_pool[key].append({
                "theorem": name,
                "minimal_tactic": min_tac,
                "original_family": orig_family,
                "wrapper_tactic": info.get("wrapper_tactic", ""),
                "first_seen_arc": info["first_seen_arc"],
            })

    # Aggregate omega pool (fallback_omega + iff_omega_pair → omega).
    omega_aggregate: dict[str, list[dict]] = defaultdict(list)
    for (fam, ns), thms in minimal_pool.items():
        if fam in {"fallback_omega", "iff_omega_pair", "constructor_omega",
                   "split_ifs_omega"}:
            omega_aggregate[ns].extend(thms)

    # aesop-irreducible: those that remain aesop-minimal.
    aesop_irreducible = minimal_pool.get(("aesop", "Finset"), []) + \
        minimal_pool.get(("aesop", "Nat"), [])

    # Pools meeting the 5-win gate after relabel.
    gate = 5
    gated_pools = []
    for (fam, ns), thms in minimal_pool.items():
        if len(thms) >= gate:
            gated_pools.append({
                "family": fam, "namespace": ns,
                "unique_count": len(thms),
                "trainable": True,
                "theorems": thms,
            })
    # The omega aggregate per namespace.
    for ns, thms in omega_aggregate.items():
        if len(thms) >= gate:
            gated_pools.append({
                "family": "omega_aggregate", "namespace": ns,
                "unique_count": len(thms),
                "trainable": True,
                "theorems": thms,
                "_aggregate_of": [
                    "fallback_omega", "iff_omega_pair",
                    "constructor_omega", "split_ifs_omega",
                ],
            })

    out: dict = {
        "n_theorems_relabeled": len(rows),
        "n_unchanged": unchanged,
        "n_relabeled": relabeled,
        "n_unresolved": unresolved,
        "relabel_crosstab": {
            o: dict(m) for o, m in relabel_counts.items()
        },
        "minimal_family_pools": {
            f"{fam}|{ns}": {
                "unique_count": len(thms),
                "theorems": thms,
            }
            for (fam, ns), thms in minimal_pool.items()
        },
        "omega_aggregate_by_namespace": {
            ns: {
                "unique_count": len(thms),
                "theorems": thms,
            }
            for ns, thms in omega_aggregate.items()
        },
        "gated_pools": gated_pools,
        "per_theorem": rows,
    }
    Path("project/data/ns23_minimal_family_pools_meta.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    md = ["# NS23 — family pool comparison (original vs minimal-tactic relabel)", ""]
    md.append(f"- theorems re-tested: {len(rows)}")
    md.append(f"- unchanged labels: {unchanged}")
    md.append(f"- relabeled: {relabeled}")
    md.append(f"- unresolved (no battery tactic closed it): {unresolved}")
    md.append("")
    md.append("## Cross-tabulation: original → minimal")
    md.append("")
    md.append("| original family | → minimal family | count |")
    md.append("|---|---|---:|")
    for orig, by_min in relabel_counts.items():
        for min_fam, n in sorted(by_min.items(), key=lambda kv: -kv[1]):
            md.append(f"| `{orig}` | `{min_fam}` | {n} |")
    md.append("")
    md.append("## Per-namespace omega aggregate")
    md.append("")
    md.append("**Aggregate of** `fallback_omega` + `iff_omega_pair` + "
              "`constructor_omega` + `split_ifs_omega`.")
    md.append("")
    md.append("| namespace | unique | gate met? |")
    md.append("|---|---:|:---:|")
    for ns, thms in omega_aggregate.items():
        gate_mark = "✓" if len(thms) >= 5 else "✗"
        md.append(f"| {ns} | **{len(thms)}** | {gate_mark} |")
    md.append("")
    md.append("## Gated pools (≥5 unique under minimal labels)")
    md.append("")
    md.append("| family | namespace | unique | aggregate of |")
    md.append("|---|---|---:|---|")
    for p in gated_pools:
        agg = ", ".join(p.get("_aggregate_of", [])) or "—"
        md.append(f"| `{p['family']}` | {p['namespace']} | "
                  f"**{p['unique_count']}** | {agg} |")
    md.append("")
    md.append("## Aesop-irreducible theorems")
    md.append("")
    if not aesop_irreducible:
        md.append("(none)")
    else:
        md.append("These theorems are NOT closed by any tactic strictly "
                  "simpler than `aesop` in the battery — they remain "
                  "aesop-minimal and form the residual aesop pool.")
        md.append("")
        md.append("| theorem | namespace | original family |")
        md.append("|---|---|---|")
        for t in aesop_irreducible:
            ns = name_to_orig.get(t["theorem"], {}).get(
                "namespace", "?")
            md.append(f"| `{t['theorem']}` | {ns} | "
                      f"{t['original_family']} |")
    md.append("")
    md.append("## Per-theorem detail")
    md.append("")
    md.append("| theorem | namespace | orig | minimal | minimal tactic | changed |")
    md.append("|---|---|---|---|---|:---:|")
    for r in rows:
        ch = "**✓**" if r["changed"] else "—"
        tac = (r["minimal_tactic"] or "—")[:50]
        md.append(f"| `{r['full_name']}` | {r['namespace']} | "
                  f"`{r['original_family']}` | "
                  f"`{r['minimal_family'] or 'unresolved'}` | "
                  f"`{tac}` | {ch} |")
    Path("project/evolve/reports/ns23_family_pool_comparison.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print("wrote project/data/ns23_minimal_family_pools_meta.json")
    print("wrote project/evolve/reports/ns23_family_pool_comparison.md")
    print()
    print(f"=== summary ===")
    print(f"theorems: {len(rows)} | unchanged: {unchanged} | "
          f"relabeled: {relabeled} | unresolved: {unresolved}")
    print()
    print("omega aggregate by namespace:")
    for ns, thms in omega_aggregate.items():
        print(f"  {ns}: {len(thms)} unique")
    print()
    print("gated pools:")
    for p in gated_pools:
        print(f"  [TRAIN] {p['family']}|{p['namespace']}: "
              f"{p['unique_count']} unique")


if __name__ == "__main__":
    main()
