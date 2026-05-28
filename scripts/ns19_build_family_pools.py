"""NS19 Stage 5 — family-pool metadata.

Aggregates per-family training-readiness counts across NS18 + NS19,
emitting project/data/ns19_family_pool_meta.json.

For each family (currently the two NS19 target families plus a
catch-all for any other family that produced wrapper-only signal),
record:
  - family name
  - rows (count of (state, tactic) candidate pairs we could harvest)
  - unique theorems
  - wrapper-only theorems (rows that the raw model could not prove)
  - tactic strings (sample, deduped)
  - source theorem sets
  - raw_already_solved count (just informational)
  - regression count (theorems lost vs NS9 wrapper, if any)
  - trainable: bool — true when ≥5 wrapper-only theorems OR ≥10
    usable rows from ≥5 unique wrapper-only thms
  - recommended_oversample_factor

Does NOT build the training JSONL. Per the NS19 spec, JSONL
production is deferred to NS20 — only if the gate is met.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

NS18_META = Path("project/data/ns18_wrapper_signal_meta.json")
NS19_META = Path("project/data/ns19_wrapper_signal_meta.json")
OUT_PATH = Path("project/data/ns19_family_pool_meta.json")


def _first_match(pattern: str) -> str | None:
    ms = sorted(glob.glob(pattern))
    return ms[0] if ms else None


def _load_solved(path: str | None) -> set[str]:
    if not path:
        return set()
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", []) if t.get("finished")}


def _wrap_baseline_path(set_name: str) -> str | None:
    cands = []
    if set_name.startswith("ns17_"):
        cands.append(f"project/evolve/eval_runs/ns17_ns15routed_wrapper_{set_name}/eval-*/metrics.json")
    if set_name.startswith("ns16_"):
        cands.append(f"project/evolve/eval_runs/ns16_ns15routed_wrapper_{set_name}/eval-*/metrics.json")
    cands.append(f"project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_{set_name}/eval-*/metrics.json")
    cands.append(f"project/evolve/eval_runs/ns19_ns9wrap_{set_name}/eval-*/metrics.json")
    for p in cands:
        m = _first_match(p)
        if m:
            return m
    return None


def _ns18_variant_path(variant: str, set_name: str) -> str | None:
    # NS18 used the dir name pattern ns18_<variant>_wrapper_<set>.
    return _first_match(
        f"project/evolve/eval_runs/ns18_{variant}_wrapper_{set_name}/eval-*/metrics.json"
    )

# Families NS19 specifically targets. NS18 labelled bare `aesop` as
# `fallback_aesop`; this script labels it `aesop`. We accept both
# labels and normalize to `aesop` so the NS18 meta blends cleanly.
TARGET_FAMILIES = ("aesop", "simp_all")
FAMILY_ALIASES = {"fallback_aesop": "aesop"}


def family_of(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t: return "empty"
    if t == "omega": return "fallback_omega"
    if t == "aesop": return "aesop"
    if t == "decide": return "fallback_decide"
    if t == "rfl": return "fallback_rfl"
    if t.startswith("constructor") and "omega" in t:
        return "constructor_omega"
    if t.startswith("split_ifs"):
        return "split_ifs"
    if "fun h => by omega" in t and t.count("by omega") >= 2:
        return "iff_omega_pair"
    if t.startswith("simp_all"):
        return "simp_all"
    if t.startswith("simp"):
        return "simp_other"
    if t.startswith("rw"):
        return "rw_named"
    if t.startswith("exact"):
        return "exact_named"
    if t.startswith("apply"):
        return "apply_named"
    return "other"


def collect() -> dict:
    """Walk per-theorem entries in NS18 + NS19 meta files and group
    wrapper-only-new wins by their winning tactic family."""
    pool: dict[str, dict] = defaultdict(lambda: {
        "rows": 0,
        "wrapper_only_thms": set(),
        "tactic_strings": set(),
        "sources": defaultdict(int),
        "namespace_breakdown": defaultdict(int),
    })

    def _ingest_ns18(meta_path: Path) -> None:
        """NS18 meta has new_wins_by_family scoped to wrapper-only-vs-raw,
        which over-counts wins NS9 wrapper already had. Refilter against
        the NS9 wrap baseline (variant_solved − wrap_solved)."""
        if not meta_path.exists():
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for variant, sets in data.items():
            if not isinstance(sets, dict):
                continue
            for set_name, row in sets.items():
                if not isinstance(row, dict):
                    continue
                # Drop sets where NS18 variant didn't strictly improve
                # over NS9 wrap. The wrapper_only_new wins ARE wrapper
                # contributions vs raw, but if NS9 already proved them
                # they don't count for NS20 training (the model already
                # has them via wrapper).
                vsolv = row.get("variant_solved", 0)
                wsolv = row.get("wrap_solved", 0)
                if vsolv <= wsolv:
                    continue
                # Re-derive the truly-new set: variant proved minus wrap
                # proved. Need to re-read metrics for the actual names.
                vp = _ns18_variant_path(variant, set_name)
                wp = _wrap_baseline_path(set_name)
                if not vp or not wp:
                    continue
                vset = _load_solved(vp)
                wset = _load_solved(wp)
                truly_new = sorted(vset - wset)
                if not truly_new:
                    continue
                # Re-classify the winning tactics on truly-new entries.
                v_per_thm = {
                    t["full_name"]: t for t in
                    json.loads(Path(vp).read_text(encoding="utf-8")).get("per_theorem", [])
                }
                for thm in truly_new:
                    tac = (
                        (v_per_thm.get(thm) or {}).get("winning_tactic")
                        or (v_per_thm.get(thm) or {}).get("last_tactic")
                        or (v_per_thm.get(thm) or {}).get("tactic")
                        or ""
                    )
                    fam = family_of(tac) if tac else "unknown"
                    fam_norm = FAMILY_ALIASES.get(fam, fam)
                    pool[fam_norm]["rows"] += 1
                    pool[fam_norm]["wrapper_only_thms"].add(thm)
                    if tac:
                        pool[fam_norm]["tactic_strings"].add(tac)
                    pool[fam_norm]["sources"][f"NS18:{variant}:{set_name}"] += 1
                    ns = thm.split(".", 1)[0] if "." in thm else "?"
                    pool[fam_norm]["namespace_breakdown"][ns] += 1

    def _ingest_ns19(meta_path: Path) -> None:
        """NS19 meta already carries new_wins_by_family scoped to
        new_vs_wrap, so we can ingest directly."""
        if not meta_path.exists():
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for variant, sets in data.items():
            if not isinstance(sets, dict):
                continue
            for set_name, row in sets.items():
                if not isinstance(row, dict):
                    continue
                fam_blob = row.get("new_wins_by_family") or {}
                for fam, thms in fam_blob.items():
                    fam_norm = FAMILY_ALIASES.get(fam, fam)
                    for thm in thms:
                        pool[fam_norm]["rows"] += 1
                        pool[fam_norm]["wrapper_only_thms"].add(thm)
                        pool[fam_norm]["sources"][f"NS19:{variant}:{set_name}"] += 1
                        ns = thm.split(".", 1)[0] if "." in thm else "?"
                        pool[fam_norm]["namespace_breakdown"][ns] += 1

    _ingest_ns18(NS18_META)
    _ingest_ns19(NS19_META)

    out: dict = {"families": {}, "targets": list(TARGET_FAMILIES)}
    for fam, data in pool.items():
        unique = len(data["wrapper_only_thms"])
        # NS19 gate: ≥5 wrapper-only unique theorems OR
        # ≥10 rows from ≥5 unique wrapper-only theorems.
        trainable_by_count = unique >= 5
        trainable_by_rows = data["rows"] >= 10 and unique >= 5
        trainable = trainable_by_count or trainable_by_rows
        # Recommended oversample factor — capped at 20x, scaled by
        # pool size so big pools see 1-2x and tiny pools see 10-20x.
        if unique <= 1:
            os_f = 20
        elif unique <= 3:
            os_f = 15
        elif unique <= 6:
            os_f = 10
        elif unique <= 12:
            os_f = 5
        else:
            os_f = 2
        out["families"][fam] = {
            "family": fam,
            "rows": data["rows"],
            "unique_theorems": unique,
            "wrapper_only_theorems": sorted(data["wrapper_only_thms"]),
            "tactic_strings_sample": sorted(data["tactic_strings"])[:10],
            "source_breakdown": dict(data["sources"]),
            "namespace_breakdown": dict(data["namespace_breakdown"]),
            "trainable": trainable,
            "trainable_by_count": trainable_by_count,
            "trainable_by_rows": trainable_by_rows,
            "recommended_oversample_factor": os_f,
        }
    # Sort families by trainable-then-unique-count desc.
    out["families"] = dict(sorted(
        out["families"].items(),
        key=lambda kv: (
            not kv[1]["trainable"], -kv[1]["unique_theorems"]
        ),
    ))
    return out


def main() -> None:
    pool = collect()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"\nfamily-pool summary (NS18 + NS19):")
    for fam, info in pool["families"].items():
        mark = "TRAIN" if info["trainable"] else "  --"
        print(f"  [{mark}] {fam:>22} | unique={info['unique_theorems']:>2} "
              f"rows={info['rows']:>2} | os_factor={info['recommended_oversample_factor']:>2}")
    targets = pool["targets"]
    print(f"\nTarget families: {targets}")
    for fam in targets:
        info = pool["families"].get(fam)
        if not info:
            print(f"  {fam}: no signal")
            continue
        gate_ok = info["trainable"]
        print(f"  {fam}: {info['unique_theorems']} unique → "
              f"{'GATE MET' if gate_ok else 'gate not met'}")


if __name__ == "__main__":
    main()
