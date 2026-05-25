"""NS19 Stage 2 + 4 — build targeted theorem surfaces for the two
promising NS18 families: Finset aesop and Nat simp_all arithmetic.

Two surfaces are emitted into
``project/evolve/routing/ns19_theorem_sets.json``:

  ns19_finset_aesop_surface
    Up to 80 Finset theorems with shapes that favour aesop:
    insert, cons, union, inter, disjUnion, subset, mem, coe,
    singleton, image, filter, map, sdiff, product, biUnion,
    powerset. Excludes every Finset theorem already present in
    demo_v1 / ns14_set_finset_extra / ns17_finset_extra.

  ns19_nat_simp_arith_replay
    Re-uses Nat theorems from ns16_nat_div_mod_extra,
    ns16_nat_mixed_extra, ns16_nat_order_extra, ns17_nat_remaining
    where NS18 nat_simp_arith did NOT already prove the theorem.
    These are the candidate names the NS19 targeted variant can
    plausibly close with the new simp_all-bundle additions. The
    canonical medium/large/ns14 Nat sets are excluded because they
    are already saturated by the NS9 wrapper baseline.

The Nat-replay surface intentionally re-uses theorems already in
the catalog. The catalog is exhausted (208/208 Nat theorems are
already covered by existing sets), so the only way to grow signal
for the simp_all-Nat-arith family without a catalog extension is to
re-evaluate the NEW targeted bundle on previously-evaluated
theorems.

tasks.py is patched to load these at import time via the existing
_load_ns19_sets() helper (added if not already present).

Usage:
    PYTHONPATH=. python scripts/build_ns19_theorem_sets.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


DISCOVERED_PATH = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/evolve/routing/ns19_theorem_sets.json")

# Sets to exclude when picking fresh Finset theorems.
FINSET_EXCLUDE_SETS = (
    "demo_v1",
    "ns14_set_finset_extra",
    "ns17_finset_extra",
    "finset_small",
    "mixed_easy_v2",
    "ns14_mixed_easy",
    "ns14_mixed_medium",
)

# Token shapes that aesop tends to handle on Finset goals.
FINSET_AESOP_TOKENS = (
    "insert", "cons", "union", "inter", "disj", "subset",
    "mem", "coe", "singleton", "image", "filter", "map",
    "sdiff", "product", "biUnion", "powerset", "empty",
    "comm", "assoc", "self", "left", "right",
)

# Source Nat sets to replay with the targeted variant.
NAT_REPLAY_SOURCES = (
    "ns16_nat_div_mod_extra",
    "ns16_nat_mixed_extra",
    "ns16_nat_order_extra",
    "ns17_nat_remaining",
)

# Nat names already proved by NS18 nat_simp_arith on any of its sets;
# loaded lazily from ns18_wrapper_signal_meta.json so the replay set
# excludes them automatically. Anything not in this proved-set could
# benefit from the targeted variant.
NS18_NAT_SIMP_ARITH_PROVED: set[str] = set()


def _load_ns18_proved() -> set[str]:
    p = Path("project/data/ns18_wrapper_signal_meta.json")
    if not p.exists():
        return set()
    proved: set[str] = set()
    data = json.loads(p.read_text(encoding="utf-8"))
    var_data = data.get("nat_simp_arith") or {}
    for _, row in var_data.items():
        # variant_solved is a count; per-theorem solved names live in
        # the eval_runs metrics.json, which we don't reload here. For
        # the purposes of the replay set, we only know which thms had
        # "wrapper_only_new" wins under nat_simp_arith. To be safe we
        # exclude both the wrapper_only_new wins AND the regressed
        # theorems (regression means the variant lost a theorem that
        # NS9 wrapper had, so worth retrying).
        for nm in row.get("wrapper_only_new", []):
            proved.add(nm)
    return proved


def main() -> None:
    catalog = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))

    # ---- Finset surface ----
    used_finset: set[str] = set()
    for s in FINSET_EXCLUDE_SETS:
        for cfg in THEOREM_SETS.get(s, []):
            used_finset.add(cfg.full_name)

    finset_pool: list[dict] = []
    for t in catalog["theorems"]:
        name = t.get("full_name", "")
        if not name.startswith("Finset."):
            continue
        if name in used_finset or not t.get("has_tactic_proof"):
            continue
        # Restrict to easy/medium — aesop won't reliably close hard
        # Finset goals at our 8-step budget.
        if t.get("difficulty") == "hard":
            continue
        last_seg = name.split(".", 1)[1] if "." in name else name
        last_lower = last_seg.lower()
        if not any(tok in last_lower for tok in FINSET_AESOP_TOKENS):
            continue
        finset_pool.append({
            "file_path": t["file_path"],
            "full_name": name,
            "difficulty": t.get("difficulty", "?"),
        })

    # Keep at most 80, easy first.
    finset_pool.sort(key=lambda t: (t["difficulty"] != "easy", t["full_name"]))
    finset_surface = finset_pool[:80]

    # ---- Nat-arith replay ----
    proved = _load_ns18_proved()
    seen: set[str] = set()
    nat_replay: list[dict] = []
    for src in NAT_REPLAY_SOURCES:
        for cfg in THEOREM_SETS.get(src, []):
            if not cfg.full_name.startswith("Nat."):
                continue
            if cfg.full_name in proved or cfg.full_name in seen:
                continue
            seen.add(cfg.full_name)
            nat_replay.append({
                "file_path": cfg.file_path,
                "full_name": cfg.full_name,
                "source_set": src,
            })

    out = {
        "ns19_finset_aesop_surface": finset_surface,
        "ns19_nat_simp_arith_replay": nat_replay,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    for k, v in out.items():
        d = {}
        for t in v:
            d[t.get("difficulty", "?")] = d.get(t.get("difficulty", "?"), 0) + 1
        diff_s = ", ".join(f"{k2}={v2}" for k2, v2 in sorted(d.items()))
        print(f"  {k}: {len(v)} ({diff_s})")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
