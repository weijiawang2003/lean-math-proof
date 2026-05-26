"""CX1 Stage 4 — build CX1 theorem sets.

Reads project/data/cx1_available_theorems.json (from the
Stage 3 availability probe) and project/discovered_theorems_cx1.json
(for family-tag metadata) and partitions the available theorems
into six sets covering the CX1 target namespaces:

  cx1_finset_image_filter   — Finset image/filter/map/card/erase/attach
  cx1_nat_gcd_dvd_mod       — Nat gcd/dvd/mod/div/coprime
  cx1_list_multiset         — List + Multiset
  cx1_bool_option_int       — Bool + Option + Int (entirely new namespaces)
  cx1_mixed_easy            — easy difficulty across namespaces
  cx1_mixed_medium          — medium difficulty across namespaces

Sets are sized at 50-100 theorems each (per the CX1 spec). Theorems
already in any prior theorem set (NS9 / NS14 / NS15 / NS16 / NS17 /
NS19 / NS20) are excluded.

Output: project/evolve/routing/cx1_theorem_sets.json. tasks.py is
patched with _load_cx1_sets().
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


AVAIL_PATH = Path("project/data/cx1_available_theorems.json")
DISCOVERED_PATH = Path("project/discovered_theorems_cx1.json")
OUT_PATH = Path("project/evolve/routing/cx1_theorem_sets.json")

# Size caps per set (per CX1 spec).
CAPS = {
    "cx1_finset_image_filter": 100,
    "cx1_nat_gcd_dvd_mod": 100,
    "cx1_list_multiset": 100,
    "cx1_bool_option_int": 80,
    "cx1_mixed_easy": 50,
    "cx1_mixed_medium": 50,
}


def _already_used() -> set[str]:
    used: set[str] = set()
    for cfgs in THEOREM_SETS.values():
        for c in cfgs:
            used.add(c.full_name)
    return used


def _has_token(name: str, tokens: tuple[str, ...]) -> bool:
    if "." not in name:
        return False
    last = name.split(".", 1)[1].lower()
    return any(tok in last for tok in tokens)


def main() -> None:
    if not AVAIL_PATH.exists():
        print(f"ERROR: {AVAIL_PATH} missing — run Stage 3 first.", file=sys.stderr)
        sys.exit(1)

    avail = json.loads(AVAIL_PATH.read_text(encoding="utf-8"))
    discovered = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))
    # Build a lookup of discovered metadata (difficulty, family_tags).
    disc_by_name: dict[str, dict] = {t["full_name"]: t for t in discovered["theorems"]}
    used = _already_used()

    # Candidate pool: available theorems not already used.
    pool: list[dict] = []
    for t in avail.get("theorems", []):
        nm = t["full_name"]
        if nm in used:
            continue
        # Attach difficulty + family_tags from the discovered record.
        meta = disc_by_name.get(nm) or {}
        pool.append({
            "file_path": t["file_path"],
            "full_name": nm,
            "difficulty": meta.get("difficulty", t.get("difficulty", "?")),
            "family_tags": meta.get("family_tags", []),
            "namespace": meta.get("namespace") or (nm.split(".", 1)[0] if "." in nm else ""),
        })

    print(f"available pool (excluding already-used): {len(pool)}")

    # --- Bucket A: Finset image/filter/map/card/erase/attach ---
    finset_tokens = ("image", "filter", "map", "card", "erase", "attach",
                     "biunion", "powerset")
    bucket_a = sorted(
        [t for t in pool
         if t["full_name"].startswith("Finset.")
         and _has_token(t["full_name"], finset_tokens)],
        key=lambda t: (t["difficulty"] != "easy", t["full_name"]),
    )[:CAPS["cx1_finset_image_filter"]]
    used_a = {t["full_name"] for t in bucket_a}

    # --- Bucket B: Nat gcd/dvd/mod/div/coprime ---
    nat_tokens = ("gcd", "dvd", "mod", "div", "coprime", "lcm")
    bucket_b = sorted(
        [t for t in pool
         if t["full_name"].startswith("Nat.")
         and _has_token(t["full_name"], nat_tokens)
         and t["full_name"] not in used_a],
        key=lambda t: (t["difficulty"] != "easy", t["full_name"]),
    )[:CAPS["cx1_nat_gcd_dvd_mod"]]
    used_b = used_a | {t["full_name"] for t in bucket_b}

    # --- Bucket C: List + Multiset ---
    bucket_c = sorted(
        [t for t in pool
         if (t["full_name"].startswith("List.") or t["full_name"].startswith("Multiset."))
         and t["full_name"] not in used_b],
        key=lambda t: (t["difficulty"] != "easy", t["full_name"]),
    )[:CAPS["cx1_list_multiset"]]
    used_c = used_b | {t["full_name"] for t in bucket_c}

    # --- Bucket D: Bool + Option + Int ---
    bucket_d = sorted(
        [t for t in pool
         if any(t["full_name"].startswith(p) for p in ("Bool.", "Option.", "Int."))
         and t["full_name"] not in used_c],
        key=lambda t: (t["difficulty"] != "easy", t["full_name"]),
    )[:CAPS["cx1_bool_option_int"]]
    used_d = used_c | {t["full_name"] for t in bucket_d}

    # --- Bucket E: cx1_mixed_easy (easy across namespaces, mixed) ---
    bucket_e_pool = sorted(
        [t for t in pool
         if t["difficulty"] == "easy" and t["full_name"] not in used_d],
        key=lambda t: t["full_name"],
    )
    # Stratify: take a roughly equal slice per namespace.
    by_ns_e: dict[str, list[dict]] = defaultdict(list)
    for t in bucket_e_pool:
        by_ns_e[t["namespace"] or "_no_ns_"].append(t)
    bucket_e: list[dict] = []
    cap_e = CAPS["cx1_mixed_easy"]
    while len(bucket_e) < cap_e and any(by_ns_e.values()):
        for ns in list(by_ns_e):
            if not by_ns_e[ns]:
                continue
            bucket_e.append(by_ns_e[ns].pop(0))
            if len(bucket_e) >= cap_e:
                break
    used_e = used_d | {t["full_name"] for t in bucket_e}

    # --- Bucket F: cx1_mixed_medium (medium across namespaces) ---
    bucket_f_pool = sorted(
        [t for t in pool
         if t["difficulty"] == "medium" and t["full_name"] not in used_e],
        key=lambda t: t["full_name"],
    )
    by_ns_f: dict[str, list[dict]] = defaultdict(list)
    for t in bucket_f_pool:
        by_ns_f[t["namespace"] or "_no_ns_"].append(t)
    bucket_f: list[dict] = []
    cap_f = CAPS["cx1_mixed_medium"]
    while len(bucket_f) < cap_f and any(by_ns_f.values()):
        for ns in list(by_ns_f):
            if not by_ns_f[ns]:
                continue
            bucket_f.append(by_ns_f[ns].pop(0))
            if len(bucket_f) >= cap_f:
                break

    out = {
        "cx1_finset_image_filter": bucket_a,
        "cx1_nat_gcd_dvd_mod": bucket_b,
        "cx1_list_multiset": bucket_c,
        "cx1_bool_option_int": bucket_d,
        "cx1_mixed_easy": bucket_e,
        "cx1_mixed_medium": bucket_f,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    for k, v in out.items():
        ns_break = Counter(t["namespace"] or "_no_ns_" for t in v)
        diff_break = Counter(t["difficulty"] for t in v)
        ns_s = ", ".join(f"{n}={c}" for n, c in ns_break.most_common(4))
        d_s = ", ".join(f"{d}={c}" for d, c in diff_break.most_common(3))
        print(f"  {k}: {len(v)} ({ns_s} | {d_s})")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
