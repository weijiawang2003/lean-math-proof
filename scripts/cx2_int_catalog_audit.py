"""CX2 Stage 1 — audit Int catalog and extend with more Int files.

Reads:
  - project/data/cx1_available_theorems.json (120 Int theorems from
    Mathlib/Data/Int/{Defs,Bitwise,GCD}.lean already presumed-available
    by CX1 Stage 3 LeanDojo probe)
  - additional Int Mathlib source files via regex scan, to discover
    fresh candidates not in the CX1 catalog (mining the *catalog*
    surface, not yet probing them through LeanDojo — that is Stage 3's
    job via the eval matrix itself)

Classifies candidates by shape using name heuristics:
  - iff_pos: short_name contains 'iff'
  - le_lt_order: short_name has le/lt/order patterns
  - add_sub_arith: arithmetic eq theorems
  - abs_natAbs_sign: abs/sign theorems
  - succ_pred: induction step theorems
  - cast_natCast: cast/conversion theorems
  - dvd_gcd_lcm: number-theory theorems (less iff_omega-friendly)
  - bitwise: bitwise theorems (unlikely iff_omega-friendly)
  - other

Output:
  - project/data/cx2_int_catalog_audit_meta.json
  - project/evolve/reports/cx2_int_catalog_audit.md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MATHLIB_ROOT = Path(
    "/Users/weijiawang/.cache/lean_dojo/"
    "leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/"
    "mathlib4"
)

# Additional Int Mathlib files to scan beyond CX1's 3 (Defs, Bitwise, GCD).
ADDITIONAL_INT_FILES = [
    "Mathlib/Data/Int/ModEq.lean",
    "Mathlib/Data/Int/Order/Lemmas.lean",
    "Mathlib/Data/Int/Order/Units.lean",
    "Mathlib/Data/Int/Lemmas.lean",
    "Mathlib/Data/Int/SuccPred.lean",
    "Mathlib/Data/Int/Cast/Lemmas.lean",
]

# Theorem declaration scanner — same regex as cx1_discover_theorems.py.
_THM_DECL = re.compile(
    r"^\s*(?:protected\s+)?(?:theorem|lemma)\s+"
    r"([A-Za-z_][A-Za-z0-9_'\.]*)\s*"
    r"(?P<rest>[^:]*?):",
    re.MULTILINE,
)
_NS_DECL = re.compile(
    r"^namespace\s+([A-Za-z_][A-Za-z0-9_'\.]*)", re.MULTILINE
)
_SEC_DECL = re.compile(
    r"^section(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?", re.MULTILINE
)
_END_DECL = re.compile(
    r"^end(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?\s*$", re.MULTILINE
)


def scan_int_file(path: Path) -> list[dict]:
    """Return list of {full_name, short_name, file_path} from a .lean file.

    Tracks namespace and section stacks separately; only namespace
    contributes to full_name prefix. `_root_.X` strips the prefix.
    """
    text = path.read_text(encoding="utf-8")
    # Build a chronological event list (namespace push/pop) by offset.
    events: list[tuple[int, str, str]] = []  # (offset, kind, name)
    for m in _NS_DECL.finditer(text):
        events.append((m.start(), "ns_push", m.group(1)))
    for m in _SEC_DECL.finditer(text):
        events.append((m.start(), "sec_push", m.group(1) or ""))
    for m in _END_DECL.finditer(text):
        events.append((m.start(), "end", m.group(1) or ""))
    events.sort()

    def ns_at(off: int) -> str:
        ns_stack: list[str] = []
        sec_stack: list[str] = []
        for eo, kind, name in events:
            if eo > off:
                break
            if kind == "ns_push":
                ns_stack.append(name)
            elif kind == "sec_push":
                sec_stack.append(name)
            elif kind == "end":
                if name and ns_stack and ns_stack[-1] == name:
                    ns_stack.pop()
                elif name and sec_stack and sec_stack[-1] == name:
                    sec_stack.pop()
                elif not name and sec_stack:
                    sec_stack.pop()
                elif not name and ns_stack:
                    ns_stack.pop()
        return ".".join(ns_stack)

    out: list[dict] = []
    for m in _THM_DECL.finditer(text):
        short = m.group(1)
        off = m.start()
        ns = ns_at(off)
        if short.startswith("_root_."):
            full = short[len("_root_."):]
            ns_eff = ""
        else:
            full = f"{ns}.{short}" if ns else short
            ns_eff = ns
        out.append({
            "full_name": full,
            "short_name": short.split(".")[-1],
            "namespace": ns_eff,
            "file_path": str(path.relative_to(MATHLIB_ROOT)),
        })
    return out


def classify(short: str) -> list[str]:
    s = short.lower()
    tags: list[str] = []
    if "iff" in s:
        tags.append("iff_candidate")
    if re.search(r"(_le_|_lt_|^le_|^lt_|_le$|_lt$|le_iff|lt_iff|"
                 r"^order_|order_iff)", s):
        tags.append("le_lt_order")
    if re.search(r"(_add_|_sub_|_mul_|_neg_|^add_|^sub_|^mul_|^neg_)", s):
        tags.append("add_sub_arith")
    if re.search(r"(abs|sign|natabs)", s):
        tags.append("abs_natAbs_sign")
    if re.search(r"(_succ|^succ|_pred|^pred|induction)", s):
        tags.append("succ_pred")
    if re.search(r"(natcast|ofnat|_cast_|cast_)", s):
        tags.append("cast_natCast")
    if re.search(r"(_dvd|^dvd|gcd|lcm|^prime|coprime)", s):
        tags.append("dvd_gcd_lcm")
    if re.search(r"(bit|bodd|bitwise|land|lor|lxor|ldiff|shift|testbit)", s):
        tags.append("bitwise")
    if re.search(r"(_mod|^mod|emod|^div|_div|ediv)", s):
        tags.append("mod_div")
    if not tags:
        tags.append("other")
    return tags


def main() -> None:
    cx1 = json.load(open("project/data/cx1_available_theorems.json"))
    cx1_ints = [t for t in cx1["theorems"]
                if t.get("full_name", "").startswith("Int.")]
    cx1_int_names = {t["full_name"] for t in cx1_ints}

    # Augment with new source files.
    additional: list[dict] = []
    file_status: dict[str, int] = {}
    for rel in ADDITIONAL_INT_FILES:
        ap = MATHLIB_ROOT / rel
        if not ap.exists():
            file_status[rel] = -1
            continue
        thms = scan_int_file(ap)
        file_status[rel] = len(thms)
        for t in thms:
            if (t["full_name"].startswith("Int.")
                    and t["full_name"] not in cx1_int_names):
                additional.append(t)

    # Classify all.
    cx1_classed = []
    for t in cx1_ints:
        tags = classify(t["short_name"])
        cx1_classed.append({**t, "cx2_tags": tags, "source": "cx1_catalog"})
    add_classed = []
    seen = set()
    for t in additional:
        if t["full_name"] in seen:
            continue
        seen.add(t["full_name"])
        tags = classify(t["short_name"])
        add_classed.append({**t, "cx2_tags": tags, "source": "cx2_scan"})

    all_classed = cx1_classed + add_classed

    # CX1 probe wrapper-only-vs-NS9 Int wins (already known wrapper-only).
    known_wins = {"Int.le_add_one_iff", "Int.le_iff_lt_or_eq",
                  "Int.emod_two_eq_zero_or_one"}

    # Tag distribution.
    tag_counts: Counter = Counter()
    for t in all_classed:
        for tag in t["cx2_tags"]:
            tag_counts[tag] += 1

    # Likely iff_omega candidates: iff_candidate + (le_lt_order OR
    # short_name contains "le"|"lt"|"pos"|"nonneg"|"neg"|"eq"|"or"|
    # "succ_lt"|"pred_lt").
    iff_omega_candidates = [
        t for t in all_classed
        if "iff_candidate" in t["cx2_tags"]
        and any(tag in t["cx2_tags"]
                for tag in ("le_lt_order", "add_sub_arith", "succ_pred"))
        and t["full_name"] not in known_wins
        and "bitwise" not in t["cx2_tags"]
        and "dvd_gcd_lcm" not in t["cx2_tags"]
    ]

    # Likely omega-only (non-iff) candidates: order/add_sub arithmetic
    # without iff, non-bitwise.
    omega_only = [
        t for t in all_classed
        if "iff_candidate" not in t["cx2_tags"]
        and (("le_lt_order" in t["cx2_tags"])
             or ("add_sub_arith" in t["cx2_tags"]))
        and "bitwise" not in t["cx2_tags"]
        and "dvd_gcd_lcm" not in t["cx2_tags"]
        and t["full_name"] not in known_wins
    ]

    # Cast theorems (likely norm_cast-closeable but worth probing).
    cast_candidates = [
        t for t in all_classed
        if "cast_natCast" in t["cx2_tags"]
        and t["full_name"] not in known_wins
    ]

    meta = {
        "cx1_int_count": len(cx1_ints),
        "cx1_additional_files_scanned": file_status,
        "cx2_additional_int_candidates": len(add_classed),
        "total_int_candidates": len(all_classed),
        "known_wrapper_only_wins": sorted(known_wins),
        "tag_distribution": dict(tag_counts.most_common()),
        "iff_omega_candidate_count": len(iff_omega_candidates),
        "omega_only_candidate_count": len(omega_only),
        "cast_candidate_count": len(cast_candidates),
        "iff_omega_candidates": [
            {"full_name": t["full_name"], "file": t["file_path"],
             "tags": t["cx2_tags"], "source": t["source"]}
            for t in iff_omega_candidates
        ],
        "omega_only_candidates": [
            {"full_name": t["full_name"], "file": t["file_path"],
             "tags": t["cx2_tags"], "source": t["source"]}
            for t in omega_only
        ],
        "cast_candidates_top20": [
            {"full_name": t["full_name"], "file": t["file_path"],
             "tags": t["cx2_tags"], "source": t["source"]}
            for t in cast_candidates[:20]
        ],
    }
    Path("project/data/cx2_int_catalog_audit_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    md = ["# CX2 — Int catalog audit", ""]
    md.append("## Surface inventory")
    md.append("")
    md.append(f"- CX1 catalog (LeanDojo-verified available): "
              f"**{len(cx1_ints)}** Int theorems")
    md.append(f"- CX2 additional source-scan candidates: "
              f"**{len(add_classed)}** fresh Int theorems")
    md.append(f"- **Total Int candidates: {len(all_classed)}**")
    md.append("")
    md.append("Additional files scanned:")
    md.append("")
    md.append("| file | theorems extracted |")
    md.append("|---|---:|")
    for rel, n in file_status.items():
        if n < 0:
            md.append(f"| {rel} | (not found) |")
        else:
            md.append(f"| {rel} | {n} |")
    md.append("")
    md.append("## Tag distribution")
    md.append("")
    md.append("| tag | count |")
    md.append("|---|---:|")
    for tag, n in tag_counts.most_common():
        md.append(f"| {tag} | {n} |")
    md.append("")
    md.append("## Pool-mining buckets")
    md.append("")
    md.append(f"- **iff_omega_pair candidates: "
              f"{len(iff_omega_candidates)}** "
              "(iff + le/lt/add/sub, non-bitwise, non-dvd)")
    md.append(f"- omega-only candidates: {len(omega_only)} "
              "(le/lt/add/sub without iff, non-bitwise, non-dvd)")
    md.append(f"- cast/natCast candidates: {len(cast_candidates)} "
              "(probed for norm_cast → omega)")
    md.append("")
    md.append("Known CX1 wrapper-only-vs-NS9 wins (excluded from mining):")
    md.append("")
    for n in sorted(known_wins):
        md.append(f"- `{n}`")
    md.append("")
    md.append("## iff_omega candidate sample (all listed)")
    md.append("")
    md.append("| theorem | file | source |")
    md.append("|---|---|---|")
    for t in iff_omega_candidates:
        md.append(f"| `{t['full_name']}` | {t['file_path']} | "
                  f"{t['source']} |")
    Path("project/evolve/reports/cx2_int_catalog_audit.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"wrote project/data/cx2_int_catalog_audit_meta.json")
    print(f"wrote project/evolve/reports/cx2_int_catalog_audit.md")
    print(f"\ntotal Int candidates: {len(all_classed)} "
          f"(CX1: {len(cx1_ints)}, CX2 fresh: {len(add_classed)})")
    print(f"iff_omega candidates: {len(iff_omega_candidates)}")
    print(f"omega-only candidates: {len(omega_only)}")
    print(f"cast candidates: {len(cast_candidates)}")


if __name__ == "__main__":
    main()
