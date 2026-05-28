"""CX3 Stage 2 — audit the Bool / Option catalog surface.

Goal: find a fresh-namespace, short-tactic family analogous to NS22's
Int/omega — but for Bool/Option (`decide`, `simp`, `cases <;> simp`).

Reads:
  - project/data/cx1_available_theorems.json   (LeanDojo-verified-available)
  - project/discovered_theorems_cx1.json        (source-scan superset)
  - tasks.THEOREM_SETS                           (already-used / already-probed)
  - additional Bool/Option Mathlib source files via regex scan, to
    surface candidates not in the CX1 catalog (catalog mining only —
    LeanDojo availability is decided later by the Stage 4 eval matrix)

Classifies by namespace (Bool / Option) and by likely proof family
using name heuristics (we do not have statement text for every
candidate, but the CX1 catalog carries `family_tags` which we fold in):
  - decide        : Bool props over a finite (Bool) domain
  - simp / simp_all
  - cases / cases_simp  : Option none/some split, then simp
  - aesop
  - constructor
  - iff           : boolean / option equivalences
  - option_map_bind : Option.map / bind / pmap / pbind / getD / elim
  - bool_logic      : Bool.and / or / not / xor / beq / bne / cond / ite

Output:
  - project/data/cx3_bool_option_catalog_audit_meta.json
  - project/evolve/reports/cx3_bool_option_catalog_audit.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


MATHLIB_ROOT = Path(
    "/Users/weijiawang/.cache/lean_dojo/"
    "leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/"
    "mathlib4"
)

# Bool/Option files beyond the two CX1 already covered
# (Bool/Basic.lean, Bool/AllAny.lean, Option/Basic.lean, Option/Defs.lean).
ADDITIONAL_FILES = [
    "Mathlib/Data/Bool/Count.lean",
    "Mathlib/Data/Bool/Set.lean",
    "Mathlib/Data/Option/NAry.lean",
    "Mathlib/Logic/Equiv/Option.lean",
    "Mathlib/Data/List/ReduceOption.lean",
]

_THM_DECL = re.compile(
    r"^\s*(?:protected\s+)?(?:theorem|lemma)\s+"
    r"([A-Za-z_][A-Za-z0-9_'\.]*)\s*"
    r"(?P<rest>[^:]*?):",
    re.MULTILINE,
)
_NS_DECL = re.compile(r"^namespace\s+([A-Za-z_][A-Za-z0-9_'\.]*)", re.MULTILINE)
_SEC_DECL = re.compile(r"^section(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?", re.MULTILINE)
_END_DECL = re.compile(r"^end(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?\s*$", re.MULTILINE)


def scan_lean_file(path: Path) -> list[dict]:
    """Return [{full_name, short_name, namespace, file_path}] for a .lean."""
    text = path.read_text(encoding="utf-8")
    events: list[tuple[int, str, str]] = []
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


def classify(full_name: str, family_tags: list[str]) -> list[str]:
    """Likely proof families from name shape + CX1 family_tags."""
    ns = full_name.split(".")[0]
    short = full_name.split(".")[-1].lower()
    tags: list[str] = []

    if "iff" in short or "iff" in family_tags:
        tags.append("iff")

    if ns == "Bool":
        # Bool props over a finite domain are decidable → `decide`.
        tags.append("bool_decide")
        if re.search(r"(and|or|not|xor|beq|bne|cond|ite)", short):
            tags.append("bool_logic")
        if re.search(r"(tonat|ofnat)", short):
            tags.append("bool_nat_cast")
        if re.search(r"(_le_|_lt_|^le_|^lt_|le$|lt$)", short) or \
                ("le" in family_tags) or ("lt" in family_tags):
            tags.append("bool_order")

    if ns == "Option":
        if re.search(r"(map|bind|pmap|pbind|pmem)", short) or \
                ("map" in family_tags):
            tags.append("option_map_bind")
        if re.search(r"(issome|isnone|getd|orelse|none|some|elim|tolist|"
                     r"guard|join|orelse|coe|iget)", short):
            tags.append("option_simp")
        if re.search(r"(bind|pbind|elim|caseson|rec)", short):
            tags.append("option_cases")
        if "mem" in short or "mem" in family_tags:
            tags.append("option_mem")

    if not tags:
        tags.append("other")
    return tags


def bucket(tags: list[str], ns: str) -> str:
    """Single best-guess minimal-tactic bucket for gate counting."""
    if ns == "Bool":
        return "likely_decide"
    # Option
    if "option_cases" in tags and "option_simp" not in tags:
        return "likely_cases_simp"
    if "option_map_bind" in tags or "option_simp" in tags or "iff" in tags:
        return "likely_simp"
    return "likely_simp"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks

    used: set[str] = set()
    for _set, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            used.add(t.full_name)

    av = json.load(open("project/data/cx1_available_theorems.json"))["theorems"]
    disc = json.load(open("project/discovered_theorems_cx1.json"))["theorems"]

    av_by_name = {t["full_name"]: t for t in av}
    disc_by_name = {t["full_name"]: t for t in disc}
    av_names = set(av_by_name)

    # ---- assemble the candidate universe -------------------------------
    # source priority: available (verified) > discovered > source-scan
    cand: dict[str, dict] = {}

    def add_cand(t: dict, availability: str, source: str) -> None:
        fn = t["full_name"]
        ns = fn.split(".")[0]
        if ns not in ("Bool", "Option"):
            return
        if fn in cand:
            return
        family_tags = t.get("family_tags", [])
        cand[fn] = {
            "full_name": fn,
            "namespace": ns,
            "file_path": t.get("file_path", ""),
            "family_tags": family_tags,
            "difficulty": t.get("difficulty", "?"),
            "has_tactic_proof": t.get("has_tactic_proof", None),
            "availability": availability,   # verified | discovered | scan
            "source": source,
            "already_used": fn in used,
            "cx3_tags": classify(fn, family_tags),
        }
        cand[fn]["bucket"] = bucket(cand[fn]["cx3_tags"], ns)

    for t in av:
        if t["full_name"].startswith(("Bool.", "Option.")):
            add_cand(t, "verified", "cx1_available")
    for t in disc:
        if t["full_name"].startswith(("Bool.", "Option.")):
            availability = "verified" if t["full_name"] in av_names \
                else "discovered"
            add_cand(t, availability, "cx1_discovered")

    file_status: dict[str, int] = {}
    for rel in ADDITIONAL_FILES:
        ap = MATHLIB_ROOT / rel
        if not ap.exists():
            file_status[rel] = -1
            continue
        thms = scan_lean_file(ap)
        n_bo = 0
        for t in thms:
            if t["full_name"].startswith(("Bool.", "Option.")):
                n_bo += 1
                availability = "verified" if t["full_name"] in av_names \
                    else ("discovered" if t["full_name"] in disc_by_name
                          else "scan")
                add_cand(t, availability, f"scan:{rel}")
        file_status[rel] = n_bo

    allc = list(cand.values())

    # ---- partitions ----------------------------------------------------
    def part(pred):
        return [c for c in allc if pred(c)]

    bool_all = part(lambda c: c["namespace"] == "Bool")
    option_all = part(lambda c: c["namespace"] == "Option")
    fresh = part(lambda c: not c["already_used"])
    fresh_verified = part(lambda c: not c["already_used"]
                          and c["availability"] == "verified")
    fresh_needs_probe = part(lambda c: not c["already_used"]
                             and c["availability"] != "verified")

    bucket_counts_fresh = Counter(c["bucket"] for c in fresh)
    tag_counts = Counter()
    for c in allc:
        for tg in c["cx3_tags"]:
            tag_counts[tg] += 1

    def names(lst):
        return sorted(c["full_name"] for c in lst)

    meta = {
        "totals": {
            "bool_candidates": len(bool_all),
            "option_candidates": len(option_all),
            "all_bool_option_candidates": len(allc),
        },
        "already_used_probed": {
            "total": sum(1 for c in allc if c["already_used"]),
            "bool": sum(1 for c in bool_all if c["already_used"]),
            "option": sum(1 for c in option_all if c["already_used"]),
        },
        "fresh_unused": {
            "total": len(fresh),
            "verified_available": len(fresh_verified),
            "needs_probe": len(fresh_needs_probe),
            "bool": sum(1 for c in fresh if c["namespace"] == "Bool"),
            "option": sum(1 for c in fresh if c["namespace"] == "Option"),
        },
        "fresh_bucket_distribution": dict(bucket_counts_fresh.most_common()),
        "tag_distribution_all": dict(tag_counts.most_common()),
        "additional_files_scanned": file_status,
        "fresh_verified_candidates": [
            {"full_name": c["full_name"], "ns": c["namespace"],
             "file": c["file_path"], "tags": c["cx3_tags"],
             "bucket": c["bucket"], "difficulty": c["difficulty"]}
            for c in sorted(fresh_verified, key=lambda c: c["full_name"])
        ],
        "fresh_needs_probe_candidates": [
            {"full_name": c["full_name"], "ns": c["namespace"],
             "file": c["file_path"], "tags": c["cx3_tags"],
             "bucket": c["bucket"], "availability": c["availability"],
             "has_tactic_proof": c["has_tactic_proof"]}
            for c in sorted(fresh_needs_probe, key=lambda c: c["full_name"])
        ],
        "bool_exhaustion_note": (
            "All verified-available Bool theorems from Bool/Basic.lean were "
            "already consumed by cx1_bool_option_int in CX1. Fresh Bool only "
            "appears via additional-file source scans (availability "
            "unverified until Stage 4 eval)."
        ),
    }
    Path("project/data/cx3_bool_option_catalog_audit_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # ---- report --------------------------------------------------------
    md: list[str] = ["# CX3 — Bool / Option catalog audit", ""]
    md.append("Mining a fresh-namespace short-tactic family analogous to "
              "NS22's Int/omega, for Bool/Option (`decide` / `simp` / "
              "`cases <;> simp`). CX3 is **mining-only** — no training.")
    md.append("")
    md.append("## Surface inventory")
    md.append("")
    md.append(f"- Bool candidates: **{len(bool_all)}**")
    md.append(f"- Option candidates: **{len(option_all)}**")
    md.append(f"- Total Bool/Option candidates: **{len(allc)}**")
    md.append(f"- Already used / probed in prior sets: "
              f"**{meta['already_used_probed']['total']}** "
              f"(Bool {meta['already_used_probed']['bool']}, "
              f"Option {meta['already_used_probed']['option']})")
    md.append(f"- **Fresh unused: {len(fresh)}** "
              f"(verified-available {len(fresh_verified)}, "
              f"needs-probe {len(fresh_needs_probe)})")
    md.append(f"  - fresh Bool: {meta['fresh_unused']['bool']}, "
              f"fresh Option: {meta['fresh_unused']['option']}")
    md.append("")
    md.append("> **Bool exhaustion.** Every verified-available Bool theorem "
              "(Bool/Basic.lean) was already consumed by "
              "`cx1_bool_option_int`. The fresh opportunity is **Option** "
              "(map/bind/pmap/pbind/isSome/isNone/getD/elim/orElse). Fresh "
              "Bool only appears via additional-file scans whose LeanDojo "
              "availability is unverified until Stage 4.")
    md.append("")
    md.append("## Additional files scanned")
    md.append("")
    md.append("| file | Bool/Option decls |")
    md.append("|---|---:|")
    for rel, n in file_status.items():
        md.append(f"| {rel} | {'(not found)' if n < 0 else n} |")
    md.append("")
    md.append("## Likely-family buckets (fresh candidates)")
    md.append("")
    md.append("| bucket | count |")
    md.append("|---|---:|")
    for b, n in bucket_counts_fresh.most_common():
        md.append(f"| {b} | {n} |")
    md.append("")
    md.append("## Tag distribution (all candidates)")
    md.append("")
    md.append("| tag | count |")
    md.append("|---|---:|")
    for tg, n in tag_counts.most_common():
        md.append(f"| {tg} | {n} |")
    md.append("")
    md.append("## Fresh verified-available candidates")
    md.append("")
    md.append("| theorem | ns | bucket | tags | file |")
    md.append("|---|---|---|---|---|")
    for c in sorted(fresh_verified, key=lambda c: c["full_name"]):
        md.append(f"| `{c['full_name']}` | {c['namespace']} | {c['bucket']} "
                  f"| {','.join(c['cx3_tags'])} | {c['file_path']} |")
    md.append("")
    md.append("## Fresh needs-probe candidates (source-scan / discovered)")
    md.append("")
    md.append("| theorem | ns | availability | bucket | file |")
    md.append("|---|---|---|---|---|")
    for c in sorted(fresh_needs_probe, key=lambda c: c["full_name"]):
        md.append(f"| `{c['full_name']}` | {c['namespace']} "
                  f"| {c['availability']} | {c['bucket']} | {c['file_path']} |")
    Path("project/evolve/reports/cx3_bool_option_catalog_audit.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print("wrote project/data/cx3_bool_option_catalog_audit_meta.json")
    print("wrote project/evolve/reports/cx3_bool_option_catalog_audit.md")
    print()
    print(f"Bool candidates: {len(bool_all)} | "
          f"Option candidates: {len(option_all)}")
    print(f"already used: {meta['already_used_probed']['total']} | "
          f"fresh unused: {len(fresh)} "
          f"(verified {len(fresh_verified)}, needs-probe "
          f"{len(fresh_needs_probe)})")
    print(f"fresh buckets: {dict(bucket_counts_fresh)}")


if __name__ == "__main__":
    main()
