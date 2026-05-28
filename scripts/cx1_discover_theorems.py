"""CX1 Stage 2 — discover more Mathlib theorems by source scanning.

Reads the locally-cached Mathlib source tree at
~/.cache/lean_dojo/leanprover-community-mathlib4-<commit>/mathlib4/
and extracts every `theorem`/`lemma` declaration from a curated list
of extension files. Output schema matches
project/discovered_theorems.json so the result can later be merged
or used independently.

This is a regex source-scanner — it does NOT invoke LeanDojo /
Lean. It tags has_tactic_proof heuristically (presence of `by`
in the proof body) and a rough difficulty by counting tactic
tokens. Availability against the live Dojo environment is the
job of Stage 3.

Output: project/discovered_theorems_cx1.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

OLD_CATALOG = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/discovered_theorems_cx1.json")

MATHLIB_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
MATHLIB_ROOT = (
    Path.home() / ".cache" / "lean_dojo"
    / f"leanprover-community-mathlib4-{MATHLIB_COMMIT}"
    / "mathlib4"
)

# Target extension files — chosen to cover the CX1 target namespaces.
TARGET_FILES = [
    # Finset (image/filter/card/lattice surface)
    "Mathlib/Data/Finset/Image.lean",
    "Mathlib/Data/Finset/Card.lean",
    "Mathlib/Data/Finset/Lattice.lean",
    "Mathlib/Data/Finset/Powerset.lean",
    "Mathlib/Data/Finset/Order.lean",
    "Mathlib/Data/Finset/Pi.lean",
    "Mathlib/Data/Finset/Sigma.lean",
    "Mathlib/Data/Finset/Fold.lean",
    "Mathlib/Data/Finset/Preimage.lean",
    "Mathlib/Data/Finset/Union.lean",
    "Mathlib/Data/Finset/Attr.lean",
    # Nat (gcd / dvd / order / basic)
    "Mathlib/Data/Nat/Basic.lean",
    "Mathlib/Data/Nat/GCD/Basic.lean",
    "Mathlib/Data/Nat/ModEq.lean",
    "Mathlib/Data/Nat/Bits.lean",
    "Mathlib/Data/Nat/Bitwise.lean",
    "Mathlib/Data/Nat/Count.lean",
    "Mathlib/Data/Nat/Dist.lean",
    "Mathlib/Data/Nat/EvenOddRec.lean",
    "Mathlib/Data/Nat/Log.lean",
    "Mathlib/Data/Nat/Pow.lean",
    "Mathlib/Data/Nat/Size.lean",
    # List
    "Mathlib/Data/List/Basic.lean",
    "Mathlib/Data/List/Defs.lean",
    "Mathlib/Data/List/Count.lean",
    "Mathlib/Data/List/Chain.lean",
    "Mathlib/Data/List/Pairwise.lean",
    "Mathlib/Data/List/Dedup.lean",
    "Mathlib/Data/List/Range.lean",
    # Multiset
    "Mathlib/Data/Multiset/Basic.lean",
    "Mathlib/Data/Multiset/Bind.lean",
    "Mathlib/Data/Multiset/Dedup.lean",
    "Mathlib/Data/Multiset/Lattice.lean",
    # Set extension
    "Mathlib/Data/Set/Function.lean",
    "Mathlib/Data/Set/Lattice.lean",
    "Mathlib/Data/Set/Finite.lean",
    # Bool / Option / Int
    "Mathlib/Data/Bool/Basic.lean",
    "Mathlib/Data/Bool/AllAny.lean",
    "Mathlib/Data/Bool/Count.lean",
    "Mathlib/Data/Option/Basic.lean",
    "Mathlib/Data/Option/Defs.lean",
    "Mathlib/Data/Option/NAry.lean",
    "Mathlib/Data/Int/Defs.lean",
    "Mathlib/Data/Int/Basic.lean",
    "Mathlib/Data/Int/GCD.lean",
    "Mathlib/Data/Int/Bitwise.lean",
    # Logic / Order fundamentals (small surfaces)
    "Mathlib/Logic/Basic.lean",
    "Mathlib/Order/Basic.lean",
]

# Family-tag substrings to attach to each theorem.
FAMILY_TOKENS = (
    "image", "filter", "map", "card", "mem", "subset", "mod",
    "gcd", "dvd", "div", "iff", "eq", "le", "lt", "comm",
    "assoc", "self", "empty", "insert", "cons", "union",
    "inter", "diff", "singleton", "powerset", "erase",
    "image_", "filter_", "biUnion", "attach", "fold", "prod",
    "sum", "max", "min",
)


# Theorem-declaration regex. Matches the start of:
#   theorem foo : ... := ...
#   lemma foo : ... := ...
#   theorem foo (x : Nat) : ...
#   theorem foo {α : Type*} : ...
#   protected theorem foo : ...
#   private theorem foo : ...
#   @[simp] theorem foo : ...
# but NOT:
#   def foo
#   theorem.foo
DECL_RE = re.compile(
    r"^(?:@\[[^\]]+\]\s*)?"
    r"(?:protected\s+|private\s+|noncomputable\s+)?"
    r"(theorem|lemma)\s+"
    r"([A-Za-z_][A-Za-z0-9_'\.]*)",
    re.MULTILINE,
)


# Lean comment stripper. Removes /- ... -/ and -- ... .
def _strip_comments(src: str) -> str:
    # Remove block comments (non-nested approximation; sufficient
    # for Mathlib source).
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    # Remove line comments.
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Namespace tracker. Critical Lean detail: ONLY `namespace X` adds
# `X.` to a theorem's full_name. `section X` is a variable-scoping
# construct and does NOT add to the full_name. `end X` ends EITHER
# a namespace or a section that started with the same X. We track
# both a namespace stack (contributes to prefix) and a section
# stack (does not), and pop the appropriate one at `end X`.
_NS_DECL = re.compile(r"^namespace\s+([A-Za-z_][A-Za-z0-9_'\.]*)", re.MULTILINE)
_SEC_DECL = re.compile(r"^section(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?", re.MULTILINE)
_END_DECL = re.compile(r"^end(?:\s+([A-Za-z_][A-Za-z0-9_'\.]*))?\s*$", re.MULTILINE)


def _ns_at(src: str, pos: int) -> str:
    """Return the namespace prefix in effect at character offset
    `pos`. Walks the source up to `pos` tracking both namespace
    and section stacks; only namespaces contribute to the prefix.
    `end X` pops whichever stack has X on top (preferring the
    most-recent of either type).
    """
    # Build a chronological list of (start_pos, kind, name) events.
    events: list[tuple[int, str, str | None]] = []
    for m in _NS_DECL.finditer(src):
        if m.start() >= pos:
            break
        events.append((m.start(), "ns_open", m.group(1)))
    for m in _SEC_DECL.finditer(src):
        if m.start() >= pos:
            break
        events.append((m.start(), "sec_open", m.group(1)))
    for m in _END_DECL.finditer(src):
        if m.start() >= pos:
            break
        events.append((m.start(), "end", m.group(1)))
    events.sort(key=lambda e: e[0])

    ns_stack: list[str] = []
    sec_stack: list[str | None] = []
    for _, kind, name in events:
        if kind == "ns_open":
            ns_stack.append(name)
        elif kind == "sec_open":
            sec_stack.append(name)
        elif kind == "end":
            # Pop whichever stack the named end targets. If
            # unnamed `end`, pop the most recent of either type
            # (Lean requires matching, so the innermost frame
            # always pops here).
            if name is None:
                # Use the relative order of stack tops by tracking
                # most-recent open event. Without per-frame
                # positions we approximate by popping whichever
                # stack is non-empty; sections nest INSIDE
                # namespaces by convention here.
                if sec_stack:
                    sec_stack.pop()
                elif ns_stack:
                    ns_stack.pop()
            else:
                # Named end. Match against the innermost matching
                # frame on either stack.
                if ns_stack and ns_stack[-1] == name:
                    ns_stack.pop()
                elif sec_stack and sec_stack[-1] == name:
                    sec_stack.pop()
                # If neither matches at the top, walk down both
                # stacks for the most recent frame named X (Lean
                # allows `end X` to close any frame named X
                # provided everything between is also closed).
                else:
                    for st in (sec_stack, ns_stack):
                        if name in st:
                            # Pop down to (and including) the X
                            # frame. Sections / namespaces below
                            # it should already be closed by now,
                            # but be defensive.
                            while st and st[-1] != name:
                                st.pop()
                            if st:
                                st.pop()
                            break
    return ".".join(ns_stack)


def _classify_difficulty(proof_blob: str) -> tuple[str, int]:
    """Greedy heuristic — easy: 1 tactic; medium: 2-4; hard: 5+."""
    # Count distinct semicolons / newlines that separate tactics,
    # plus comma-separated lists.
    p = proof_blob.strip()
    # Strip leading `by`.
    if p.startswith("by"):
        p = p[2:].strip()
    if not p:
        return ("?", 0)
    # Rough tactic-token count.
    tokens = re.split(r"[\n;<>]+|\s·\s", p)
    tokens = [t for t in tokens if t.strip()]
    n = len(tokens)
    if n <= 1:
        return ("easy", n)
    if n <= 4:
        return ("medium", n)
    return ("hard", n)


def _family_tags(short_name: str) -> list[str]:
    s = short_name.lower()
    return [tok for tok in FAMILY_TOKENS if tok in s]


def scan_file(rel_path: str) -> list[dict]:
    abs_path = MATHLIB_ROOT / rel_path
    if not abs_path.exists():
        return []
    src_raw = abs_path.read_text(encoding="utf-8", errors="replace")
    src = _strip_comments(src_raw)
    out: list[dict] = []
    for m in DECL_RE.finditer(src):
        kind = m.group(1)
        short = m.group(2)
        ns = _ns_at(src, m.start())
        # Lean `_root_.X` syntax: declares X in the root namespace,
        # NOT in the current `namespace Y` context. Strip the
        # `_root_.` prefix and skip the namespace concatenation.
        if short.startswith("_root_."):
            full_name = short[len("_root_."):]
            ns = ""  # root namespace, override
        else:
            full_name = f"{ns}.{short}" if ns else short
        # Detect tactic proof: scan from declaration head to next
        # blank-line-prefixed `theorem|lemma|def|namespace|end`. The
        # body between `:=` and the next top-level decl is the proof.
        chunk_end = src.find("\n\n", m.end())
        if chunk_end == -1:
            chunk_end = m.end() + 800
        chunk = src[m.start():chunk_end]
        # Find `:=` body.
        eq_idx = chunk.find(":=")
        proof_blob = chunk[eq_idx + 2:] if eq_idx != -1 else ""
        has_tactic = bool(re.search(r"\bby\b", proof_blob)) or proof_blob.strip().startswith("by")
        diff, nt = _classify_difficulty(proof_blob)
        out.append({
            "file_path": rel_path,
            "full_name": full_name,
            "short_name": short,
            "namespace": ns,
            "kind": kind,
            "has_tactic_proof": has_tactic,
            "num_tactics_approx": nt,
            "difficulty": diff,
            "family_tags": _family_tags(short),
        })
    return out


def main() -> None:
    if not MATHLIB_ROOT.exists():
        print(f"ERROR: Mathlib root not found at {MATHLIB_ROOT}", file=sys.stderr)
        sys.exit(1)

    old = json.loads(OLD_CATALOG.read_text(encoding="utf-8"))
    old_names = {t["full_name"] for t in old["theorems"]}

    all_thms: list[dict] = []
    per_file: Counter[str] = Counter()
    missing: list[str] = []
    for rel in TARGET_FILES:
        rows = scan_file(rel)
        if not rows:
            missing.append(rel)
            continue
        per_file[rel] = len(rows)
        all_thms.extend(rows)

    # Mark which are already in the old catalog.
    for t in all_thms:
        t["already_in_old_catalog"] = t["full_name"] in old_names

    fresh = [t for t in all_thms if not t["already_in_old_catalog"]]
    with_tactic = [t for t in fresh if t["has_tactic_proof"]]

    by_ns_total: Counter[str] = Counter()
    by_ns_fresh: Counter[str] = Counter()
    by_ns_fresh_easy: Counter[str] = Counter()
    by_ns_fresh_med: Counter[str] = Counter()
    by_ns_fresh_hard: Counter[str] = Counter()
    for t in all_thms:
        ns_first = (t["full_name"].split(".", 1)[0] if "." in t["full_name"]
                    else "_no_ns_")
        by_ns_total[ns_first] += 1
    for t in fresh:
        ns_first = (t["full_name"].split(".", 1)[0] if "." in t["full_name"]
                    else "_no_ns_")
        by_ns_fresh[ns_first] += 1
        if t["has_tactic_proof"]:
            d = t["difficulty"]
            if d == "easy":
                by_ns_fresh_easy[ns_first] += 1
            elif d == "medium":
                by_ns_fresh_med[ns_first] += 1
            elif d == "hard":
                by_ns_fresh_hard[ns_first] += 1

    out = {
        "mathlib_commit": MATHLIB_COMMIT,
        "scan_method": "regex-source-scanner",
        "files_scanned": list(per_file.keys()),
        "files_missing": missing,
        "total_extracted": len(all_thms),
        "fresh_vs_old_catalog": len(fresh),
        "fresh_with_tactic_proof": len(with_tactic),
        "by_namespace_total": dict(by_ns_total),
        "by_namespace_fresh": dict(by_ns_fresh),
        "by_namespace_fresh_easy": dict(by_ns_fresh_easy),
        "by_namespace_fresh_medium": dict(by_ns_fresh_med),
        "by_namespace_fresh_hard": dict(by_ns_fresh_hard),
        "per_file_count": dict(per_file),
        "theorems": all_thms,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"  scanned files: {len(per_file)} / target {len(TARGET_FILES)}")
    if missing:
        print(f"  MISSING files: {missing}")
    print(f"  total extracted: {len(all_thms)}")
    print(f"  fresh (not in old catalog): {len(fresh)}")
    print(f"  fresh + has_tactic_proof: {len(with_tactic)}")
    print(f"\n  fresh by namespace:")
    for ns, c in by_ns_fresh.most_common(12):
        print(f"    {ns}: {c} (easy={by_ns_fresh_easy[ns]}, "
              f"med={by_ns_fresh_med[ns]}, hard={by_ns_fresh_hard[ns]})")


if __name__ == "__main__":
    main()
