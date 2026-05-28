"""Lemma-name availability checker for the cached LeanDojo Mathlib tree.

Reports, for each candidate lemma name, whether it is *declared* somewhere
in the cached Mathlib source — and where — by grepping the local clone at

    ~/.cache/lean_dojo/leanprover-community-mathlib4-<commit>/mathlib4/

The checker is deliberately cheap (no Lean type-check, no Dojo bootstrap).
It is the static version of the runtime "unknown constant 'Foo'" signal:
if a name is not declared anywhere in the cache, calling it from a
template will fail with `unknown constant`.

Each report row carries:

  status     AVAILABLE / MISSING / CIRCULAR
  name       fully-qualified lemma name (e.g. Nat.div_mul_cancel)
  source     path:line of the declaration (relative to mathlib4/)
  note       extra context (which `namespace`, whether it is `private`,
             whether it appears in `_UNAVAILABLE_LEMMAS`, whether it is
             one of our target theorems → circular)

Run from repo root:

    python scripts/check_lean_names.py Nat.div_mul_cancel Nat.dvd_mul_left
    python scripts/check_lean_names.py --json  Nat.dvd_iff_div_mul_eq
    python scripts/check_lean_names.py --target Nat.dvd_iff_div_mul_eq \\
        Nat.div_mul_cancel Nat.dvd_mul_left

A `--target` is the theorem we are *trying to prove*; the checker compares
each candidate's source line against the target's line so candidates
declared AFTER the target in the same file are reported MISSING (out of
scope for the LeanDojo trace replay even though they exist in Mathlib).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Cache root — the only mathlib4 commit hash we have locally is
# 29dcec07; if a future LeanDojo run pulls a different commit, this
# resolver will pick the most recently modified one.
_LEAN_DOJO_CACHE = Path.home() / ".cache" / "lean_dojo"


def find_mathlib_root() -> Path | None:
    if not _LEAN_DOJO_CACHE.exists():
        return None
    candidates = sorted(
        _LEAN_DOJO_CACHE.glob("leanprover-community-mathlib4-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        m = c / "mathlib4"
        if m.exists():
            return m
    return None


def is_trace_scratch(path: Path) -> bool:
    """LeanDojo's trace replay injects randomly-suffixed copies of target
    files (e.g. ``Defsgly9co8r.lean`` is a copy of ``Defs.lean``). They
    contain the very theorems we are trying to prove and must not count
    as availability evidence.

    A file is a trace scratch iff a sibling ``.lean`` file exists in the
    same directory whose stem is a strict prefix of this file's stem,
    AND the tail (the part of the stem after the prefix) is at least 4
    characters long, is pure ``[a-z0-9_]``, and contains at least one
    digit or underscore. This catches ``Defsgly9co8r`` (sibling
    ``Defs.lean``, tail ``gly9co8r`` with digits) while letting real
    Mathlib files like ``Lemmas.lean``, ``Bitwise.lean``, or
    ``Hilbert90.lean`` through.
    """
    if not path.name.endswith(".lean"):
        return False
    stem = path.stem
    parent = path.parent
    for cut in range(1, len(stem)):
        prefix = stem[:cut]
        tail = stem[cut:]
        if len(tail) < 4:
            continue
        if not re.fullmatch(r"[a-z0-9_]+", tail):
            continue
        if not re.search(r"[0-9_]", tail):
            continue
        if (parent / f"{prefix}.lean").exists():
            return True
    return False


# Recognised declaration keywords. Order matters for the regex alternation.
_DECL_KWS = (
    "theorem",
    "lemma",
    "def",
    "abbrev",
    "instance",
    "protected theorem",
    "protected lemma",
    "protected def",
    "private theorem",
    "private lemma",
    "private def",
    "noncomputable theorem",
    "noncomputable def",
)
_DECL_KW_RE = "|".join(re.escape(k) for k in _DECL_KWS)


def _build_decl_re(short: str) -> re.Pattern[str]:
    r"""Match a single-line declaration head where the name being declared
    is exactly `short`. Allows leading whitespace and common attribute
    prefixes like ``@[simp]``. Uses a Lean-aware right boundary
    ``(?![\w'])`` rather than ``\b`` so apostrophe-suffixed names like
    ``le_div_iff_mul_le'`` are matched correctly (Python's ``\b`` treats
    ``'`` as a non-word character and refuses to anchor between two
    non-word chars)."""
    return re.compile(
        rf"^\s*(?:@\[[^\]]+\]\s*)?(?:{_DECL_KW_RE})\s+{re.escape(short)}(?![\w'])"
    )


@dataclass
class Finding:
    name: str
    status: str  # AVAILABLE | MISSING | CIRCULAR
    source: str | None = None  # path:line
    note: str = ""
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "source": self.source,
            "note": self.note,
            "aliases": self.aliases,
        }


def _scan_file_for_decl(path: Path, short: str) -> tuple[int, str] | None:
    """Return ``(line_no, decl_line)`` for the first match of `short` as a
    declaration head, scanning `path`. Tracks the current `namespace`
    stack so an unqualified short name only matches when the stack puts
    it under the namespace prefix of the user-supplied fully-qualified
    name (handled by the caller; this function just finds head lines).
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    regex = _build_decl_re(short)
    for i, line in enumerate(text.splitlines(), start=1):
        if regex.match(line):
            return (i, line.strip())
    return None


def _namespace_stack_at_line(path: Path, target_line: int) -> list[str]:
    """Return the namespace stack active at `target_line` (1-indexed)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    stack: list[str] = []
    ns_open_re = re.compile(r"^\s*namespace\s+([A-Za-z_][\w.]*)\s*$")
    ns_close_re = re.compile(r"^\s*end(?:\s+([A-Za-z_][\w.]*))?\s*$")
    for i, line in enumerate(text.splitlines(), start=1):
        if i > target_line:
            break
        m = ns_open_re.match(line)
        if m:
            stack.append(m.group(1))
            continue
        m = ns_close_re.match(line)
        if m and stack:
            stack.pop()
    return stack


def lookup(
    name: str,
    mathlib_root: Path,
    *,
    target_name: str | None = None,
    target_lines: dict[str, int] | None = None,
) -> Finding:
    """Resolve one fully-qualified Mathlib name. `name` looks like
    ``Nat.div_mul_cancel`` (or ``Nat.div_mul_cancel'`` with apostrophe).
    Searches all .lean files in mathlib_root/Mathlib for either:

      * a head declaring the fully-qualified name (``lemma Nat.foo``)
      * a head declaring the short name (``lemma foo``) inside a
        namespace stack that resolves to that fully-qualified name.

    Returns a :class:`Finding`. When ``target_name``/``target_lines`` are
    given, names declared at or after the target's line in the SAME file
    are marked MISSING (out of scope for trace replay)."""
    if "." in name:
        ns, short = name.rsplit(".", 1)
    else:
        ns, short = "", name

    best: Finding | None = None
    aliases: list[str] = []

    # Scan Mathlib first, then the bundled Lean core stdlib under
    # `.lake/packages/lean4/src/lean/Init/` — that is where
    # `Nat.div_mul_cancel`, `Nat.mul_comm`, `Nat.dvd_mul_left`, etc.
    # actually live.
    scan_roots = [
        mathlib_root / "Mathlib",
        mathlib_root / ".lake" / "packages" / "lean4" / "src" / "lean" / "Init",
    ]
    lean_files: list[Path] = []
    for root in scan_roots:
        if root.exists():
            lean_files.extend(root.rglob("*.lean"))
    for lean_file in lean_files:
        if is_trace_scratch(lean_file):
            continue

        # 1. exact qualified head (e.g. `theorem Nat.foo`)
        full_hit = _scan_file_for_decl(lean_file, name)
        if full_hit:
            line_no, decl_line = full_hit
            try:
                rel = str(lean_file.relative_to(mathlib_root))
            except ValueError:
                rel = str(lean_file)
            f = Finding(
                name=name,
                status="AVAILABLE",
                source=f"{rel}:{line_no}",
                note="fully-qualified declaration head",
            )
            if "private" in decl_line:
                f.note += " (private — file-local only)"
            return f  # exact head wins

        # 2. short head inside matching namespace
        short_hit = _scan_file_for_decl(lean_file, short)
        if short_hit:
            line_no, decl_line = short_hit
            stack = _namespace_stack_at_line(lean_file, line_no)
            resolved = ".".join(stack + [short])
            if ns and resolved == name:
                try:
                    rel = str(lean_file.relative_to(mathlib_root))
                except ValueError:
                    rel = str(lean_file)
                f = Finding(
                    name=name,
                    status="AVAILABLE",
                    source=f"{rel}:{line_no}",
                    note=f"short name under `namespace {ns}`",
                )
                if "private" in decl_line:
                    f.note += " (private — file-local only)"
                if best is None or line_no < int(best.source.rsplit(":", 1)[1]):
                    best = f
            elif not ns:
                try:
                    rel = str(lean_file.relative_to(mathlib_root))
                except ValueError:
                    rel = str(lean_file)
                aliases.append(f"{resolved} @ {rel}:{line_no}")

    if best is None:
        return Finding(name=name, status="MISSING", aliases=aliases)

    # Circularity / scope check.
    if target_name and best.source:
        src_path, src_line = best.source.rsplit(":", 1)
        target_line_in_same_file = (target_lines or {}).get(src_path)
        if target_line_in_same_file and int(src_line) >= target_line_in_same_file:
            return Finding(
                name=name,
                status="CIRCULAR",
                source=best.source,
                note=(
                    f"declared at or after target {target_name} "
                    f"(line {target_line_in_same_file}) in the same file — "
                    "out of scope for LeanDojo trace replay"
                ),
            )
    return best


def _print_human(findings: list[Finding]) -> None:
    width = max((len(f.name) for f in findings), default=10)
    for f in findings:
        head = f"{f.status:9s} {f.name:<{width}}"
        if f.source:
            head += f"  {f.source}"
        if f.note:
            head += f"  ({f.note})"
        print(head)
        for a in f.aliases[:3]:
            print(f"    alias? {a}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="Fully-qualified lemma names (e.g. Nat.div_mul_cancel)")
    ap.add_argument(
        "--target", default=None,
        help="If given, flag candidates declared at or after the target's line "
             "in the same file as CIRCULAR (out of scope for trace replay)."
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = ap.parse_args()

    mathlib = find_mathlib_root()
    if mathlib is None:
        print(
            "ERROR: no cached LeanDojo Mathlib tree at ~/.cache/lean_dojo/. "
            "Run an eval once to populate it, or set LEAN_DOJO_CACHE_DIR.",
            file=sys.stderr,
        )
        return 2

    target_lines: dict[str, int] = {}
    if args.target:
        tgt = lookup(args.target, mathlib)
        if tgt.status == "AVAILABLE" and tgt.source:
            src_path, src_line = tgt.source.rsplit(":", 1)
            target_lines[src_path] = int(src_line)
        else:
            print(
                f"WARN: target {args.target} not found in Mathlib cache "
                f"({tgt.status}). Circularity check disabled.",
                file=sys.stderr,
            )

    findings = [
        lookup(n, mathlib, target_name=args.target, target_lines=target_lines)
        for n in args.names
    ]

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2))
    else:
        if args.target and target_lines:
            print(f"# target = {args.target} ({list(target_lines.items())[0]})")
        _print_human(findings)

    missing = sum(1 for f in findings if f.status == "MISSING")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
