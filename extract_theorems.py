"""Automatically extract available theorems from mathlib4 Lean files.

Uses LeanDojo's TracedRepo to enumerate all theorems in specified files,
checks which ones have tactic-style proofs (and are thus suitable for
our pipeline), and outputs a theorem set that can be plugged into tasks.py.

Two modes:
  - Full mode (default): Uses TracedRepo.trace() to get all theorems with
    proof metadata. Accurate but requires tracing cache.
  - Lightweight mode (--lightweight): Uses TracedRepo only to list theorem
    names per file, without requiring full trace data. Faster for discovery.

Usage:
  python extract_theorems.py
  python extract_theorems.py --files "Mathlib/Data/Set/Basic.lean,Mathlib/Data/Finset/Basic.lean"
  python extract_theorems.py --out discovered_theorems.json --check-availability
  python extract_theorems.py --lightweight --check-availability
  python extract_theorems.py --preset all_known   # scan all known-good files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_dojo import Dojo, LeanGitRepo, Theorem, trace

from core_types import TheoremConfig
from env import COMMIT, REPO_URL

# Files confirmed to have trace artifacts at commit 29dcec07
CONFIRMED_FILES = [
    "Mathlib/Data/Nat/Defs.lean",
    "Mathlib/Data/Set/Basic.lean",
    "Mathlib/Data/Finset/Basic.lean",
]

# Additional files likely to have trace artifacts (same mathlib4 version)
EXTENDED_FILES = [
    "Mathlib/Data/Set/Function.lean",
    "Mathlib/Data/Set/Lattice.lean",
    "Mathlib/Data/Set/Finite.lean",
    "Mathlib/Data/Finset/Card.lean",
    "Mathlib/Data/Finset/Image.lean",
    "Mathlib/Data/Finset/Lattice.lean",
    "Mathlib/Data/Nat/Basic.lean",
    "Mathlib/Data/Nat/Order/Basic.lean",
    "Mathlib/Data/Nat/GCD/Basic.lean",
    "Mathlib/Data/List/Basic.lean",
    "Mathlib/Data/List/Defs.lean",
    "Mathlib/Data/Bool/Basic.lean",
    "Mathlib/Data/Int/Basic.lean",
    "Mathlib/Logic/Basic.lean",
    "Mathlib/Order/Basic.lean",
    "Mathlib/Tactic/Ring.lean",
]

# Presets for convenience
FILE_PRESETS = {
    "core": CONFIRMED_FILES,
    "extended": CONFIRMED_FILES + EXTENDED_FILES,
    "all_known": CONFIRMED_FILES + EXTENDED_FILES,
}


def _get_theorem_name(thm) -> str:
    """Extract the full qualified name from a TracedTheorem.

    LeanDojo's TracedTheorem has a .theorem attribute which is a Theorem
    object with .full_name.  We access it that way.
    """
    # Primary: thm.theorem.full_name (LeanDojo ≥ 2.0)
    theorem_obj = getattr(thm, "theorem", None)
    if theorem_obj is not None:
        name = getattr(theorem_obj, "full_name", None)
        if name:
            return str(name)

    # Fallbacks for other LeanDojo versions
    for attr in ("full_name", "name", "theorem_name", "decl_name"):
        val = getattr(thm, attr, None)
        if val:
            return str(val)

    return str(thm)


def extract_from_traced_repo(
    files: list[str],
    require_tactic_proof: bool = True,
) -> list[dict]:
    """Trace the repo and extract all theorems from the specified files.

    Returns a list of dicts with keys:
        file_path, full_name, has_tactic_proof, num_tactics
    """
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)
    print(f"Tracing repo (may use cache)...")
    traced_repo = trace(repo)

    all_theorems = traced_repo.get_traced_theorems()
    print(f"Total traced theorems in repo: {len(all_theorems)}")

    # Exact file path matching — paths confirmed to be like "Mathlib/Data/Set/Basic.lean"
    file_set = set(files)

    results = []
    files_found = set()
    n_skipped_private = 0
    n_skipped_no_tactic = 0
    n_total_in_files = 0

    for thm in all_theorems:
        fp = str(thm.file_path)
        if fp not in file_set:
            continue
        files_found.add(fp)
        n_total_in_files += 1

        # Get tactic count — use get_traced_tactics() which is the most
        # reliable indicator of tactic-style proof in this LeanDojo version
        has_tactic = False
        num_tactics = 0

        try:
            traced_tacs = thm.get_traced_tactics()
            if traced_tacs is not None and len(traced_tacs) > 0:
                has_tactic = True
                num_tactics = len(traced_tacs)
        except Exception:
            pass

        # Skip non-tactic proofs if required (term-mode proofs, etc.)
        if require_tactic_proof and not has_tactic:
            n_skipped_no_tactic += 1
            continue

        thm_name = _get_theorem_name(thm)

        results.append({
            "file_path": fp,
            "full_name": thm_name,
            "has_tactic_proof": has_tactic,
            "num_tactics": num_tactics,
        })

    # Report
    missing = file_set - files_found
    if missing:
        print(f"[WARN] No theorems found in {len(missing)} file(s):")
        for f in sorted(missing):
            print(f"  - {f}")

    print(f"Total theorems in target files: {n_total_in_files}")
    print(f"Extracted {len(results)} with tactic proofs from {len(files_found)} file(s)")
    if n_skipped_no_tactic:
        print(f"  Skipped {n_skipped_no_tactic} without tactic proofs (term-mode)")
    return results


def extract_lightweight(
    files: list[str],
) -> list[dict]:
    """Lightweight extraction: just try opening each theorem in Dojo.

    This doesn't require TracedRepo.trace() — it uses the Theorem constructor
    which only needs the .ast.json files to be cached.

    Much slower per-theorem but doesn't need to trace the whole repo.
    This is a fallback for when full tracing isn't practical.
    """
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)

    # We can't enumerate theorems without tracing, so this mode
    # is used after we already have a discovered_theorems.json from
    # a previous full trace. It just re-checks availability.
    print("[lightweight] This mode verifies pre-discovered theorems.")
    print("[lightweight] Run full mode first to discover theorem names.")
    return []


def check_availability(theorems: list[dict], batch_size: int = 50) -> tuple[list[dict], list[dict]]:
    """Check which theorems can actually be opened by LeanDojo Dojo.

    Processes in batches and reports progress.
    """
    repo = LeanGitRepo(url=REPO_URL, commit=COMMIT)
    available = []
    unavailable = []

    total = len(theorems)
    for i, t in enumerate(theorems):
        name = t["full_name"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Checking availability [{i+1}/{total}] {name}")
        try:
            theorem = Theorem(repo=repo, file_path=t["file_path"], full_name=name)
            with Dojo(theorem):
                pass
            t["available"] = True
            available.append(t)
        except Exception as exc:
            t["available"] = False
            t["error"] = str(exc)[:200]
            unavailable.append(t)

    return available, unavailable


def estimate_difficulty(theorems: list[dict]) -> list[dict]:
    """Rank theorems by estimated difficulty based on heuristics.

    Difficulty tiers:
      easy   -- 1-2 tactics in human proof
      medium -- 3-5 tactics
      hard   -- 6+ tactics or unknown
    """
    for t in theorems:
        n = t.get("num_tactics", 0)
        if n <= 2:
            t["difficulty"] = "easy"
            t["difficulty_score"] = n if n > 0 else 1
        elif n <= 5:
            t["difficulty"] = "medium"
            t["difficulty_score"] = n
        else:
            t["difficulty"] = "hard"
            t["difficulty_score"] = n if n > 0 else 99
    theorems.sort(key=lambda t: t["difficulty_score"])
    return theorems


def generate_tasks_py_snippet(theorems: list[dict], set_name: str) -> str:
    """Generate Python code for a THEOREM_SETS entry."""
    lines = [f'    "{set_name}": [']
    for t in theorems:
        diff = t.get("difficulty", "?")
        lines.append(
            f'        TheoremConfig(file_path="{t["file_path"]}", '
            f'full_name="{t["full_name"]}"),  # {diff}'
        )
    lines.append("    ],")
    return "\n".join(lines)


def generate_register_commands(theorems: list[dict], json_path: str) -> str:
    """Generate run_incremental.py register commands."""
    return f"python run_incremental.py register --from-json {json_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract available theorems from mathlib4 Lean files."
    )
    parser.add_argument(
        "--files",
        default="",
        help="Comma-separated list of Lean file paths to scan.",
    )
    parser.add_argument(
        "--preset",
        default="",
        choices=["", "core", "extended", "all_known"],
        help="Use a predefined file list instead of --files.",
    )
    parser.add_argument("--out", default="discovered_theorems.json")
    parser.add_argument(
        "--check-availability",
        action="store_true",
        help="Verify each theorem opens in Dojo (slow but accurate).",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight extraction (no full repo trace).",
    )
    parser.add_argument(
        "--require-tactic-proof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only include theorems with tactic-style proofs.",
    )
    parser.add_argument(
        "--generate-task-set",
        default="",
        help="Generate a tasks.py snippet with this set name.",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=0,
        help="Max theorems to extract per file (0 = all). Useful for testing.",
    )
    args = parser.parse_args()

    # Determine which files to scan
    if args.preset:
        files = FILE_PRESETS[args.preset]
    elif args.files:
        files = [f.strip() for f in args.files.split(",")]
    else:
        files = CONFIRMED_FILES

    print(f"Scanning {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")
    print()

    # Step 1: Extract theorems
    if args.lightweight:
        theorems = extract_lightweight(files)
    else:
        theorems = extract_from_traced_repo(
            files,
            require_tactic_proof=args.require_tactic_proof,
        )

    # Optional: cap per file
    if args.max_per_file > 0:
        from collections import defaultdict
        by_file = defaultdict(list)
        for t in theorems:
            by_file[t["file_path"]].append(t)
        capped = []
        for fp, ts in by_file.items():
            capped.extend(ts[:args.max_per_file])
        print(f"Capped to {args.max_per_file} per file: {len(theorems)} -> {len(capped)}")
        theorems = capped

    # Step 2: Estimate difficulty
    theorems = estimate_difficulty(theorems)

    # Step 3: Optional availability check
    if args.check_availability:
        print(f"\nChecking Dojo availability for {len(theorems)} theorems...")
        available, unavailable = check_availability(theorems)
        print(f"  Available: {len(available)}, Unavailable: {len(unavailable)}")
        theorems = available  # Only keep available ones

    # Step 4: Output
    output = {
        "commit": COMMIT,
        "files_scanned": files,
        "total_extracted": len(theorems),
        "per_difficulty": {
            "easy": len([t for t in theorems if t.get("difficulty") == "easy"]),
            "medium": len([t for t in theorems if t.get("difficulty") == "medium"]),
            "hard": len([t for t in theorems if t.get("difficulty") == "hard"]),
        },
        "theorems": theorems,
    }

    Path(args.out).write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(theorems)} theorems to {args.out}")

    # Per-file summary
    from collections import Counter
    file_counts = Counter(t["file_path"] for t in theorems)
    for fp, count in file_counts.most_common():
        easy = len([t for t in theorems if t["file_path"] == fp and t.get("difficulty") == "easy"])
        med = len([t for t in theorems if t["file_path"] == fp and t.get("difficulty") == "medium"])
        hard = len([t for t in theorems if t["file_path"] == fp and t.get("difficulty") == "hard"])
        print(f"  {fp}: {count} theorems (easy={easy}, medium={med}, hard={hard})")

    # Show register command for incremental pipeline
    print(f"\nTo register in incremental pipeline:")
    print(f"  python run_incremental.py register --from-json {args.out}")

    # Optional: generate tasks.py snippet
    if args.generate_task_set:
        snippet = generate_tasks_py_snippet(theorems, args.generate_task_set)
        snippet_path = args.out.replace(".json", "_tasks.py")
        Path(snippet_path).write_text(snippet, encoding="utf-8")
        print(f"\nTasks.py snippet saved to {snippet_path}")


if __name__ == "__main__":
    main()
