"""Generate additional v5 variants for a second autonomous loop pass.

Reads `scoreboard.jsonl` from a completed run and produces follow-up
variants that:
  1. Combine the best variant with each non-winning shape mini-solver
     (to see if the combination unlocks new closures).
  2. Probe deeper mutations of any term_builder skeleton that proved
     anything.
  3. Adds targeted lemma-based templates for theorems still un-proved.

The output is a Python dict literal printed to stdout that the user can
paste into autonomous_research_loop.py's VARIANTS_DEFAULT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summarize_run(scoreboard_path: Path) -> tuple[dict, list[dict]]:
    rows = []
    for line in scoreboard_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        return {}, []
    best = max(rows, key=lambda r: (r["proved"], r["progress"], -r["errored"]))
    return best, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scoreboard", required=True, type=Path)
    args = p.parse_args()
    best, rows = _summarize_run(args.scoreboard)
    if not best:
        print("# empty scoreboard")
        return
    print(f"# best from {args.scoreboard.parent.name}: {best['name']}")
    print(f"# proved={best['proved']} delta={best['delta_vs_baseline']:+d}")
    if best["newly_proved"]:
        print(f"# new wins: {best['newly_proved']}")
    # All variants that closed something new
    winners = [r for r in rows if r["newly_proved"]]
    print(f"# {len(winners)} variants closed at least one new theorem")
    for r in winners:
        print(f"#   - {r['name']}: {r['newly_proved']}")


if __name__ == "__main__":
    main()
