"""Analyze v5 autonomous runs — find the best candidate across runs and
produce a unified scoreboard.

Usage:
    python -m evolve.analyze_v5_runs --runs-dir project/evolve/autonomous_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="project/evolve/autonomous_runs", type=Path)
    args = p.parse_args()
    runs = sorted(args.runs_dir.glob("v5-*"))
    all_rows = []
    for run in runs:
        sb = run / "scoreboard.jsonl"
        if not sb.exists():
            continue
        for line in sb.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["run"] = run.name
            all_rows.append(d)
    print(f"# v5 unified scoreboard — {len(all_rows)} candidates across {len(runs)} runs\n")
    print("| run | variant | proved | Δ | prog | err | tb a/adv/p | new wins | regressions |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in sorted(all_rows, key=lambda r: (-r["proved"], r["progress"], r["errored"])):
        np = ", ".join(d["newly_proved"]) if d.get("newly_proved") else "—"
        nl = ", ".join(d["newly_lost"]) if d.get("newly_lost") else "—"
        tb = f"{d['term_builder_attempt']}/{d['term_builder_advanced']}/{d['term_builder_proved']}"
        print(f"| {d['run']} | {d['name']} | {d['proved']} | {d['delta_vs_baseline']:+d} | {d['progress']} | {d['errored']} | {tb} | {np} | {nl} |")
    print()
    if all_rows:
        best = max(all_rows, key=lambda r: (r["proved"], r["progress"], -r["errored"]))
        print(f"## Best overall\n")
        print(f"- run: `{best['run']}`")
        print(f"- variant: `{best['name']}`")
        print(f"- direction: {best['direction']}")
        print(f"- proved: **{best['proved']}**")
        print(f"- description: {best['description']}")
        if best.get("newly_proved"):
            print(f"- newly proved: {best['newly_proved']}")
        # Union of all newly_proved across runs
        all_new = set()
        for d in all_rows:
            for thm in d.get("newly_proved") or []:
                all_new.add(thm)
        print(f"\n## Union of newly proved across all v5 candidates ({len(all_new)})\n")
        for thm in sorted(all_new):
            # Find which variants first proved this
            provers = [d['name'] for d in all_rows if thm in (d.get('newly_proved') or [])]
            print(f"- `{thm}` — proved by: {provers}")


if __name__ == "__main__":
    main()
