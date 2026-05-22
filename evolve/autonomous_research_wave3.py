"""Wave 3 — adaptive variants seeded from the followup run's winners.

This is the place where the v5 loop *finally* uses prior-cycle results
to inform the next-cycle genome. Up to wave 2 the variants were
hand-curated. Wave 3 reads `scoreboard.jsonl` from the followup run,
finds the variants that closed new theorems, and seeds 5-10 mutations
around them.

Usage:
    python -m evolve.autonomous_research_wave3 \\
        --seed-scoreboard project/evolve/autonomous_runs/v5-followup-<id>/scoreboard.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evolve.autonomous_research_loop import (
    CycleResult, REPO_ROOT, Variant, baseline_genome,
    write_strategy_config, run_eval, summarize_metrics,
    _render_final_report,
)


def _load_winning_variant_genome(
    followup_run_dir: Path, variant_name: str
) -> dict[str, Any]:
    """Read the genome.json that was used by a winning followup variant."""
    p = followup_run_dir / "eval" / variant_name / "genome.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _mutate_inner_tactics(g: dict[str, Any]) -> dict[str, Any]:
    """Swap inner tactics in any term_builder / priority skeleton."""
    g = deepcopy(g)
    swaps = [
        ("simp_all", "simp [*]"),
        ("omega", "simp_arith"),
        ("simp [h]", "subst h; rfl"),
    ]
    def _apply(s: str) -> str:
        for a, b in swaps:
            s = s.replace(a, b)
        return s
    for k in list(g.get("term_builder_templates", {}).keys()):
        g["term_builder_templates"][k] = [_apply(t) for t in g["term_builder_templates"][k]]
    for k in list(g.get("priority_templates", {}).keys()):
        g["priority_templates"][k] = [_apply(t) for t in g["priority_templates"][k]]
    return g


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-scoreboard", required=True, type=Path)
    p.add_argument("--theorem-set", default="nat_defs_medium")
    p.add_argument("--ckpt-dir", default="project/models/gen_v5")
    p.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    p.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--num-children", type=int, default=4)
    args = p.parse_args()

    seed_run_dir = args.seed_scoreboard.parent
    rows = []
    for line in args.seed_scoreboard.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    # Pick winners — variants that beat baseline OR had new wins
    winners = [
        r for r in rows
        if r["proved"] > 26 or r.get("newly_proved")
    ]
    if not winners:
        # Fallback to top-3 by proved/progress
        winners = sorted(rows, key=lambda r: (r["proved"], r["progress"], -r["errored"]), reverse=True)[:3]
    print(f"# wave 3 — seeding from {len(winners)} winning followup variants")

    # Build child variants
    variants: list[Variant] = []
    for w in winners:
        try:
            seed_genome = _load_winning_variant_genome(seed_run_dir, w["name"])
        except FileNotFoundError:
            continue
        # Child 1: identity (re-verify)
        variants.append(Variant(
            name=f"w3-{w['name']}-repro",
            description=f"reproduce {w['name']} (proved {w['proved']}, new {w.get('newly_proved')})",
            direction="C",
            apply=lambda g, sg=seed_genome: deepcopy(sg),
        ))
        # Child 2: inner-tactic mutation
        variants.append(Variant(
            name=f"w3-{w['name']}-mut-inner",
            description=f"inner-tactic mutation of {w['name']}",
            direction="C",
            apply=lambda g, sg=seed_genome: _mutate_inner_tactics(sg),
        ))
        if len(variants) >= args.num_children * 2:
            break

    if not variants:
        print("# no winning variants to seed from")
        return

    run_id = f"v5-wave3-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    config_out_path = out_root / "config.json"
    config_out_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    started = time.time()
    deadline = started + args.max_hours * 3600
    log_lines: list[str] = []
    def log(m): print(m, flush=True); log_lines.append(m); log_path.write_text("\n".join(log_lines)+"\n")
    log(f"# v5 wave3 run — {run_id}\n")
    log(f"- seed: {args.seed_scoreboard}")
    log(f"- variants: {len(variants)}")
    results: list[CycleResult] = []
    baseline_proved_set = None
    baseline_proved_count = 26
    for i, variant in enumerate(variants):
        now = time.time()
        if now >= deadline:
            log("\n## stopping — max-hours")
            break
        log(f"\n## cycle {i+1} — {variant.name}  [{variant.direction}]")
        log(f"- {variant.description}")
        metrics_path = run_eval(variant, args.theorem_set, args.ckpt_dir, eval_root, args.per_eval_timeout_seconds)
        if metrics_path is None:
            log("- FAILED"); continue
        (proved, progress, errored, pbo, tb_a, tb_adv, tb_p,
         newly_proved, newly_lost, proved_set) = summarize_metrics(metrics_path, baseline_proved_set)
        if baseline_proved_set is None:
            baseline_proved_set = proved_set
            baseline_proved_count = proved
        delta = proved - baseline_proved_count
        elapsed = time.time() - now
        cyc = CycleResult(
            name=variant.name, direction=variant.direction,
            description=variant.description,
            proved=proved, progress=progress, errored=errored,
            proved_by_origin=pbo,
            term_builder_attempt=tb_a, term_builder_advanced=tb_adv,
            term_builder_proved=tb_p,
            runtime_seconds=elapsed, delta_vs_baseline=delta,
            newly_proved=newly_proved, newly_lost=newly_lost,
            eval_dir=str(metrics_path.parent),
        )
        results.append(cyc)
        with scoreboard_path.open("a") as f:
            f.write(json.dumps(cyc.to_dict(), ensure_ascii=False)+"\n")
        log(f"- proved: {proved} (Δ {delta:+d}) progress: {progress}")
        log(f"- origins: {pbo}")
        if newly_proved:
            log(f"- NEW: {newly_proved}")
    elapsed_total = time.time() - started
    final = out_root / "final_report.md"
    final.write_text(_render_final_report(run_id, args, results, elapsed_total))
    log(f"\n## complete — {elapsed_total/3600:.2f}h")


if __name__ == "__main__":
    main()
