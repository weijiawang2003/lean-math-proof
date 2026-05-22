"""Wave 5 — mutations around v5-27 master (31/38) to probe whether further
wins are reachable by inner-tier slot mutation.

The v5-27 priority_templates list is the current best. Wave 5 tests:
  - reordering templates within a shape key (which fires first matters)
  - adding more lemma names to mul/div priorities
  - extending to "le" shape (for Nat.div_le_div_right)
  - trying split_ifs variants tuned for Nat.add_mod_eq_ite
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evolve.autonomous_research_loop import (
    CycleResult, REPO_ROOT, Variant, baseline_genome,
    write_strategy_config, run_eval, summarize_metrics,
    _render_final_report,
)


def _w5_master_base(g: dict[str, Any]) -> dict[str, Any]:
    """Apply the v5-27 master template set (the current best)."""
    g["priority_templates"] = {
        "iff": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.div_mul_cancel h, fun h => ⟨_, h.symm⟩⟩",
            "rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]",
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "constructor <;> intro h_split <;> simp_all",
        ],
        "lt": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "exact (Nat.le_div_iff_mul_le {hyp_pos}).mpr (by simpa using {hyp_le})",
        ],
        "any": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all <;> omega",
        ],
    }
    g["priority_template_budget"] = 14
    return g


def _w5_add_le_shape(g: dict[str, Any]) -> dict[str, Any]:
    """v5-29: master + le-shape priority for Nat.div_le_div_right."""
    g = _w5_master_base(g)
    g["priority_templates"]["le"] = [
        "exact Nat.div_le_div_right {hyp_le}",
        "exact Nat.div_le_div_left {hyp_le} _ _",
        "exact Nat.div_le_div {hyp_le}",
        "exact (Nat.div_le_div_iff_right _ _).mpr {hyp_le}",
        "apply Nat.div_le_div_right; exact {hyp_le}",
    ]
    return g


def _w5_add_mod_eq_ite_skeleton(g: dict[str, Any]) -> dict[str, Any]:
    """v5-30: master + priority for Nat.add_mod_eq_ite. The goal has
    `if k ≤ m%k + n%k then ... else ...`; goal_shape after Nat goal-
    classifier returns "le" (since `≤` appears inside the if). We try
    `simp [Nat.add_mod_def]; split_ifs <;> omega` directly."""
    g = _w5_master_base(g)
    g["priority_templates"]["le"] = list(g["priority_templates"].get("le", [])) + [
        "simp [Nat.add_mod_def]; split_ifs <;> omega",
        "rw [Nat.add_mod_def]; split_ifs <;> omega",
        "split_ifs with h <;> [omega; omega]",
        "by_cases h : k ≤ m % k + n % k <;> simp [h, Nat.add_mod_def] <;> omega",
    ]
    return g


def _w5_iff_reorder(g: dict[str, Any]) -> dict[str, Any]:
    """v5-31: master with iff list reordered — div templates last so
    omega-omega fires first on trivial iffs (verify cap-truncation
    isn't dropping things). Test of within-slot ordering sensitivity."""
    g = _w5_master_base(g)
    g["priority_templates"]["iff"] = [
        # generic iff first
        "exact ⟨fun h => by omega, fun h => by omega⟩",
        "constructor <;> intro h_split <;> simp_all",
        # mul second
        "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
        "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
        # div last
        "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
        "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
    ]
    return g


def _w5_dvd_iff_specific(g: dict[str, Any]) -> dict[str, Any]:
    """v5-32: master + a more specific attempt at Nat.dvd_iff_div_mul_eq."""
    g = _w5_master_base(g)
    g["priority_templates"]["iff"] = list(g["priority_templates"]["iff"]) + [
        "exact ⟨fun h => Nat.div_mul_cancel h, fun h => ⟨n / d, h.symm⟩⟩",
        "refine ⟨fun h => ?_, fun h => ?_⟩ <;> [exact Nat.div_mul_cancel h; exact ⟨_, h.symm⟩]",
        "constructor; · exact Nat.div_mul_cancel; · intro h; exact ⟨_, h.symm⟩",
    ]
    return g


def _w5_eq_one_of_mul(g: dict[str, Any]) -> dict[str, Any]:
    """v5-33: master + eq-shape priority for Nat.eq_one_of_mul_eq_one_left.
    Goal `n = 1` with `H : m * n = 1`. Use mul_eq_one decomposition."""
    g = _w5_master_base(g)
    g["priority_templates"]["eq"] = [
        # Mathlib has Nat.mul_eq_one_iff which gives m=1 ∧ n=1
        "rcases (Nat.mul_eq_one).mp H with ⟨_, hn⟩; exact hn",
        "exact (Nat.mul_eq_one.mp H).2",
        "exact (Nat.mul_eq_one_iff_eq_one_and_eq_one.mp H).2",
        # symmetric form attempts
        "have := Nat.eq_one_of_pos_of_self_mul_self_eq_one (by omega : 0 < n) (by linarith)",
        "omega",
        "simp_all",
    ]
    return g


VARIANTS_WAVE5: list[Variant] = [
    Variant("v5-29-w5-le-shape", "wave5: master + le-shape for div_le_div_right", "B+priority", _w5_add_le_shape),
    Variant("v5-30-w5-add-mod-ite", "wave5: master + add_mod_eq_ite priorities", "B+priority", _w5_add_mod_eq_ite_skeleton),
    Variant("v5-31-w5-iff-reorder", "wave5: iff list reordering sensitivity test", "C+priority", _w5_iff_reorder),
    Variant("v5-32-w5-dvd-specific", "wave5: master + dvd_iff_div_mul_eq attempts", "A+priority", _w5_dvd_iff_specific),
    Variant("v5-33-w5-eq-one-of-mul", "wave5: master + eq-shape for eq_one_of_mul", "B+priority", _w5_eq_one_of_mul),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-hours", type=float, default=1.0)
    args = parser.parse_args()

    run_id = f"v5-wave5-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    config_out_path = out_root / "config.json"
    config_out_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    import glob
    base_metrics = glob.glob(str(REPO_ROOT / 'project/evolve/autonomous_runs/v5-auto-*/eval/v5-00-baseline-repro/eval-*/metrics.json'))
    if base_metrics:
        m = json.loads(open(base_metrics[0]).read())
        baseline_proved_set = {t['full_name'] for t in m['per_theorem'] if t.get('finished')}
    else:
        baseline_proved_set = set()
    baseline_proved_count = len(baseline_proved_set)

    started = time.time()
    deadline = started + args.max_hours * 3600
    log_lines: list[str] = []
    def log(m): print(m, flush=True); log_lines.append(m); log_path.write_text("\n".join(log_lines)+"\n")
    log(f"# v5 wave5 — {run_id}\n")
    log(f"- variants: {len(VARIANTS_WAVE5)}")
    log(f"- baseline (v5-00): {baseline_proved_count}/38")

    results: list[CycleResult] = []
    for i, variant in enumerate(VARIANTS_WAVE5):
        now = time.time()
        if now >= deadline:
            log("\n## stopping — max-hours")
            break
        log(f"\n## cycle {i+1} — {variant.name}")
        log(f"- {variant.description}")
        metrics_path = run_eval(variant, args.theorem_set, args.ckpt_dir, eval_root, args.per_eval_timeout_seconds)
        if metrics_path is None:
            log("- FAILED"); continue
        proved, progress, errored, pbo, tb_a, tb_adv, tb_p, newly_proved, newly_lost, proved_set = summarize_metrics(metrics_path, baseline_proved_set)
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
        log(f"- proved: {proved} (Δ {delta:+d})")
        log(f"- origins: {pbo}")
        if newly_proved:
            log(f"- NEW WINS: {newly_proved}")
        if newly_lost:
            log(f"- REGRESSIONS: {newly_lost}")
    elapsed_total = time.time() - started
    final = out_root / "final_report.md"
    final.write_text(_render_final_report(run_id, args, results, elapsed_total))
    log(f"\n## complete — {elapsed_total/3600:.2f}h, {len(results)} cycles")


if __name__ == "__main__":
    main()
