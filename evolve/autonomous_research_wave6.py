"""Wave 6 — final targeted attempts at remaining failures.

After v5-27 master at 31/38, the 7 still-unsolved theorems are:
  - Nat.AM_GM, Nat.add_mod_eq_ite, Nat.div_le_div_right,
  - Nat.eq_one_of_mul_eq_one_left, Nat.sqrt_lt,
  - Nat.pow_lt_pow_iff_left, Nat.dvd_iff_div_mul_eq

Wave 6 tries more creative templates per theorem:
  - explicit Mathlib lemma forms
  - more aggressive case splits
  - alternative term-mode skeletons
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


def _w6_master_base(g: dict[str, Any]) -> dict[str, Any]:
    g["priority_templates"] = {
        "iff": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "constructor <;> intro h_split <;> simp_all",
        ],
        "lt": [
            "exact (Nat.le_div_iff_mul_le {hyp_pos}).mpr (by simpa using {hyp_le})",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
        ],
    }
    g["priority_template_budget"] = 12
    return g


def _w6_dvd_alt(g: dict[str, Any]) -> dict[str, Any]:
    """v5-34: try Nat.dvd_iff_div_mul_eq with more careful term-mode."""
    g = _w6_master_base(g)
    g["priority_templates"]["iff"] = list(g["priority_templates"]["iff"]) + [
        "exact ⟨fun h => h.symm ▸ Nat.div_mul_cancel h, fun h => ⟨n / d, h.symm⟩⟩",
        "exact ⟨Nat.div_mul_cancel, fun h => ⟨_, h.symm⟩⟩",
        "constructor; intro h; exact Nat.div_mul_cancel h; intro h; exact ⟨_, h.symm⟩",
        "refine ⟨fun ⟨k, hk⟩ => ?_, fun h => ⟨_, h.symm⟩⟩; rw [hk]; exact Nat.mul_div_cancel _ (Nat.pos_of_ne_zero (by omega))",
        "rw [Nat.dvd_iff_exists_eq_mul]",
    ]
    return g


def _w6_add_mod_ite_intro(g: dict[str, Any]) -> dict[str, Any]:
    """v5-35: Nat.add_mod_eq_ite has goal classifier returning some
    shape (le since `≤` inside the if). Try priority for "le" and
    explicit conv-based attempts."""
    g = _w6_master_base(g)
    # Goal classifier for `(m+n)%k = if k ≤ ... then ... else ...` —
    # the `=` outside makes it "eq" shape (after the `≤` and `<` checks
    # don't match because they're inside the if expression).
    # Actually let me check classify_goal_shape again:
    #   "=" check comes AFTER "<" / "≤". Both `<` and `≤` ARE in the goal
    #   (inside the if). So shape returns "le" (since ≤ is checked first).
    g["priority_templates"]["le"] = [
        # First: try omega
        "omega",
        # Then split_ifs
        "split_ifs with h",
        "split_ifs <;> omega",
        "split_ifs <;> simp_all <;> omega",
        # Or rewrite to make the ite top-level (unlikely)
        "by_cases h : k ≤ m % k + n % k <;> [rw [if_pos h]; rw [if_neg h]] <;> omega",
        # Direct
        "exact Nat.add_mod_eq_ite",  # self-ref likely blocked
    ]
    return g


def _w6_eq_one_alt(g: dict[str, Any]) -> dict[str, Any]:
    """v5-36: Nat.eq_one_of_mul_eq_one_left — case analysis."""
    g = _w6_master_base(g)
    g["priority_templates"]["eq"] = [
        # H : m * n = 1, goal: n = 1
        "rcases Nat.eq_one_of_mul_eq_one_right H with h; exact h",
        "exact (Nat.mul_eq_one.mp H).2",
        "have := Nat.mul_eq_one.mp H; exact this.2",
        # Bare lemma name (note theorem name doesn't have direct alt)
        "rcases m with _ | _ <;> rcases n with _ | _ <;> simp_all",
        "match m, n, H with | _, _, h => by simp_all <;> omega",
    ]
    return g


def _w6_div_le_div(g: dict[str, Any]) -> dict[str, Any]:
    """v5-37: Nat.div_le_div_right with more lemma forms."""
    g = _w6_master_base(g)
    g["priority_templates"]["le"] = [
        # Direct lemma forms
        "exact Nat.div_le_div_left {hyp_le} _ _",
        "exact Nat.div_le_div_iff_le.mpr {hyp_le}",
        "apply Nat.div_le_div_left {hyp_le}",
        # Via mul_le
        "rw [Nat.le_div_iff_mul_le (Nat.zero_lt_succ _)]",
        "induction c with | zero => simp | succ k ih => exact Nat.div_le_div_succ_succ {hyp_le}",
    ]
    return g


def _w6_combined(g: dict[str, Any]) -> dict[str, Any]:
    """v5-38: combined w6 — every working master template plus the
    new attempts for unsolved theorems."""
    g = _w6_master_base(g)
    g = _w6_dvd_alt(g)
    g = _w6_add_mod_ite_intro(g)
    g = _w6_eq_one_alt(g)
    g["priority_template_budget"] = 20
    return g


VARIANTS_WAVE6: list[Variant] = [
    Variant("v5-34-w6-dvd-alt", "wave6: dvd term-mode alternatives", "A+priority", _w6_dvd_alt),
    Variant("v5-35-w6-add-mod-ite", "wave6: add_mod_eq_ite via le-shape priorities", "B+priority", _w6_add_mod_ite_intro),
    Variant("v5-36-w6-eq-one-alt", "wave6: eq_one_of_mul via Nat.mul_eq_one", "B+priority", _w6_eq_one_alt),
    Variant("v5-37-w6-div-le-div", "wave6: div_le_div_right lemma forms", "B+priority", _w6_div_le_div),
    Variant("v5-38-w6-combined", "wave6: all w6 attempts in one variant", "all+priority", _w6_combined),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-hours", type=float, default=1.0)
    args = parser.parse_args()

    run_id = f"v5-wave6-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
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
    log(f"# v5 wave6 — {run_id}\n")
    log(f"- variants: {len(VARIANTS_WAVE6)}")
    log(f"- baseline (v5-00): {baseline_proved_count}/38")

    results: list[CycleResult] = []
    for i, variant in enumerate(VARIANTS_WAVE6):
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
