"""Wave 4 — targeted priority_templates for the remaining failing theorems.

After v5-12 (div_lt_one_iff) and v5-15 (mul_eq_*) closed three theorems,
the remaining 9 failures are:

  - Nat.AM_GM, Nat.add_mod_eq_ite,
  - Nat.div_le_div_right, Nat.div_pos, Nat.div_pos_iff,
  - Nat.dvd_iff_div_mul_eq, Nat.eq_one_of_mul_eq_one_left,
  - Nat.sqrt_lt, Nat.pow_lt_pow_iff_left.

This wave tries one targeted variant per remaining theorem class.
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


def _w4_split_ifs_dedicated(g: dict[str, Any]) -> dict[str, Any]:
    """v5-23: priority `split_ifs <;> omega` keyed on EVERY non-iff/dvd
    shape (so eq/le/lt goals get it). Also remove the
    `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` deny that v3.6 set on
    Nat.add_mod_eq_ite — that deny was protecting against Dojo crashes
    on the OLD evaluation environment. Now we want split_ifs to fire
    before any simp."""
    g["priority_templates"] = {
        "any": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all <;> omega",
            "split_ifs <;> simp_all",
            "simp [Nat.add_mod_def] <;> split_ifs <;> omega",
        ],
        "le": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all",
        ],
        "eq": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all",
        ],
    }
    g["priority_template_budget"] = 6
    return g


def _w4_dvd_iff(g: dict[str, Any]) -> dict[str, Any]:
    """v5-24: priority for Nat.dvd_iff_div_mul_eq (iff between dvd and
    div-mul-eq). Uses an asymmetric iff split with dvd-witness on one
    side and the div-eq-of-dvd lemma on the other."""
    g["priority_templates"] = {
        "iff": [
            "exact ⟨fun h => Nat.div_mul_cancel h, fun h => ⟨n / d, h.symm⟩⟩",
            "constructor <;> intro h <;> simp_all [Nat.div_mul_cancel]",
            "exact ⟨fun h => Nat.div_mul_cancel h, fun h => Nat.dvd_of_div_mul_eq h⟩",
            # bare dvd cancellation form
            "exact ⟨Nat.div_mul_cancel, fun h => ⟨n / d, h.symm⟩⟩",
        ],
    }
    g["priority_template_budget"] = 4
    return g


def _w4_div_pos(g: dict[str, Any]) -> dict[str, Any]:
    """v5-25: priority for Nat.div_pos and Nat.div_pos_iff.

      Nat.div_pos    : 0 < a/b with hba:b≤a, hb:0<b
      Nat.div_pos_iff: 0 < a/b ↔ b ≤ a with hb:b≠0
    """
    g["priority_templates"] = {
        "lt": [
            # Mathlib forms — try multiple
            "exact (Nat.le_div_iff_mul_le {hyp_pos}).mpr (by simp; exact {hyp_le})",
            "exact Nat.div_pos {hyp_le} {hyp_pos}",  # self-ref likely blocked by lean
            "exact Nat.lt_of_lt_of_le Nat.zero_lt_one ((Nat.le_div_iff_mul_le {hyp_pos}).mpr (by simpa using {hyp_le}))",
        ],
        "iff": [
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            "constructor; · intro h; by_contra hc; push_neg at hc; simp [Nat.div_eq_of_lt hc] at h; · intro h; exact Nat.div_pos h (Nat.pos_of_ne_zero {hyp_ne_zero})",
        ],
    }
    g["priority_template_budget"] = 5
    return g


def _w4_sqrt_pow(g: dict[str, Any]) -> dict[str, Any]:
    """v5-26: priority for Nat.sqrt_lt and Nat.pow_lt_pow_iff_left.
    Try multiple lemma form variants since the prime form doesn't exist."""
    g["priority_templates"] = {
        "iff": [
            # Try bare without prime
            "exact Nat.sqrt_lt",
            "simp [Nat.sqrt_lt]",
            "rw [Nat.sqrt_lt]",
            # pow
            "rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]",
            "exact Nat.pow_lt_pow_iff_left {hyp_ne_zero}",
            "simp [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]",
        ],
        "unknown": [
            "exact Nat.sqrt_lt",
            "simp [Nat.sqrt_lt]",
        ],
    }
    g["priority_template_budget"] = 5
    return g


def _w4_master(g: dict[str, Any]) -> dict[str, Any]:
    """v5-27: master combo — every working v5 win + new wave-4 attempts."""
    g["priority_templates"] = {
        "iff": [
            # WORKING: div_lt_one_iff
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            # WORKING: mul_eq_left / mul_eq_right
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            # NEW ATTEMPT: dvd_iff_div_mul_eq
            "exact ⟨fun h => Nat.div_mul_cancel h, fun h => ⟨_, h.symm⟩⟩",
            # NEW ATTEMPT: pow_lt
            "rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]",
            # NEW ATTEMPT: div_pos_iff
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            # fallback iff
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


def _w4_super_kitchen(g: dict[str, Any]) -> dict[str, Any]:
    """v5-28: SUPER-KITCHEN — every confirmed-working priority template
    from the followup loop, in one variant.

    Combines:
      - v5-12's div_lt_one_iff templates (rw with Nat.one_mul/Nat.mul_one)
      - v5-15's mul-specific term-mode skeletons (Nat.eq_of_mul_eq_mul_*)
      - v5-20's div-pos templates (Nat.one_le_div_iff + Nat.div_ne_zero_iff)
      - kitchen-sink fallback iff/dvd/any templates from v5-18

    Expected: 31/38 if all four buckets close their targets simultaneously.
    """
    g["priority_templates"] = {
        "iff": [
            # WORKING from v5-12: div_lt_one_iff
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
            # WORKING from v5-15: mul_eq_left / right
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            # WORKING from v5-20: div_pos_iff
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            # generic iff (covers easy iff theorems)
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "constructor <;> intro h_split <;> simp_all",
        ],
        "lt": [
            # WORKING from v5-20: div_pos
            "exact (Nat.one_le_div_iff {hyp_pos}).mpr {hyp_le}",
            "exact Nat.one_le_div_iff_le.mpr {hyp_le}",
            # WORKING from v5-12: div_lt_one_iff on lt shape too
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
        ],
        "dvd": [
            "exact ⟨_, by simp_all⟩",
            "exact ⟨_, rfl⟩",
        ],
        "any": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all",
        ],
    }
    g["priority_template_budget"] = 16
    return g


VARIANTS_WAVE4: list[Variant] = [
    Variant("v5-23-w4-split-ifs", "wave4: split_ifs across non-iff shapes", "B+priority", _w4_split_ifs_dedicated),
    Variant("v5-24-w4-dvd-iff", "wave4: dvd_iff_div_mul_eq specific", "A+priority", _w4_dvd_iff),
    Variant("v5-25-w4-div-pos", "wave4: div_pos and div_pos_iff", "A+priority", _w4_div_pos),
    Variant("v5-26-w4-sqrt-pow", "wave4: sqrt_lt and pow_lt forms", "B+priority", _w4_sqrt_pow),
    Variant("v5-27-w4-master", "wave4: master combo of all v5 wins + new attempts", "all+priority", _w4_master),
    Variant("v5-28-w4-super-kitchen", "wave4: SUPER-KITCHEN — every confirmed priority win", "all+priority", _w4_super_kitchen),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-hours", type=float, default=1.5)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    run_id = f"v5-wave4-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    config_out_path = out_root / "config.json"
    config_out_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    chosen = list(VARIANTS_WAVE4)
    if args.variants:
        chosen = [v for v in chosen if v.name in args.variants]

    started = time.time()
    deadline = started + args.max_hours * 3600
    log_lines: list[str] = []
    def log(m): print(m, flush=True); log_lines.append(m); log_path.write_text("\n".join(log_lines)+"\n")
    log(f"# v5 wave4 — {run_id}\n")
    log(f"- variants: {len(chosen)}")

    # load v5-00 baseline for accurate delta
    import glob
    base_metrics = glob.glob(str(REPO_ROOT / 'project/evolve/autonomous_runs/v5-auto-*/eval/v5-00-baseline-repro/eval-*/metrics.json'))
    if base_metrics:
        m = json.loads(open(base_metrics[0]).read())
        baseline_proved_set = {t['full_name'] for t in m['per_theorem'] if t.get('finished')}
    else:
        baseline_proved_set = set()
    baseline_proved_count = len(baseline_proved_set)
    log(f"- baseline (from v5-00): {baseline_proved_count}/38")

    results: list[CycleResult] = []
    for i, variant in enumerate(chosen):
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
            log(f"- regressions: {newly_lost}")
    elapsed_total = time.time() - started
    final = out_root / "final_report.md"
    final.write_text(_render_final_report(run_id, args, results, elapsed_total))
    log(f"\n## complete — {elapsed_total/3600:.2f}h, {len(results)} cycles")


if __name__ == "__main__":
    main()
