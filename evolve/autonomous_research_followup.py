"""Follow-up autonomous-research variants for the v5 second pass.

Adds priority_templates (run before generative_topk) so a known-good
family template fires at step 1 before the model's `simp [...]` can
derail the rewrite chain. Also explores a few targeted mul/dvd/sqrt
templates that the v3.6 → v5-11 first pass was missing.
"""
from __future__ import annotations

import argparse
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evolve.autonomous_research_loop import (
    CycleResult, REPO_ROOT, Variant, VARIANTS_DEFAULT, baseline_genome,
    write_strategy_config, run_eval, summarize_metrics,
    _render_final_report,
)
import json


def _priority_div_hyp_pos(g: dict[str, Any]) -> dict[str, Any]:
    """v5-12: priority_templates with {hyp_pos} div rewrites. These
    fire BEFORE generative_topk so the model's `simp [...]` can't
    derail the goal at step 1."""
    g["priority_templates"] = {
        "iff": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
            "simp [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        ],
        "lt": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
        ],
    }
    g["priority_template_budget"] = 4
    return g


def _priority_iff_constructor(g: dict[str, Any]) -> dict[str, Any]:
    """v5-13: priority constructor split for iff goals. Run BEFORE
    generative_topk to avoid weak-simp derailment."""
    g["priority_templates"] = {
        "iff": [
            "constructor <;> intro h_split <;> omega",
            "constructor <;> intro h_split <;> simp_all",
            "constructor <;> intro h_split <;> (first | omega | simp_all)",
        ],
    }
    g["priority_template_budget"] = 3
    return g


def _priority_combo(g: dict[str, Any]) -> dict[str, Any]:
    """v5-14: combo of priority div_hyp_pos + family mul_eq +
    split_ifs + term_builder iff basic. The most-fully-equipped
    candidate from the first pass + priority hooks."""
    g["priority_templates"] = {
        "iff": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "constructor <;> intro h_split <;> simp_all",
        ],
    }
    g["priority_template_budget"] = 6
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp [h]⟩",
            "exact ⟨fun h => by subst h; simp_all, fun h => by simp [h]⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> (first | omega | simp_all)",
        ],
    }
    g["term_builder_budget"] = 6
    g["fallback_tactics"] = list(g["fallback_tactics"]) + [
        "split_ifs <;> omega",
        "split_ifs <;> simp_all",
    ]
    # mul family
    g["theorem_family_tactics"]["mul_eq"] = [
        "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
        "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
        "constructor <;> intro h_split <;> simp_all",
        "omega",
    ]
    g["family_budgets"]["mul_eq"] = 6
    return g


def _priority_mul_specific(g: dict[str, Any]) -> dict[str, Any]:
    """v5-15: priority templates for mul_eq iff specifically — uses
    Nat.eq_of_mul_eq_mul_left/_right as term-mode skeletons."""
    g["priority_templates"] = {
        "iff": [
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
        ],
    }
    g["priority_template_budget"] = 4
    return g


def _priority_sqrt_pow(g: dict[str, Any]) -> dict[str, Any]:
    """v5-16: priority templates for sqrt and pow iff goals."""
    g["priority_templates"] = {
        "iff": [
            "exact Nat.sqrt_lt'",
            "simp [Nat.sqrt_lt']",
            "rw [Nat.sqrt_lt']",
        ],
    }
    g["priority_template_budget"] = 3
    return g


def _priority_term_iff_advanced(g: dict[str, Any]) -> dict[str, Any]:
    """v5-17: priority term_builder iff with omega/simp_all/subst."""
    g["priority_templates"] = {
        "iff": [
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp [h]⟩",
            "exact ⟨fun h => by subst h; simp_all, fun h => by simp [h]⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> (first | omega | simp_all)",
        ],
    }
    g["priority_template_budget"] = 6
    return g


def _priority_kitchen_sink(g: dict[str, Any]) -> dict[str, Any]:
    """v5-18: kitchen sink — every plausible priority template for
    each goal shape. High budget. The most aggressive candidate."""
    g["priority_templates"] = {
        "iff": [
            # div
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
            # mul_eq
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            # sqrt
            "exact Nat.sqrt_lt'",
            "simp [Nat.sqrt_lt']",
            # term-mode generic
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp [h]⟩",
            # constructor
            "constructor <;> intro h_split <;> omega",
            "constructor <;> intro h_split <;> simp_all",
        ],
        "dvd": [
            "exact ⟨_, by simp_all⟩",
            "exact ⟨_, rfl⟩",
        ],
        "lt": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        ],
        # for Nat.add_mod_eq_ite: the goal classifier returns "unknown"
        # (no `↔` or other head connective), so attach the split_ifs
        # rescue templates to the "any" key so they fire on every state
        # we couldn't classify and the wrapper hasn't otherwise routed.
        "any": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all",
            "split_ifs <;> simp_all <;> omega",
        ],
    }
    g["priority_template_budget"] = 12
    # also include the basic term_builder for backwards compat
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by simp_all, fun h => by simp [h]⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> (first | omega | simp_all)",
        ],
    }
    g["term_builder_budget"] = 4
    g["fallback_tactics"] = list(g["fallback_tactics"]) + [
        "split_ifs <;> omega",
        "split_ifs <;> simp_all",
    ]
    return g


def _priority_split_ifs_any(g: dict[str, Any]) -> dict[str, Any]:
    """v5-19: priority split_ifs under "any" key — fires on all
    non-iff/non-dvd states. Targets Nat.add_mod_eq_ite."""
    g["priority_templates"] = {
        "any": [
            "split_ifs <;> omega",
            "split_ifs <;> simp_all",
            "split_ifs <;> simp_all <;> omega",
        ],
    }
    g["priority_template_budget"] = 3
    return g


def _priority_div_pos(g: dict[str, Any]) -> dict[str, Any]:
    """v5-20: priority for Nat.div_pos (lt shape) and Nat.div_pos_iff
    (iff shape). Tries the one_le_div lemma + mpr."""
    g["priority_templates"] = {
        "lt": [
            "exact (Nat.one_le_div_iff {hyp_pos}).mpr {hyp_le}",
            "rw [Nat.lt_iff_add_one_le, Nat.add_one_le_iff_lt]",
            "exact Nat.one_le_iff_ne_zero.mpr (fun h => by simp [Nat.div_eq_of_lt, h] at *)",
        ],
        "iff": [
            "rw [Nat.pos_iff_ne_zero, Nat.div_ne_zero_iff {hyp_ne_zero}]",
            "rw [Nat.pos_iff_ne_zero]; exact Nat.div_ne_zero_iff {hyp_ne_zero}",
            "exact ⟨fun h => Nat.not_lt.mp (fun hc => by simp [Nat.div_eq_of_lt hc] at h), fun h => Nat.div_pos h (Nat.pos_of_ne_zero {hyp_ne_zero})⟩",
        ],
    }
    g["priority_template_budget"] = 6
    return g


def _deny_derailing_simps(g: dict[str, Any]) -> dict[str, Any]:
    """v5-22: explicitly deny `simp [Nat.one_mul]` / `simp [Nat.mul_one]`
    on the failing iff/lt theorems so the family rewrites get a chance.
    Then add priority_templates for those theorems. The deny-list
    complements priority_templates — it removes the derailing tactics
    even from generative_topk's output."""
    g["theorem_tactic_denylist"] = dict(g.get("theorem_tactic_denylist", {}))
    derailers = [
        "simp [Nat.one_mul]",
        "simp [Nat.mul_one]",
        "simp [Nat.sub_self]",
        "simp [List.length_cons]",
        "simp [List.map]",
        "simp [List.filter]",
    ]
    for thm in [
        "Nat.div_lt_one_iff",
        "Nat.div_pos_iff",
        "Nat.div_pos",
        "Nat.mul_eq_left",
        "Nat.mul_eq_right",
        "Nat.sqrt_lt",
        "Nat.pow_lt_pow_iff_left",
        "Nat.dvd_iff_div_mul_eq",
    ]:
        g["theorem_tactic_denylist"][thm] = list(
            g["theorem_tactic_denylist"].get(thm, [])
        ) + derailers
    # Also add the priority_templates so the rw fires at step 1
    g["priority_templates"] = {
        "iff": [
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.one_mul]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.one_mul]",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_left (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.mul_one _).symm), fun h => by simp [h]⟩",
            "exact ⟨fun h => Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero {hyp_ne_zero}) (h.trans (Nat.one_mul _).symm), fun h => by simp [h]⟩",
            "exact Nat.sqrt_lt'",
            "simp [Nat.sqrt_lt']",
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> simp_all",
        ],
    }
    g["priority_template_budget"] = 8
    return g


def _priority_iff_omega_simp(g: dict[str, Any]) -> dict[str, Any]:
    """v5-21: minimal priority — just the term_builder iff basic templates,
    but EMITTED BEFORE generative_topk. This is the cleanest test of whether
    forcing term-mode at step 1 changes anything."""
    g["priority_templates"] = {
        "iff": [
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> omega",
            "refine ⟨?_, ?_⟩ <;> intro h <;> simp_all",
            "refine ⟨?_, ?_⟩ <;> intro h <;> (first | omega | simp_all)",
        ],
    }
    g["priority_template_budget"] = 4
    return g


VARIANTS_FOLLOWUP: list[Variant] = [
    Variant("v5-12-prio-div-hyp", "priority div rewrites with hyp_pos", "B+priority", _priority_div_hyp_pos),
    Variant("v5-13-prio-iff-constructor", "priority iff constructor split", "A+priority", _priority_iff_constructor),
    Variant("v5-14-prio-combo", "priority + family + term + split_ifs", "A+B+priority", _priority_combo),
    Variant("v5-15-prio-mul-specific", "priority mul_eq term-mode skeletons", "B+priority", _priority_mul_specific),
    Variant("v5-16-prio-sqrt-pow", "priority sqrt and pow", "B+priority", _priority_sqrt_pow),
    Variant("v5-17-prio-term-iff", "priority term_builder iff (advanced)", "A+priority", _priority_term_iff_advanced),
    Variant("v5-18-prio-kitchen", "kitchen sink: every priority template", "all+priority", _priority_kitchen_sink),
    Variant("v5-19-prio-split-ifs", "priority split_ifs under 'any' key", "B+priority", _priority_split_ifs_any),
    Variant("v5-20-prio-div-pos", "priority div_pos and div_pos_iff", "B+priority", _priority_div_pos),
    Variant("v5-21-prio-iff-basic", "priority iff term_builder basic (minimal)", "A+priority", _priority_iff_omega_simp),
    Variant("v5-22-deny-derailers", "deny derailing simps + priority kitchen", "B+priority+deny", _deny_derailing_simps),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--baseline-metrics", type=Path, default=None,
                        help="Compare delta against this metrics.json instead "
                             "of the first cycle's result. Useful when this "
                             "followup loop is comparing against a prior run.")
    args = parser.parse_args()

    run_id = f"v5-followup-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    best_path = out_root / "best_candidate.json"
    config_out_path = out_root / "config.json"
    config_out_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    chosen = list(VARIANTS_FOLLOWUP)
    if args.variants:
        chosen = [v for v in chosen if v.name in args.variants]

    started = time.time()
    deadline = started + args.max_hours * 3600
    log_lines: list[str] = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"# v5 followup run — {run_id}\n")
    log(f"- variants queued: {len(chosen)}")

    results: list[CycleResult] = []
    baseline_proved_set: set[str] | None = None
    baseline_proved_count: int = 26  # known from first pass
    if args.baseline_metrics:
        m = json.loads(args.baseline_metrics.read_text())
        baseline_proved_set = {t['full_name'] for t in m['per_theorem'] if t.get('finished')}
        baseline_proved_count = len(baseline_proved_set)
        log(f"- baseline from {args.baseline_metrics}: {baseline_proved_count}/38")

    for i, variant in enumerate(chosen):
        now = time.time()
        if now >= deadline:
            log("\n## stopping — hit max-hours")
            break
        remaining = deadline - now
        per_eval_to = min(args.per_eval_timeout_seconds, max(120, int(remaining)))
        log(f"\n## cycle {i+1} — {variant.name}  [{variant.direction}]")
        log(f"- {variant.description}")
        log(f"- elapsed: {(now-started)/3600:.2f}h  remaining: {remaining/3600:.2f}h")
        metrics_path = run_eval(variant, args.theorem_set, args.ckpt_dir, eval_root, per_eval_to)
        if metrics_path is None:
            log("- FAILED")
            continue
        (proved, progress, errored, pbo, tb_a, tb_adv, tb_p,
         newly_proved, newly_lost, proved_set) = summarize_metrics(
            metrics_path, baseline_proved_set,
        )
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
        with scoreboard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cyc.to_dict(), ensure_ascii=False) + "\n")
        log(f"- proved: {proved} (Δ {delta:+d})  prog: {progress}  err: {errored}")
        log(f"- origins: {pbo}")
        if tb_a:
            log(f"- term_builder: {tb_a}/{tb_adv}/{tb_p}")
        if newly_proved:
            log(f"- NEW WINS: {newly_proved}")
        if newly_lost:
            log(f"- REGRESSIONS: {newly_lost}")

    elapsed_total = time.time() - started
    if results:
        best = max(results, key=lambda r: (r.proved, r.progress, -r.errored))
        best_path.write_text(json.dumps(best.to_dict(), indent=2), encoding="utf-8")
        log(f"\n## complete — {elapsed_total/3600:.2f}h, {len(results)} cycles, best={best.name} proved={best.proved}")
    final = out_root / "final_report.md"
    final.write_text(_render_final_report(run_id, args, results, elapsed_total), encoding="utf-8")
    print(f"\n[autonomous_research_followup] complete: {out_root}")


if __name__ == "__main__":
    main()
