"""NS3 — targeted lemma-name retrieval for the 7 remaining failures.

Each variant builds on v5-27-w4-master's genome and patches `priority_templates`
with verbatim Mathlib-proof templates for one of the promising failures
identified in `project/evolve/reports/ns3_lemma_audit.md`:

  - ns3-dvd          : fix `Nat.dvd_iff_div_mul_eq` iff template (use the
                       actual Mathlib proof body, not the broken `⟨_, h.symm⟩`).
  - ns3-eq-one-mul   : new `eq` slot for `Nat.eq_one_of_mul_eq_one_left`.
  - ns3-add-mod-ite  : multi-step `rw + split_ifs` for `Nat.add_mod_eq_ite`
                       (likely BLOCKED at single-line granularity; gated by
                       deny-list to prevent any DojoCrash propagating).
  - ns3-div-le       : new `le` slot for `Nat.div_le_div_right` using `gcongr`
                       plus an explicit case-split form.
  - ns3-sqrt-pow     : fix `Nat.sqrt_lt` and `Nat.pow_lt_pow_iff_left`
                       iff templates (use Mathlib's simp-only forms;
                       no self-reference).
  - ns3-combined     : all of the above merged.

Constraints honoured: no schema change, no wrapper change, no checkpoint
touch. Each variant only edits `priority_templates` / `priority_template_budget`
/ `theorem_tactic_denylist`.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evolve.autonomous_research_loop import (
    CycleResult, REPO_ROOT, Variant, baseline_genome,
    run_eval, summarize_metrics, _render_final_report,
)

# v5-27-w4-master's genome path (kept stable across NS-stages).
V5_27_GENOME = (
    REPO_ROOT
    / "project/evolve/autonomous_runs/v5-wave4-20260522-111556-3063e7"
    / "eval/v5-27-w4-master/genome.json"
)


def v5_27_master_genome() -> dict[str, Any]:
    """Return a fresh deepcopy of v5-27-w4-master's full genome."""
    base = baseline_genome()
    base.update(json.loads(V5_27_GENOME.read_text()))
    return deepcopy(base)


# --- candidate patches ----------------------------------------------------


def _ns3_dvd(g: dict[str, Any]) -> dict[str, Any]:
    """Replace the broken dvd template in the iff slot with the verbatim
    Mathlib proof of `Nat.dvd_iff_div_mul_eq`."""
    pt = deepcopy(g["priority_templates"])
    iff_slot = list(pt.get("iff", []))
    # Drop the broken `⟨_, h.symm⟩` form.
    broken = "exact ⟨fun h => Nat.div_mul_cancel h, fun h => ⟨_, h.symm⟩⟩"
    iff_slot = [t for t in iff_slot if t != broken]
    # Add the Mathlib-verbatim form.
    fixed = "exact ⟨fun h => Nat.div_mul_cancel h, fun h => by rw [← h]; exact Nat.dvd_mul_left _ _⟩"
    if fixed not in iff_slot:
        iff_slot.insert(0, fixed)
    pt["iff"] = iff_slot
    g["priority_templates"] = pt
    return g


def _ns3_eq_one_mul(g: dict[str, Any]) -> dict[str, Any]:
    """Add an `eq` slot with the Mathlib proof of
    `Nat.eq_one_of_mul_eq_one_left`, plus a generic-monoid `mul_eq_one`
    fallback."""
    pt = deepcopy(g["priority_templates"])
    pt["eq"] = [
        "exact Nat.eq_one_of_mul_eq_one_right (by rwa [Nat.mul_comm])",
        "simp_all [mul_eq_one]",
    ]
    g["priority_templates"] = pt
    g["priority_template_budget"] = max(g.get("priority_template_budget", 14), 16)
    return g


def _ns3_add_mod_ite(g: dict[str, Any]) -> dict[str, Any]:
    """Try a single-line multi-step template for `Nat.add_mod_eq_ite`.
    Gated by `theorem_tactic_denylist` for safety: if it DojoCrashes the
    deny list will keep the per-theorem run from poisoning the eval
    process. Lives in the `any` slot."""
    pt = deepcopy(g["priority_templates"])
    any_slot = list(pt.get("any", []))
    # The Mathlib proof body, compressed onto one line with `;` separators.
    # split_ifs with h then per-branch `<;>` collapsed to a single
    # asymmetric `first | exact ... | exact ...` chain.
    candidates = [
        # Compressed Mathlib body — relies on Lean 4 accepting `· ...; · ...`
        # in a single tactic line. Likely fails to parse.
        "cases k <;> [skip; (rw [Nat.add_mod]; split_ifs with h <;> "
        "first | (exact Nat.mod_eq_of_lt (Nat.lt_of_not_ge h)) | "
        "(rw [Nat.mod_eq_sub_mod h]; exact Nat.mod_eq_of_lt "
        "((Nat.sub_lt_iff_lt_add h).mpr (Nat.add_lt_add "
        "(Nat.mod_lt _ (by omega)) (Nat.mod_lt _ (by omega))))))]",
        # Simpler attempt — just `rw + split_ifs` and let the model fill in.
        "rw [Nat.add_mod]; split_ifs with h <;> "
        "simp [Nat.mod_eq_of_lt, Nat.mod_eq_sub_mod, Nat.lt_of_not_ge, h]",
    ]
    for c in candidates:
        if c not in any_slot:
            any_slot.append(c)
    pt["any"] = any_slot
    g["priority_templates"] = pt
    # Safety net: if any candidate crashes Dojo we'll know to deny it.
    deny = deepcopy(g.get("theorem_tactic_denylist", {}))
    # Pre-emptively keep the existing add_mod_eq_ite deny entry (it bans
    # `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` which the v3 traces showed
    # was a DojoCrash trigger).
    g["theorem_tactic_denylist"] = deny
    g["priority_template_budget"] = max(g.get("priority_template_budget", 14), 18)
    return g


def _ns3_div_le(g: dict[str, Any]) -> dict[str, Any]:
    """Add a `le` slot for `Nat.div_le_div_right` — gcongr first, then
    explicit case-split form using `Nat.le_div_iff_mul_le'`."""
    pt = deepcopy(g["priority_templates"])
    pt["le"] = [
        "gcongr",
        # Explicit case-split: handles c=0 via simp and c>0 via the
        # le_div_iff_mul_le' transport. Uses `{hyp_le}` if h is named
        # via the placeholder, else falls back to `‹_›` (in-scope
        # search).
        "by_cases hc : c = 0 <;> "
        "[simp [hc]; exact (Nat.le_div_iff_mul_le' (Nat.pos_of_ne_zero hc)).2 "
        "(Nat.le_trans (Nat.div_mul_le_self _ _) {hyp_le})]",
    ]
    g["priority_templates"] = pt
    g["priority_template_budget"] = max(g.get("priority_template_budget", 14), 16)
    return g


def _ns3_sqrt_pow(g: dict[str, Any]) -> dict[str, Any]:
    """Replace the self-referential `rw [Nat.pow_lt_pow_iff_left ...]`
    template in the iff slot with Mathlib's `simp only [...]` body,
    AND add a sqrt_lt template using `Nat.le_sqrt`.

    The new templates are APPENDED, not prepended, because their
    ``simp only [← Nat.not_le, ...]`` form has a destructive side
    effect on unrelated iff goals (it flips any ``<`` in the goal to
    ``¬ ≤``, which derails the existing `rw [Nat.pos_iff_ne_zero,
    Nat.div_ne_zero_iff ...]` template that closes ``Nat.div_pos_iff``).
    By appending, the existing specific templates run first; the
    sqrt/pow templates only fire when the existing ones fail to match
    (which they do on sqrt-shaped or pow-shaped goals because they
    reference ``/``, ``Nat.div_lt_iff_lt_mul`` etc. that aren't in
    those goals)."""
    pt = deepcopy(g["priority_templates"])
    iff_slot = list(pt.get("iff", []))
    # Drop self-referential pow template (matches if present).
    self_ref = "rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]"
    iff_slot = [t for t in iff_slot if t != self_ref]
    # Append the Mathlib-verbatim forms LAST among the specifics so
    # they don't shadow other working templates. NS1's stable-sort
    # still groups specifics before generics.
    pow_fix = "simp only [← Nat.not_le, Nat.pow_le_pow_iff_left {hyp_ne_zero}]"
    sqrt_fix = "simp only [← Nat.not_le, Nat.le_sqrt]"
    # Insert just before the first generic (so they stay grouped with
    # specifics in declared order).
    from evolve.strategy_wrapper import classify_template_specificity
    insert_at = len(iff_slot)
    for i, t in enumerate(iff_slot):
        if classify_template_specificity(t)[1] == "generic":
            insert_at = i
            break
    for t in (sqrt_fix, pow_fix):
        if t not in iff_slot:
            iff_slot.insert(insert_at, t)
            insert_at += 1
    pt["iff"] = iff_slot
    g["priority_templates"] = pt
    return g


def _ns3_combined(g: dict[str, Any]) -> dict[str, Any]:
    """All NS3 patches stacked. Order: dvd → sqrt-pow → eq → div-le →
    add-mod (highest risk last; lowest probability of DojoCrash effect
    if it does crash).

    Cross-shape broadcast: the add_mod_eq_ite multi-step template is
    also mirrored into the `eq` slot, because the wrapper's shape gate
    is exclusive — once an `eq` slot exists, eq-shaped goals never fall
    back to `any`. ``Nat.add_mod_eq_ite`` classifies as `eq` (it's an
    equation), so without this mirror the new eq slot would shadow the
    add_mod multi-step template that ns3-add-mod-ite proved closes it.
    """
    g = _ns3_dvd(g)
    g = _ns3_sqrt_pow(g)
    g = _ns3_eq_one_mul(g)
    g = _ns3_div_le(g)
    g = _ns3_add_mod_ite(g)
    # Cross-shape broadcast: the wrapper's shape gate is exclusive —
    # once a specific shape slot exists, goals of that shape never fall
    # back to `any`. ``Nat.add_mod_eq_ite`` classifies as `le` (its
    # if-then-else contains a `≤` inside) and could in principle hit
    # `eq` too, so we mirror every `any` template into both. The shape
    # classifier checks ``≤`` before ``=``, so in practice it's `le`,
    # but mirroring into eq is cheap and keeps the broadcast complete.
    pt = deepcopy(g["priority_templates"])
    any_slot = pt.get("any", [])
    for target_shape in ("eq", "le"):
        target_slot = list(pt.get(target_shape, []))
        for t in any_slot:
            if t not in target_slot:
                target_slot.append(t)
        pt[target_shape] = target_slot
    g["priority_templates"] = pt
    g["priority_template_budget"] = max(g.get("priority_template_budget", 14), 24)
    return g


# --- variant registry -----------------------------------------------------

# Each variant's `apply` ignores the caller-supplied genome (which would
# be v4.7 baseline) and re-seeds from v5-27-w4-master instead, so NS3
# starts from the NS1-stable best rather than the wave-0 baseline.

def _seed_from_v5_27(patch: Callable[[dict[str, Any]], dict[str, Any]]):
    def apply(_g: dict[str, Any]) -> dict[str, Any]:
        return patch(v5_27_master_genome())
    return apply


VARIANTS_NS3: list[Variant] = [
    Variant("ns3-dvd",          "NS3: fix dvd_iff_div_mul_eq template",          "A+priority", _seed_from_v5_27(_ns3_dvd)),
    Variant("ns3-eq-one-mul",   "NS3: new eq slot for eq_one_of_mul_eq_one_left", "A+priority", _seed_from_v5_27(_ns3_eq_one_mul)),
    Variant("ns3-add-mod-ite",  "NS3: multi-step add_mod_eq_ite template",       "C+priority", _seed_from_v5_27(_ns3_add_mod_ite)),
    Variant("ns3-div-le",       "NS3: new le slot for div_le_div_right (gcongr)", "A+priority", _seed_from_v5_27(_ns3_div_le)),
    Variant("ns3-sqrt-pow",     "NS3: fix sqrt_lt + pow_lt_pow_iff_left",        "A+priority", _seed_from_v5_27(_ns3_sqrt_pow)),
    Variant("ns3-combined",     "NS3: all promising patches stacked",            "all+priority", _seed_from_v5_27(_ns3_combined)),
]


# --- driver ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variants by name; default = all.")
    args = parser.parse_args()

    run_id = f"v5-ns3-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    config_out_path = out_root / "config.json"
    config_out_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    chosen = list(VARIANTS_NS3)
    if args.variants:
        chosen = [v for v in chosen if v.name in args.variants]

    started = time.time()
    deadline = started + args.max_hours * 3600
    log_lines: list[str] = []
    def log(m: str) -> None:
        print(m, flush=True)
        log_lines.append(m)
        log_path.write_text("\n".join(log_lines) + "\n")
    log(f"# NS3 — {run_id}\n")
    log(f"- variants: {len(chosen)}")

    # Baseline = v5-27 master proved-set (the 31 we already have).
    baseline_metrics_path = sorted(
        (REPO_ROOT / "project/evolve/autonomous_runs/v5-wave4-20260522-111556-3063e7"
         "/eval/v5-27-w4-master").glob("eval-*/metrics.json")
    )
    if baseline_metrics_path:
        m = json.loads(baseline_metrics_path[0].read_text())
        baseline_proved_set = {
            t["full_name"] for t in m["per_theorem"] if t.get("winning_tactic")
        }
    else:
        baseline_proved_set = set()
    baseline_proved_count = len(baseline_proved_set)
    log(f"- baseline (v5-27 master): {baseline_proved_count}/38")

    results: list[CycleResult] = []
    for i, variant in enumerate(chosen):
        now = time.time()
        if now >= deadline:
            log("\n## stopping — max-hours")
            break
        log(f"\n## cycle {i+1} — {variant.name}")
        log(f"- {variant.description}")

        # The variant's `apply` already re-seeds from v5-27 master via
        # `_seed_from_v5_27`, so we can call run_eval directly.
        metrics_path = run_eval(
            variant, args.theorem_set, args.ckpt_dir, eval_root,
            args.per_eval_timeout_seconds,
        )
        if metrics_path is None:
            log("- FAILED")
            continue
        (proved, progress, errored, pbo, tb_a, tb_adv, tb_p,
         newly_proved, newly_lost, proved_set) = summarize_metrics(
            metrics_path, baseline_proved_set,
        )
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
            f.write(json.dumps(cyc.to_dict(), ensure_ascii=False) + "\n")
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
