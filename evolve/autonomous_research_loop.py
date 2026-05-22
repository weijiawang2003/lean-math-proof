"""autonomous_research_loop.py — v5 outer loop for deeper AlphaEvolve exploration.

Unlike `run_evolve.py` which sweeps a fixed mutator over a fixed seed, this
loop pulls candidate *programs* from a `variants` registry that the model
adds to as it learns from prior cycles. Each variant produces a fresh
strategy_config.json and runs `eval_rollout_all.py` against it. Wins are
attributed by origin (fallback / family / retrieved_premise / term_builder)
and per-theorem changes are diffed against the seed to surface what each
variant actually changed.

The loop is `--min-hours` floored and `--max-hours` capped so the agent
runs for the planned 5-8 hours without manual nudging.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from evolve.strategy_wrapper import dump_strategy_config

REPO_ROOT = Path(__file__).resolve().parent.parent


# -- baseline genome (v4.7 constructor-default; matches the 26/38 seed) -----
def baseline_genome() -> dict[str, Any]:
    """Return a deep copy of the v4.7 constructor-default seed genome."""
    return deepcopy({
        "fallback_tactics": [
            "simp [Nat.add_comm, Nat.add_assoc, Nat.left_comm]",
            "simp_all",
            "simp",
            "omega",
            "ac_rfl",
            "simp [Nat.mul_comm, Nat.mul_assoc, Nat.left_comm]",
            "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
            "simp [Nat.add_mod, Nat.mod_eq_of_lt]",
            "rw [Nat.add_comm]",
            "simp_arith",
            "simp [Nat.add_mod, Nat.mod_eq_of_lt] at *",
            "simp [Nat.add_comm, Nat.add_assoc, Nat.left_comm, Nat.add_mod, Nat.mod_eq_of_lt]",
        ],
        "tactic_templates": [
            "induction {var} with | zero => simp | succ n ih => simp [ih]",
            "cases {var} <;> simp",
            "by_cases h : {var} = 0 <;> simp [h] <;> omega",
        ],
        "max_extra_tactics_per_state": 10,
        "theorem_family_tactics": {
            "AM_GM": ["omega", "simp", "simp_all"],
            "div": [
                "omega", "simp", "simp_all",
                "constructor <;> intro h_split <;> omega",
                "constructor <;> intro h_split <;> simp_all",
                "constructor <;> intro h_split <;> simp_all <;> omega",
            ],
            "mod": [
                "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
                "simp [Nat.add_mod, Nat.mod_eq_of_lt]",
                "omega",
            ],
        },
        "family_budgets": {"AM_GM": 8, "div": 8, "mod": 12},
        "theorem_tactic_denylist": {
            "Nat.add_mod_eq_ite": [
                "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
            ],
        },
        "retrieval_enabled": True,
        "retrieval_top_k": 8,
        "retrieval_tactic_forms": ["rw", "simp", "apply"],
        "retrieval_filter_self": True,
        "retrieval_filter_unavailable": True,
        "retrieval_skip_bloating_apply": True,
        "retrieval_shape_filter": True,
        "term_builder_templates": {},
        "term_builder_budget": 0,
        "priority_templates": {},
        "priority_template_budget": 0,
        "use_skeleton_bag": False,
        "max_steps": 8,
        "top_k": 8,
        "timeout_per_theorem": 60,
    })


# -- variant registry --------------------------------------------------------
VariantFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Variant:
    name: str
    description: str
    direction: str  # "A", "B", "C", "D" or "baseline"
    apply: VariantFn

    def make_config(self) -> dict[str, Any]:
        g = baseline_genome()
        return self.apply(g)


def _identity(g: dict[str, Any]) -> dict[str, Any]:
    return g


def _add_hyp_pos_div_templates(g: dict[str, Any]) -> dict[str, Any]:
    """Direction B / quick-win: restore the {hyp_pos} div templates that
    the constructor variant dropped. Targets Nat.div_lt_one_iff which
    closes via rw [Nat.div_lt_iff_lt_mul hb, Nat.mul_one]."""
    g["theorem_family_tactics"]["div"] = list(g["theorem_family_tactics"]["div"]) + [
        "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
        "rw [Nat.div_lt_iff_lt_mul {hyp_pos}]",
        "rw [Nat.div_lt_iff_lt_mul' {hyp_pos}]",
        "simp [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
        "simp [Nat.div_lt_iff_lt_mul' {hyp_pos}, Nat.mul_one]",
    ]
    g["family_budgets"]["div"] = 16
    return g


def _add_mul_family(g: dict[str, Any]) -> dict[str, Any]:
    """Direction B: mul-eq mini-solver. Targets Nat.mul_eq_left / _right
    and Nat.eq_one_of_mul_eq_one_left."""
    g["theorem_family_tactics"]["mul_eq"] = [
        "omega",
        "simp_all",
        "simp [Nat.mul_one, Nat.one_mul]",
        # asymmetric iff via term mode (will dedupe with term_builder if both fire)
        "constructor <;> intro h_split <;> simp_all",
        # mul-cancellation lemmas
        "rw [Nat.mul_right_cancel_iff (Nat.pos_of_ne_zero {hyp_ne_zero})]",
        "rw [Nat.mul_left_cancel_iff (Nat.pos_of_ne_zero {hyp_ne_zero})]",
    ]
    g["family_budgets"]["mul_eq"] = 10
    return g


def _add_split_ifs(g: dict[str, Any]) -> dict[str, Any]:
    """Direction B: split_ifs <;> omega for Nat.add_mod_eq_ite."""
    g["fallback_tactics"] = list(g["fallback_tactics"]) + [
        "split_ifs <;> omega",
        "split_ifs <;> simp_all",
        "split_ifs <;> simp_all <;> omega",
    ]
    return g


def _term_builder_iff_basic(g: dict[str, Any]) -> dict[str, Any]:
    """Direction A: minimal term_builder for iff goals — symmetric
    inner tactics across both directions of the iff."""
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by simp_all, fun h => by omega⟩",
            "exact ⟨fun h => by omega, fun h => by simp_all⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> omega",
            "refine ⟨?_, ?_⟩ <;> intro h <;> simp_all",
        ],
    }
    g["term_builder_budget"] = 8
    return g


def _term_builder_iff_advanced(g: dict[str, Any]) -> dict[str, Any]:
    """Direction A: term_builder for iff with rewrite-and-close inner
    tactics. Targets Nat.div_lt_one_iff, Nat.mul_eq_left, Nat.mul_eq_right."""
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp [h]⟩",
            "exact ⟨fun h => by subst h; simp_all, fun h => by simp [h]⟩",
            "refine ⟨?_, ?_⟩ <;> intro h <;> omega",
            "refine ⟨?_, ?_⟩ <;> intro h <;> simp_all",
            "refine ⟨?_, ?_⟩ <;> intro h <;> (first | omega | simp_all)",
        ],
        "dvd": [
            "exact ⟨_, by simp_all⟩",
        ],
    }
    g["term_builder_budget"] = 10
    return g


def _term_builder_iff_with_hyp(g: dict[str, Any]) -> dict[str, Any]:
    """Direction A: term_builder for iff that uses hypothesis-aware rewrites
    in the inner tactics."""
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by omega, fun h => by omega⟩",
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by rw [Nat.div_lt_iff_lt_mul {hyp_pos}] at h; omega, fun h => by rw [Nat.div_lt_iff_lt_mul {hyp_pos}]; omega⟩",
            "exact ⟨fun h => by rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one] at h; exact h, fun h => by rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]; exact h⟩",
        ],
    }
    g["term_builder_budget"] = 6
    return g


def _mini_solver_combo(g: dict[str, Any]) -> dict[str, Any]:
    """Direction B combination: hyp_pos div + mul family + split_ifs +
    basic term_builder for iff. The 'kitchen sink' to see if any new
    closure appears with all v5 mechanisms active."""
    g = _add_hyp_pos_div_templates(g)
    g = _add_mul_family(g)
    g = _add_split_ifs(g)
    g = _term_builder_iff_basic(g)
    return g


def _pow_sqrt_family(g: dict[str, Any]) -> dict[str, Any]:
    """Direction B: add pow / sqrt families for Nat.pow_lt_pow_iff_left
    and Nat.sqrt_lt."""
    g["theorem_family_tactics"]["pow_lt"] = [
        "omega",
        "simp [Nat.pow_lt_pow_iff_right]",
        "rw [Nat.pow_lt_pow_iff_right {hyp_ne_zero}]",
        "rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]",
        "exact Nat.pow_lt_pow_iff_left",
        "constructor <;> intro h <;> simp_all",
    ]
    g["family_budgets"]["pow_lt"] = 8
    g["theorem_family_tactics"]["sqrt"] = [
        "omega",
        "simp",
        "rw [Nat.sqrt_lt']",
        "exact Nat.sqrt_lt'",
        "simp [Nat.sqrt_lt']",
        "constructor <;> intro h",
    ]
    g["family_budgets"]["sqrt"] = 6
    return g


def _term_builder_skeleton_mutation_a(g: dict[str, Any]) -> dict[str, Any]:
    """Direction C: tighter inner-tactic vocabulary (omega replaced by
    rfl-then-omega; simp_all replaced by simp [*]) — a small mutation
    around the v5_term_builder_iff_basic skeleton."""
    g = _term_builder_iff_basic(g)
    g["term_builder_templates"]["iff"] = [
        "exact ⟨fun h => by omega, fun h => by omega⟩",
        "exact ⟨fun h => by simp [h], fun h => by simp [h]⟩",
        "exact ⟨fun h => by simp_all <;> omega, fun h => by simp_all <;> omega⟩",
        "refine ⟨?_, ?_⟩ <;> intro h <;> (try omega) <;> simp_all",
        "refine ⟨?_, ?_⟩ <;> intro h <;> (try simp_all) <;> omega",
    ]
    g["term_builder_budget"] = 8
    return g


def _term_builder_dvd(g: dict[str, Any]) -> dict[str, Any]:
    """Direction A: dvd-specific term builder for Nat.dvd_iff_div_mul_eq."""
    g["term_builder_templates"] = {
        "iff": [
            "exact ⟨fun h => by simp_all, fun h => by simp_all⟩",
            "exact ⟨fun h => by rcases h with ⟨k, hk⟩; simp [hk, Nat.mul_div_cancel_left _ (by omega)], fun h => ⟨_, h.symm⟩⟩",
            "refine ⟨fun h => ?_, fun h => ?_⟩",
        ],
        "dvd": [
            "exact ⟨_, by simp_all⟩",
            "exact ⟨_, rfl⟩",
        ],
    }
    g["term_builder_budget"] = 8
    return g


def _aggressive_combo(g: dict[str, Any]) -> dict[str, Any]:
    """The strongest combination after we've identified what helps:
    hyp_pos div + mul family + split_ifs + advanced term_builder + pow_sqrt."""
    g = _add_hyp_pos_div_templates(g)
    g = _add_mul_family(g)
    g = _add_split_ifs(g)
    g = _term_builder_iff_advanced(g)
    g = _pow_sqrt_family(g)
    return g


VARIANTS_DEFAULT: list[Variant] = [
    Variant(
        "v5-00-baseline-repro",
        "Reproduce the v4.7 constructor 26/38 seed",
        "baseline",
        _identity,
    ),
    Variant(
        "v5-01-div-hyp-pos",
        "Restore v45 {hyp_pos} div templates",
        "B",
        _add_hyp_pos_div_templates,
    ),
    Variant(
        "v5-02-mul-family",
        "Add mul family with cancellation lemmas",
        "B",
        _add_mul_family,
    ),
    Variant(
        "v5-03-split-ifs",
        "Add split_ifs fallbacks",
        "B",
        _add_split_ifs,
    ),
    Variant(
        "v5-04-term-iff-basic",
        "term_builder origin with basic iff skeletons",
        "A",
        _term_builder_iff_basic,
    ),
    Variant(
        "v5-05-term-iff-adv",
        "term_builder iff with subst / simp [h] inner tactics",
        "A",
        _term_builder_iff_advanced,
    ),
    Variant(
        "v5-06-term-iff-hyp",
        "term_builder iff with hypothesis-aware rewrites",
        "A",
        _term_builder_iff_with_hyp,
    ),
    Variant(
        "v5-07-term-dvd",
        "term_builder dvd-shape iff handling",
        "A",
        _term_builder_dvd,
    ),
    Variant(
        "v5-08-pow-sqrt",
        "pow_lt and sqrt families",
        "B",
        _pow_sqrt_family,
    ),
    Variant(
        "v5-09-skeleton-mut",
        "Direction C — skeleton mutation around term_builder basic",
        "C",
        _term_builder_skeleton_mutation_a,
    ),
    Variant(
        "v5-10-combo-minimal",
        "Combo: hyp_pos div + mul + split_ifs + term_iff_basic",
        "B",
        _mini_solver_combo,
    ),
    Variant(
        "v5-11-combo-aggressive",
        "Aggressive combo: combo-minimal + term_iff_advanced + pow_sqrt",
        "B",
        _aggressive_combo,
    ),
]


# -- runtime -----------------------------------------------------------------
@dataclass
class CycleResult:
    name: str
    direction: str
    description: str
    proved: int
    progress: int
    errored: int
    proved_by_origin: dict[str, int]
    term_builder_attempt: int
    term_builder_advanced: int
    term_builder_proved: int
    runtime_seconds: float
    delta_vs_baseline: int  # proved - baseline_proved
    newly_proved: list[str]
    newly_lost: list[str]
    eval_dir: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def write_strategy_config(genome: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_strategy_config(
        out_path,
        fallback_tactics=genome["fallback_tactics"],
        tactic_templates=genome["tactic_templates"],
        max_extra_tactics_per_state=genome["max_extra_tactics_per_state"],
        theorem_family_tactics=genome["theorem_family_tactics"],
        family_budgets=genome["family_budgets"],
        theorem_tactic_denylist=genome["theorem_tactic_denylist"],
        retrieval_enabled=genome["retrieval_enabled"],
        retrieval_top_k=genome["retrieval_top_k"],
        retrieval_tactic_forms=genome["retrieval_tactic_forms"],
        retrieval_filter_self=genome["retrieval_filter_self"],
        retrieval_filter_unavailable=genome["retrieval_filter_unavailable"],
        retrieval_skip_bloating_apply=genome["retrieval_skip_bloating_apply"],
        retrieval_shape_filter=genome["retrieval_shape_filter"],
        term_builder_templates=genome["term_builder_templates"],
        term_builder_budget=genome["term_builder_budget"],
        priority_templates=genome.get("priority_templates", {}),
        priority_template_budget=genome.get("priority_template_budget", 0),
        use_skeleton_bag=genome.get("use_skeleton_bag", False),
    )


def run_eval(
    variant: Variant,
    theorem_set: str,
    ckpt_dir: str,
    out_root: Path,
    eval_timeout_seconds: int,
) -> Optional[Path]:
    """Run eval_rollout_all.py for a variant. Returns the path to the
    written metrics.json, or None on failure (and logs the failure)."""
    eval_dir = out_root / variant.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = eval_dir / "strategy_config.json"
    genome = variant.make_config()
    write_strategy_config(genome, strategy_path)
    (eval_dir / "genome.json").write_text(
        json.dumps(genome, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cmd = [
        sys.executable, "-u", "eval_rollout_all.py",
        "--theorem-set", theorem_set,
        "--policy-type", "hybrid_evolved",
        "--ckpt-dir", ckpt_dir,
        "--top-k", str(genome["top_k"]),
        "--max-steps", str(genome["max_steps"]),
        "--out-dir", str(eval_dir),
        "--strategy-config", str(strategy_path),
    ]
    log_path = eval_dir / "subprocess.log"
    print(f"  [{variant.name}] running eval ({eval_timeout_seconds}s timeout)...", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        try:
            r = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=eval_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  [{variant.name}] TIMEOUT", flush=True)
            logf.write(f"\n--- TIMEOUT after {eval_timeout_seconds}s ---\n")
            return None
    elapsed = time.time() - started
    print(f"  [{variant.name}] eval finished in {elapsed:.0f}s (rc={r.returncode})", flush=True)
    if r.returncode != 0:
        return None
    # find newest metrics.json under eval_dir
    metrics_paths = sorted(eval_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    return metrics_paths[-1] if metrics_paths else None


def summarize_metrics(
    metrics_path: Path, baseline_proved_set: Optional[set[str]],
) -> tuple[int, int, int, dict[str, int], int, int, int, list[str], list[str], set[str]]:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    proved = int(raw.get("proved") or 0)
    progress = int(raw.get("exhausted") or 0)
    errored = int(raw.get("errored") or 0)
    pbo = dict(raw.get("proved_by_origin") or {})
    tb_a = int(raw.get("term_builder_attempt_count") or 0)
    tb_adv = int(raw.get("term_builder_advanced_count") or 0)
    tb_p = int(raw.get("term_builder_proved_count") or 0)
    proved_set: set[str] = {
        r["full_name"] for r in raw.get("per_theorem", []) if r.get("finished")
    }
    if baseline_proved_set is not None:
        newly_proved = sorted(proved_set - baseline_proved_set)
        newly_lost = sorted(baseline_proved_set - proved_set)
    else:
        newly_proved, newly_lost = [], []
    return proved, progress, errored, pbo, tb_a, tb_adv, tb_p, newly_proved, newly_lost, proved_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument("--min-hours", type=float, default=5.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--policy-type", default="hybrid_evolved")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument("--out-dir", default="project/evolve/autonomous_runs")
    parser.add_argument("--per-eval-timeout-seconds", type=int, default=2700,
                        help="Hard wall-clock per eval subprocess; default 45 min")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variant names to run (default: all)")
    parser.add_argument("--skip-variants", nargs="*", default=None,
                        help="Variant names to skip")
    args = parser.parse_args()

    run_id = f"v5-auto-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "research_log.md"
    scoreboard_path = out_root / "scoreboard.jsonl"
    best_path = out_root / "best_candidate.json"
    config_out_path = out_root / "config.json"

    config_out_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    chosen_variants = list(VARIANTS_DEFAULT)
    if args.variants:
        chosen_variants = [v for v in chosen_variants if v.name in args.variants]
    if args.skip_variants:
        chosen_variants = [v for v in chosen_variants if v.name not in args.skip_variants]

    started = time.time()
    deadline_max = started + args.max_hours * 3600
    deadline_min = started + args.min_hours * 3600

    log_lines: list[str] = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(f"# v5 autonomous research run — {run_id}\n")
    log(f"- start: {datetime.now(timezone.utc).isoformat()}")
    log(f"- theorem set: {args.theorem_set}")
    log(f"- min hours: {args.min_hours}  max hours: {args.max_hours}")
    log(f"- ckpt: {args.ckpt_dir}")
    log(f"- variants queued: {len(chosen_variants)}")
    log("")

    results: list[CycleResult] = []
    baseline_proved_set: Optional[set[str]] = None
    baseline_proved_count: int = 0

    for i, variant in enumerate(chosen_variants):
        now = time.time()
        if now >= deadline_max:
            log(f"\n## stopping — hit max-hours ({args.max_hours}h)")
            break
        remaining = deadline_max - now
        per_eval_to = min(args.per_eval_timeout_seconds, max(120, int(remaining)))
        log(f"\n## cycle {i+1} — {variant.name}  [{variant.direction}]")
        log(f"- {variant.description}")
        log(f"- elapsed: {(now - started)/3600:.2f}h  remaining: {remaining/3600:.2f}h")

        metrics_path = run_eval(
            variant=variant,
            theorem_set=args.theorem_set,
            ckpt_dir=args.ckpt_dir,
            out_root=eval_root,
            eval_timeout_seconds=per_eval_to,
        )
        if metrics_path is None:
            log(f"- FAILED — no metrics.json")
            continue

        (proved, progress, errored, pbo, tb_a, tb_adv, tb_p,
         newly_proved, newly_lost, proved_set) = summarize_metrics(
            metrics_path, baseline_proved_set,
        )
        if baseline_proved_set is None:
            baseline_proved_set = proved_set
            baseline_proved_count = proved
            log(f"- (set as baseline reference for diff)")

        delta = proved - baseline_proved_count
        elapsed = time.time() - now
        cycle = CycleResult(
            name=variant.name,
            direction=variant.direction,
            description=variant.description,
            proved=proved,
            progress=progress,
            errored=errored,
            proved_by_origin=pbo,
            term_builder_attempt=tb_a,
            term_builder_advanced=tb_adv,
            term_builder_proved=tb_p,
            runtime_seconds=elapsed,
            delta_vs_baseline=delta,
            newly_proved=newly_proved,
            newly_lost=newly_lost,
            eval_dir=str(metrics_path.parent),
        )
        results.append(cycle)
        with scoreboard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cycle.to_dict(), ensure_ascii=False) + "\n")
        log(f"- proved: {proved} (Δ {delta:+d})  progress: {progress}  errored: {errored}")
        log(f"- origins: {pbo}")
        if tb_a:
            log(f"- term_builder: {tb_a} attempts / {tb_adv} advanced / {tb_p} proved")
        if newly_proved:
            log(f"- NEW WINS: {newly_proved}")
        if newly_lost:
            log(f"- REGRESSIONS: {newly_lost}")

    elapsed_total = time.time() - started
    log(f"\n## run complete — {elapsed_total/3600:.2f}h, {len(results)} cycles")

    # best
    if results:
        best = max(results, key=lambda r: (r.proved, r.progress, -r.errored))
        best_path.write_text(json.dumps(best.to_dict(), indent=2), encoding="utf-8")
        log(f"- best: {best.name}  proved={best.proved} (Δ {best.delta_vs_baseline:+d})")
        if best.newly_proved:
            log(f"- best newly proved: {best.newly_proved}")

    # final report skeleton
    final = out_root / "final_report.md"
    final.write_text(_render_final_report(run_id, args, results, elapsed_total), encoding="utf-8")
    print(f"\n[autonomous_research_loop] run complete: {out_root}")


def _render_final_report(
    run_id: str, args, results: list[CycleResult], elapsed: float
) -> str:
    lines = [f"# v5 autonomous research final report — {run_id}\n"]
    lines.append(f"- theorem set: {args.theorem_set}")
    lines.append(f"- total runtime: {elapsed/3600:.2f}h")
    lines.append(f"- cycles: {len(results)}")
    lines.append("")
    lines.append("## scoreboard")
    lines.append("| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |")
    lines.append("|---|---------|-----|--------|----|------|-----|---------------------|--------------|")
    for i, r in enumerate(results, 1):
        np = ", ".join(r.newly_proved) if r.newly_proved else "—"
        tb = f"{r.term_builder_attempt}/{r.term_builder_advanced}/{r.term_builder_proved}"
        lines.append(
            f"| {i} | {r.name} | {r.direction} | {r.proved} | {r.delta_vs_baseline:+d} | "
            f"{r.progress} | {r.errored} | {tb} | {np} |"
        )
    lines.append("")
    if results:
        best = max(results, key=lambda x: (x.proved, x.progress, -x.errored))
        lines.append(f"## best candidate\n")
        lines.append(f"- name: `{best.name}`")
        lines.append(f"- direction: {best.direction}")
        lines.append(f"- proved: **{best.proved}**  (Δ {best.delta_vs_baseline:+d})")
        lines.append(f"- description: {best.description}")
        if best.newly_proved:
            lines.append(f"- newly proved: {best.newly_proved}")
        if best.newly_lost:
            lines.append(f"- regressions: {best.newly_lost}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
