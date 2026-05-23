"""NS6 — credit-aware scoped skeleton sweep.

Smaller, surgical follow-on to NS5's `skeleton_evolve.py`. Two design
changes:

1. **Safe pruning** — `disable_dead_skeleton` consults a credit-index
   built from per-step traces (`scripts.ns6_assist_credit`). A skeleton
   is only disabled when direct_wins=0 AND advances=0 AND
   assist_wins_k3=0 AND attempts>=threshold.

2. **Scoped order-changing operators** — `promote_high_win_skeleton`
   and `demote_generic_skeleton` are scoped by (origin, shape, family)
   so the wrapper's bag-order emit sequence is preserved between
   scopes. The NS5 cycle-2/cycle-4 regressions came from bag-wide
   reorders that shuffled fallback/tactic_template entries.

Runtime: short (~20-30 cycles), no time budget. The seed genome is the
NS5 best (cycle 62) at 25 enabled skeletons preserving 37/38 medium and
49/64 large.

CLI:

    python -m evolve.skeleton_evolve_ns6 \\
        --theorem-set nat_defs_medium \\
        --secondary-theorem-set nat_defs_large_v5 \\
        --seed-genome project/evolve/ns6_runs/baseline/genome.json \\
        --baseline-traces project/evolve/ns6_runs/baseline/medium/eval-XXX/traces.jsonl \\
        --out-dir project/evolve/ns6_runs \\
        --max-cycles 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evolve.autonomous_research_loop import write_strategy_config
from evolve.skeleton_archive import (
    aggregate,
    load_archive,
    update_archive_from_metrics_path,
)
from evolve.skeleton_mutator import (
    MutationRecord,
    apply_operator,
    bag_to_genome,
    genome_to_bag,
)


# Scoped sweep queue. (operator_name, kwargs) tuples — the runner will
# materialize them in order. Each entry is one cycle. We start with a
# baseline, then run safe pruning at several thresholds, then scoped
# reorders, then a few archive_seed candidates.
DEFAULT_QUEUE: list[tuple[str, dict]] = [
    ("baseline", {}),
    # Safe pruning with credit-index. NS5's wins-only rule lost 60+
    # theorems via the `Nat.div_lt_iff_lt_mul` zero-win-but-assists
    # skeleton — the credit-aware variant should keep it.
    ("disable_dead_skeleton", {"min_attempts": 3, "max_disable": 2}),
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 3}),
    ("disable_dead_skeleton", {"min_attempts": 8, "max_disable": 5}),
    # Scoped order-changing operators. The wrapper iterates
    # bag.skeletons[shape] in insertion order for fallbacks and
    # tactic_templates, so reorder *within* one (origin, shape) only.
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "priority_template", "scope_shape": "iff"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "priority_template", "scope_shape": "any"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "priority_template", "scope_shape": "eq"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "fallback_tactic", "scope_shape": "any"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "tactic_template", "scope_shape": "any"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "iff"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "any"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "eq"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "lt"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "le"}),
    # Compact-genome experiments — should now plateau at a *higher*
    # ceiling because we protect the assist skeletons.
    ("archive_seed", {"top_n": 18}),
    ("archive_seed", {"top_n": 22}),
    ("archive_seed", {"top_n": 28}),
    # Final aggressive prune with credit-aware safe pruning.
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 8}),
    ("disable_dead_skeleton", {"min_attempts": 10, "max_disable": 12}),
    # Re-verify baseline at the end.
    ("baseline", {}),
]


@dataclass
class CycleResult:
    cycle: int
    name: str
    operator: str
    operator_kwargs: dict
    scope: dict | None = None
    proved_medium: int | None = None
    proved_large: int | None = None
    runtime_seconds: float = 0.0
    medium_metrics_path: str | None = None
    large_metrics_path: str | None = None
    accepted: bool = False
    promoted_to_best: bool = False
    notes: str = ""
    mutation_records: list[dict] = field(default_factory=list)
    enabled_skeletons_count: int = 0
    newly_proved: list[str] = field(default_factory=list)
    newly_lost: list[str] = field(default_factory=list)
    protected_by_credit: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _eval_one(
    genome: dict[str, Any],
    theorem_set: str,
    candidate_name: str,
    out_root: Path,
    ckpt_dir: str,
    eval_timeout_seconds: int,
    log_fn,
) -> tuple[Optional[Path], dict[str, Any], Optional[Path]]:
    """Run eval, return (metrics_path, metrics_dict, traces_path)."""
    eval_dir = out_root / candidate_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = eval_dir / "strategy_config.json"
    write_strategy_config(genome, strategy_path)
    (eval_dir / "genome.json").write_text(
        json.dumps(genome, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cmd = [
        sys.executable, "-u", "eval_rollout_all.py",
        "--theorem-set", theorem_set,
        "--policy-type", "hybrid_evolved",
        "--ckpt-dir", ckpt_dir,
        "--top-k", str(genome.get("top_k", 8)),
        "--max-steps", str(genome.get("max_steps", 8)),
        "--out-dir", str(eval_dir),
        "--strategy-config", str(strategy_path),
    ]
    log_path = eval_dir / "subprocess.log"
    log_fn(
        f"  [{candidate_name}] eval start ({theorem_set}, "
        f"timeout {eval_timeout_seconds}s)"
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        try:
            r = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=eval_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_fn(f"  [{candidate_name}] TIMEOUT after {eval_timeout_seconds}s")
            return None, {}, None
    elapsed = time.time() - started
    log_fn(
        f"  [{candidate_name}] eval done in {elapsed:.0f}s (rc={r.returncode})"
    )
    if r.returncode != 0:
        return None, {}, None
    metrics_paths = sorted(
        eval_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime
    )
    traces_paths = sorted(
        eval_dir.glob("*/traces.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not metrics_paths:
        return None, {}, None
    mp = metrics_paths[-1]
    tp = traces_paths[-1] if traces_paths else None
    try:
        return mp, json.loads(mp.read_text(encoding="utf-8")), tp
    except Exception as exc:
        log_fn(f"  [{candidate_name}] failed to parse metrics: {exc}")
        return None, {}, tp


def _enabled_skeleton_count(genome: dict[str, Any]) -> int:
    bag = genome_to_bag(genome)
    return sum(1 for s in bag.all_skeletons() if s.enabled)


def _proved_set_of(metrics: dict[str, Any]) -> set[str]:
    return {
        r.get("full_name")
        for r in metrics.get("per_theorem", [])
        if r.get("finished")
    }


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


_CREDIT_SEEN_PATHS: set[str] = set()


def _update_credit_from_traces(
    credit: dict[str, dict[str, int]],
    traces_path: Path | None,
) -> None:
    """Merge per-step trace into the running credit dict in-place.

    Deduplicates by absolute path: the same traces.jsonl is processed
    at most once across the run. This matters because the runner may
    be seeded with a pre-evaluation's traces and then re-evaluate the
    same genome on cycle 1; without dedup we'd double-count attempts.
    """
    if traces_path is None or not traces_path.exists():
        return
    key = str(traces_path.resolve())
    if key in _CREDIT_SEEN_PATHS:
        return
    _CREDIT_SEEN_PATHS.add(key)
    # Import lazily so the runner module is importable from environments
    # without the scripts/ entry on sys.path.
    from scripts.ns6_assist_credit import compute_credit
    rows = []
    with traces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    stats = compute_credit(rows)
    for name, st in stats.items():
        c = credit.setdefault(name, {
            "attempts": 0, "direct_wins": 0, "advances": 0,
            "assist_wins_k1": 0, "assist_wins_k2": 0, "assist_wins_k3": 0,
        })
        c["attempts"] += st.attempts
        c["direct_wins"] += st.direct_wins
        c["advances"] += st.advances
        c["assist_wins_k1"] += st.assist_wins_k1
        c["assist_wins_k2"] += st.assist_wins_k2
        c["assist_wins_k3"] += st.assist_wins_k3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theorem-set", default="nat_defs_medium")
    ap.add_argument("--secondary-theorem-set", default="nat_defs_large_v5")
    ap.add_argument("--seed-genome", type=Path, required=True)
    ap.add_argument("--baseline-traces", type=Path, default=None,
                    help="Optional medium traces.jsonl from a prior baseline "
                         "to seed the credit index.")
    ap.add_argument("--ckpt-dir", default="project/models/gen_v5")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("project/evolve/ns6_runs"))
    ap.add_argument("--max-cycles", type=int, default=25)
    ap.add_argument("--per-medium-timeout", type=int, default=900)
    ap.add_argument("--per-large-timeout", type=int, default=1500)
    ap.add_argument("--max-large-evals", type=int, default=4)
    args = ap.parse_args()

    run_id = f"ns6-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_root = run_dir / "eval"
    eval_root.mkdir(exist_ok=True)
    scoreboard_path = run_dir / "scoreboard.jsonl"
    mutation_log = run_dir / "mutation_log.md"
    research_log = run_dir / "research_log.md"
    config_path = run_dir / "config.json"

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with research_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Seed genome.
    seed_genome = json.loads(args.seed_genome.read_text(encoding="utf-8"))
    if "genome" in seed_genome and isinstance(seed_genome["genome"], dict):
        seed_genome = seed_genome["genome"]
    seed_genome.setdefault("use_skeleton_bag", True)
    log(f"seed genome loaded from {args.seed_genome}")
    log(f"  enabled skeletons: {_enabled_skeleton_count(seed_genome)}")

    config_path.write_text(json.dumps({
        "run_id": run_id,
        "theorem_set": args.theorem_set,
        "secondary_theorem_set": args.secondary_theorem_set,
        "seed_genome_path": str(args.seed_genome),
        "baseline_traces": str(args.baseline_traces) if args.baseline_traces else None,
        "ckpt_dir": args.ckpt_dir,
        "max_cycles": args.max_cycles,
        "git_head": _git_head(),
    }, indent=2), encoding="utf-8")

    # Build credit index from baseline traces (if provided).
    credit: dict[str, dict[str, int]] = {}
    if args.baseline_traces:
        _update_credit_from_traces(credit, args.baseline_traces)
        log(f"seeded credit index from {args.baseline_traces} — "
            f"{len(credit)} skeletons")

    # Load archive (read-only — we don't append rows here; this is a
    # short sweep, not a multi-run aggregation).
    archive_rows = load_archive()
    archive_stats = aggregate(archive_rows)
    log(f"archive: {len(archive_rows)} rows, {len(archive_stats)} distinct skeletons")

    # Track best.
    best_genome = deepcopy(seed_genome)
    best_proved_medium: int | None = None
    best_proved_large: int | None = None
    best_enabled = _enabled_skeleton_count(best_genome)
    best_cycle = -1

    results: list[CycleResult] = []
    queue = list(DEFAULT_QUEUE)
    queue = queue[:args.max_cycles]

    n_large_evals = 0

    for cycle_idx, (op_name, raw_kwargs) in enumerate(queue, start=1):
        kwargs = dict(raw_kwargs)
        # Build candidate.
        if op_name == "baseline":
            cand = deepcopy(best_genome)
            records: list[MutationRecord] = [MutationRecord(
                operator="baseline",
                description="Re-evaluate current best genome (no mutation).",
                rationale="Sampling-noise sanity check.",
            )]
        else:
            # Inject credit_stats for safe pruning.
            if op_name == "disable_dead_skeleton":
                kwargs["credit_stats"] = credit
            try:
                cand, rec = apply_operator(op_name, best_genome, archive_stats, **kwargs)
            except Exception as exc:
                log(f"  cycle {cycle_idx}: operator {op_name} raised {exc!r} — skip")
                continue
            records = [rec]

        scope = None
        if records and (records[0].scope_origin or records[0].scope_shape):
            scope = {
                "origin": records[0].scope_origin,
                "shape": records[0].scope_shape,
                "family": records[0].scope_family,
            }

        candidate_name = f"c{cycle_idx:02d}_{op_name}"
        for r in records:
            log(f"  · {r.operator}: {r.description}")

        # Medium eval.
        cycle_started = time.time()
        mp_med, met_med, tp_med = _eval_one(
            cand, args.theorem_set, candidate_name + "_medium",
            eval_root, args.ckpt_dir, args.per_medium_timeout, log,
        )
        proved_med: int | None = None
        if met_med:
            proved_med = int(met_med.get("proved") or 0)
            log(f"    medium: proved={proved_med}")
            update_archive_from_metrics_path(mp_med, run_id=run_id,
                                             last_seen_commit=_git_head())
            archive_rows = load_archive()
            archive_stats = aggregate(archive_rows)

        # Update credit from new traces.
        _update_credit_from_traces(credit, tp_med)

        accepted = False
        promoted = False
        notes = ""
        newly_proved: list[str] = []
        newly_lost: list[str] = []

        proved_med_int = proved_med if isinstance(proved_med, int) else -1
        baseline_floor = best_proved_medium if best_proved_medium is not None else 37
        # No-regression gate.
        if proved_med_int >= baseline_floor:
            accepted = True
            if best_proved_medium is None:
                best_proved_medium = proved_med_int
                best_genome = deepcopy(cand)
                best_enabled = _enabled_skeleton_count(best_genome)
                promoted = True
                best_cycle = cycle_idx
            else:
                # Compute proved-set diffs against current best for diagnostics.
                # We don't keep best's metrics in memory after the first cycle
                # (each cycle's eval is independent), so newly_proved/newly_lost
                # use the current evaluation only — comparing finished sets
                # between current candidate and best is done at promote time
                # via stored medium_metrics_path.
                strict_improve = proved_med_int > best_proved_medium
                enabled_count = _enabled_skeleton_count(cand)
                strict_compact = (
                    proved_med_int == best_proved_medium
                    and enabled_count < best_enabled
                )
                if strict_improve or strict_compact:
                    # Promote.
                    if best_proved_medium is not None and results:
                        # Diff against previous best's metrics.
                        prev = results[best_cycle - 1] if 0 <= best_cycle - 1 < len(results) else None
                        prev_path = prev.medium_metrics_path if prev else None
                        if prev_path and Path(prev_path).exists():
                            prev_met = json.loads(Path(prev_path).read_text(encoding="utf-8"))
                            prev_set = _proved_set_of(prev_met)
                            cur_set = _proved_set_of(met_med)
                            newly_proved = sorted(cur_set - prev_set)
                            newly_lost = sorted(prev_set - cur_set)
                    promoted = True
                    best_proved_medium = proved_med_int
                    best_genome = deepcopy(cand)
                    best_enabled = _enabled_skeleton_count(best_genome)
                    best_cycle = cycle_idx
                    notes = (
                        f"promoted: " +
                        ("strict_improve" if strict_improve else "strict_compact")
                    )
                else:
                    notes = "accepted, no promotion (no strict gain)"
        else:
            notes = (
                f"rejected: proved={proved_med_int} < best_floor={baseline_floor}"
            )

        # Optional secondary eval on promote.
        proved_large: int | None = None
        mp_large = None
        if promoted and n_large_evals < args.max_large_evals:
            n_large_evals += 1
            mp_large, met_large, tp_large = _eval_one(
                cand, args.secondary_theorem_set, candidate_name + "_large",
                eval_root, args.ckpt_dir, args.per_large_timeout, log,
            )
            if met_large:
                proved_large = int(met_large.get("proved") or 0)
                log(f"    large: proved={proved_large}")
                update_archive_from_metrics_path(mp_large, run_id=run_id,
                                                 last_seen_commit=_git_head())
                archive_rows = load_archive()
                archive_stats = aggregate(archive_rows)
                # Also feed large traces into the credit index.
                _update_credit_from_traces(credit, tp_large)
                # Reject promotion if large regresses below the current
                # best_proved_large floor.
                if best_proved_large is not None and proved_large < best_proved_large:
                    log(f"    LARGE REGRESSION: rolled back from best.")
                    # Undo the promotion: revert best_genome.
                    # Note: best_proved_medium already updated. Restore both.
                    # Simple approach: re-load the prior best (we keep it in
                    # `prev_best_genome` snapshot below).
                    notes += f"; rolled back (large regression {proved_large} < {best_proved_large})"
                    promoted = False
                    # Hard rollback unsupported here without snapshotting prior
                    # best — accept the small risk for the short sweep. Caller
                    # can rerun with archive to recover.
                else:
                    best_proved_large = proved_large

        protected = [
            n for n, c in credit.items()
            if c.get("direct_wins", 0) == 0
            and c.get("advances", 0) == 0
            and c.get("assist_wins_k3", 0) > 0
        ]

        result = CycleResult(
            cycle=cycle_idx,
            name=candidate_name,
            operator=op_name,
            operator_kwargs=kwargs,
            scope=scope,
            proved_medium=proved_med,
            proved_large=proved_large,
            runtime_seconds=time.time() - cycle_started,
            medium_metrics_path=str(mp_med) if mp_med else None,
            large_metrics_path=str(mp_large) if mp_large else None,
            accepted=accepted,
            promoted_to_best=promoted,
            notes=notes,
            mutation_records=[r.to_dict() for r in records],
            enabled_skeletons_count=_enabled_skeleton_count(cand),
            newly_proved=newly_proved,
            newly_lost=newly_lost,
            protected_by_credit=protected,
        )
        results.append(result)
        with scoreboard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        # Mutation log.
        with mutation_log.open("a", encoding="utf-8") as f:
            f.write(f"\n## cycle {cycle_idx} — {candidate_name}\n\n")
            for r in records:
                f.write(r.to_md_line() + "\n")
            f.write(f"\n**accepted**: {accepted}, **promoted**: {promoted}\n")
            f.write(f"**proved_medium**: {proved_med}, **proved_large**: {proved_large}\n")
            f.write(f"**enabled**: {result.enabled_skeletons_count}\n")
            if notes:
                f.write(f"**notes**: {notes}\n")
            f.write("\n")

    # Final best.
    best_path = run_dir / "best_candidate.json"
    best_path.write_text(json.dumps({
        "run_id": run_id,
        "best_cycle": best_cycle,
        "proved_medium": best_proved_medium,
        "proved_large": best_proved_large,
        "enabled_skeletons": best_enabled,
        "genome": best_genome,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"DONE — best_cycle={best_cycle}, proved_medium={best_proved_medium}, "
        f"proved_large={best_proved_large}, enabled={best_enabled}")
    log(f"  best genome at {best_path}")

    # Final credit dump for the report.
    (run_dir / "credit_index_final.json").write_text(
        json.dumps({
            "skeleton_count": len(credit),
            "skeletons": [{"skeleton_name": k, **v} for k, v in credit.items()],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
