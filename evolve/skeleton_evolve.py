"""NS5 skeleton-evolution runner.

Time-bounded autonomous loop that mutates the *skeleton* representation
of the strategy genome, evaluates each candidate against
`nat_defs_medium` (and `nat_defs_large_v5` when the candidate clears
the no-regression gate), updates the skeleton archive, and writes a
final report.

CLI:

    python -m evolve.skeleton_evolve \\
        --theorem-set nat_defs_medium \\
        --secondary-theorem-set nat_defs_large_v5 \\
        --min-hours 6 \\
        --max-hours 8 \\
        --ckpt-dir project/models/gen_v5 \\
        --out-dir project/evolve/skeleton_runs

Cycle behaviour:

  1. Pick the next mutation operator from the queue (or fall back to a
     no-op cycle when the archive is sparse).
  2. Produce a new candidate dict from the current best genome.
  3. Write strategy_config.json, run eval_rollout_all.py on the medium
     set.
  4. Parse metrics, ingest into the archive.
  5. Decide accept / reject (no regression on medium).
  6. Promote-to-best when better than incumbent.
  7. On promotion, optionally evaluate the new best on the secondary
     set.

The runner stops when EITHER:
  - elapsed >= min-hours AND no pending high-priority cycle, OR
  - elapsed >= max-hours.

All artifacts go under `<out-dir>/<run_id>/`. Nothing is committed —
the caller (Claude / the human) commits source files separately.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from evolve.autonomous_research_loop import (
    REPO_ROOT,
    baseline_genome,
    summarize_metrics,
    write_strategy_config,
)
from evolve.autonomous_research_ns3 import _ns3_combined, V5_27_GENOME
from evolve.skeleton_archive import (
    DEFAULT_ARCHIVE_PATH,
    DEFAULT_INDEX_PATH,
    aggregate,
    dead_skeletons,
    load_archive,
    summarize as archive_summarize,
    top_skeletons_by_wins,
    update_archive_from_metrics_path,
    write_index,
)
from evolve.skeleton_mutator import (
    OPERATORS,
    MutationRecord,
    append_mutation_log,
    apply_operator,
    bag_to_genome,
    genome_to_bag,
)


# ---------------------------------------------------------------------- defaults

# Operator queue. Repeats are intentional — we want multiple variants per
# operator to amortize sampling noise and let the archive accumulate.
# The runner will keep iterating until min-hours, then drain whatever's
# left in the queue (capped by max-hours).
DEFAULT_OPERATOR_QUEUE: list[tuple[str, dict]] = [
    # Pass 1 — seed the archive.
    ("baseline", {}),
    ("demote_generic_skeleton", {}),
    ("disable_dead_skeleton", {"min_attempts": 10, "max_disable": 1}),
    ("promote_high_win_skeleton", {"top_n": 5}),
    ("clone_skeleton_to_shape", {"top_n": 5}),
    ("budget_trim", {}),
    # Pass 2 — dead-skel pruning at varying thresholds.
    ("disable_dead_skeleton", {"min_attempts": 8, "max_disable": 1}),
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 1}),
    ("disable_dead_skeleton", {"min_attempts": 12, "max_disable": 2}),
    ("disable_dead_skeleton", {"min_attempts": 15, "max_disable": 3}),
    ("disable_dead_skeleton", {"min_attempts": 20, "max_disable": 5}),
    # Pass 3 — archive-guided promotion / cloning.
    ("promote_high_win_skeleton", {"top_n": 10}),
    ("clone_skeleton_to_shape", {"top_n": 10}),
    ("demote_generic_skeleton", {}),
    ("promote_high_win_skeleton", {"top_n": 20}),
    ("clone_skeleton_to_shape", {"top_n": 20}),
    # Pass 4 — budget tuning.
    ("budget_trim", {}),
    ("budget_trim", {}),
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 3}),
    ("promote_high_win_skeleton", {"top_n": 30}),
    # Pass 5 — compact-genome experiment.
    ("archive_seed", {"top_n": 15}),
    ("archive_seed", {"top_n": 20}),
    ("archive_seed", {"top_n": 25}),
    ("archive_seed", {"top_n": 30}),
    ("archive_seed", {"top_n": 40}),
    # Pass 6 — final aggressive trim once archive is well-populated.
    ("disable_dead_skeleton", {"min_attempts": 3, "max_disable": 2}),
    ("disable_dead_skeleton", {"min_attempts": 4, "max_disable": 3}),
    ("disable_dead_skeleton", {"min_attempts": 6, "max_disable": 4}),
    ("disable_dead_skeleton", {"min_attempts": 10, "max_disable": 5}),
    ("promote_high_win_skeleton", {"top_n": 15}),
    # Pass 7 — exploratory cloning at high top-n.
    ("clone_skeleton_to_shape", {"top_n": 30}),
    ("clone_skeleton_to_shape", {"top_n": 40}),
    ("demote_generic_skeleton", {}),
    ("budget_trim", {}),
    ("archive_seed", {"top_n": 50}),
    # Pass 8 — re-run favourites with refreshed archive.
    ("disable_dead_skeleton", {"min_attempts": 8, "max_disable": 6}),
    ("disable_dead_skeleton", {"min_attempts": 10, "max_disable": 8}),
    ("promote_high_win_skeleton", {"top_n": 25}),
    ("archive_seed", {"top_n": 35}),
    ("clone_skeleton_to_shape", {"top_n": 15}),
    # Pass 9 — repeats for sampling stability.
    ("baseline", {}),
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 4}),
    ("promote_high_win_skeleton", {"top_n": 12}),
    ("clone_skeleton_to_shape", {"top_n": 8}),
    ("demote_generic_skeleton", {}),
    ("budget_trim", {}),
    ("archive_seed", {"top_n": 22}),
    ("disable_dead_skeleton", {"min_attempts": 15, "max_disable": 6}),
    ("promote_high_win_skeleton", {"top_n": 18}),
    ("archive_seed", {"top_n": 28}),
    # Pass 10 — final wrap.
    ("disable_dead_skeleton", {"min_attempts": 8, "max_disable": 10}),
    ("promote_high_win_skeleton", {"top_n": 40}),
    ("clone_skeleton_to_shape", {"top_n": 50}),
    ("archive_seed", {"top_n": 18}),
    ("baseline", {}),
]


# ---------------------------------------------------------------------- bookkeeping
@dataclass
class CycleResult:
    cycle: int
    name: str
    operator: str
    operator_kwargs: dict
    proved_medium: int | None = None
    proved_large: int | None = None
    runtime_seconds: float = 0.0
    medium_metrics_path: str | None = None
    large_metrics_path: str | None = None
    accepted: bool = False
    promoted_to_best: bool = False
    notes: str = ""
    mutation_records: list[dict] = field(default_factory=list)
    # Surface skeleton attribution counts for easy reporting.
    skeleton_attempt_count: int = 0
    skeleton_advanced_count: int = 0
    skeleton_proved_count: int = 0
    enabled_skeletons_count: int = 0
    newly_proved: list[str] = field(default_factory=list)
    newly_lost: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- runner core
def _seed_genome() -> dict[str, Any]:
    """The ns3-combined genome with `use_skeleton_bag=True`. This is the
    current 37/38 baseline.
    """
    base = baseline_genome()
    base.update(json.loads(V5_27_GENOME.read_text()))
    g = _ns3_combined(deepcopy(base))
    g["use_skeleton_bag"] = True
    return g


def _eval_one(
    genome: dict[str, Any],
    theorem_set: str,
    candidate_name: str,
    out_root: Path,
    ckpt_dir: str,
    eval_timeout_seconds: int,
    log_fn,
) -> tuple[Optional[Path], dict[str, Any]]:
    """Write strategy config, run eval_rollout_all.py, return
    `(metrics_path, parsed_metrics)`. `metrics_path` is None on failure.
    """
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
            return None, {}
    elapsed = time.time() - started
    log_fn(
        f"  [{candidate_name}] eval done in {elapsed:.0f}s (rc={r.returncode})"
    )
    if r.returncode != 0:
        return None, {}
    metrics_paths = sorted(
        eval_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime
    )
    if not metrics_paths:
        return None, {}
    mp = metrics_paths[-1]
    try:
        return mp, json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:
        log_fn(f"  [{candidate_name}] failed to parse metrics: {exc}")
        return None, {}


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


def _enabled_skeleton_count(genome: dict[str, Any]) -> int:
    bag = genome_to_bag(genome)
    return sum(1 for s in bag.all_skeletons() if s.enabled)


def _proved_set_of(metrics: dict[str, Any]) -> set[str]:
    return {
        r.get("full_name")
        for r in metrics.get("per_theorem", [])
        if r.get("finished")
    }


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorem-set", default="nat_defs_medium")
    parser.add_argument(
        "--secondary-theorem-set", default="nat_defs_large_v5",
        help="Set to '' to disable secondary evaluation."
    )
    parser.add_argument("--min-hours", type=float, default=6.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5")
    parser.add_argument(
        "--out-dir", default="project/evolve/skeleton_runs",
        help="Root for this run's outputs."
    )
    parser.add_argument(
        "--archive-path",
        default=str(DEFAULT_ARCHIVE_PATH),
        help="Skeleton archive JSONL.",
    )
    parser.add_argument(
        "--archive-index",
        default=str(DEFAULT_INDEX_PATH),
        help="Skeleton archive index JSON.",
    )
    parser.add_argument(
        "--per-medium-timeout", type=int, default=1500,
        help="Hard wall-clock per medium eval subprocess (s)."
    )
    parser.add_argument(
        "--per-large-timeout", type=int, default=2700,
        help="Hard wall-clock per large eval subprocess (s)."
    )
    parser.add_argument(
        "--baseline-proved-medium", type=int, default=37,
        help="Acceptance gate: candidate must >= this on the medium set."
    )
    parser.add_argument(
        "--baseline-proved-large", type=int, default=49,
        help="Promotion-of-large gate: candidate's large result must >= this."
    )
    parser.add_argument(
        "--max-large-evals", type=int, default=4,
        help="Cap on the number of large evaluations across the run."
    )
    parser.add_argument(
        "--operator-queue", default=None,
        help="Optional JSON list of {operator, kwargs} dicts. If absent, "
        "uses the built-in NS5 queue.",
    )
    args = parser.parse_args()

    run_id = (
        f"ns5-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    out_root = REPO_ROOT / args.out_dir / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    config_path = out_root / "config.json"
    scoreboard_path = out_root / "scoreboard.jsonl"
    best_path = out_root / "best_candidate.json"
    final_report_path = out_root / "final_report.md"
    mutation_log_path = out_root / "mutation_log.md"
    archive_snapshot_path = out_root / "archive_snapshot.jsonl"

    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    log_lines: list[str] = []
    log_path = out_root / "research_log.md"

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    head = _git_head()
    started = time.time()
    deadline_min = started + args.min_hours * 3600
    deadline_max = started + args.max_hours * 3600

    log(f"# NS5 skeleton evolution — {run_id}\n")
    log(f"- start: {datetime.now(timezone.utc).isoformat()}")
    log(f"- branch HEAD: {head}")
    log(f"- theorem_set: {args.theorem_set}")
    log(f"- secondary: {args.secondary_theorem_set or '(disabled)'}")
    log(f"- ckpt: {args.ckpt_dir}")
    log(f"- min/max hours: {args.min_hours}/{args.max_hours}")
    log(f"- archive: {args.archive_path}")
    log("")

    # Load existing archive (may be empty).
    archive_rows = load_archive(args.archive_path)
    stats = aggregate(archive_rows)
    log(
        f"archive at start: {len(archive_rows)} rows, "
        f"{len(stats)} distinct skeletons"
    )

    base_queue: list[tuple[str, dict]]
    if args.operator_queue and Path(args.operator_queue).exists():
        with open(args.operator_queue, "r", encoding="utf-8") as f:
            base_queue = [(item["operator"], item.get("kwargs", {})) for item in json.load(f)]
    else:
        base_queue = list(DEFAULT_OPERATOR_QUEUE)
    queue = list(base_queue)

    # The current best genome and its metrics. Start at the seed
    # (ns3-combined with use_skeleton_bag=True).
    best_genome = _seed_genome()
    best_proved_medium: int | None = None
    best_proved_large: int | None = None
    best_proved_set: set[str] | None = None
    cycle_results: list[CycleResult] = []
    large_evals_used = 0

    log("queue: " + ", ".join(f"{op}{kw}" for op, kw in queue))
    log("")

    cycle_idx = 0
    queue_pass = 1
    while queue:
        op_name, op_kwargs = queue.pop(0)
        cycle_idx += 1
        now = time.time()
        elapsed_h = (now - started) / 3600
        remaining_h = (deadline_max - now) / 3600
        if now >= deadline_max:
            log(f"\n## stopping — hit max-hours ({args.max_hours}h)")
            break
        # When queue drains: stop if past min-hours, otherwise loop.
        if not queue:
            if now >= deadline_min:
                log(f"\n## queue drained past min-hours — finishing")
                # Process this final entry and let loop exit naturally.
            else:
                queue_pass += 1
                log(
                    f"\n  queue drained at cycle {cycle_idx} "
                    f"(elapsed {elapsed_h:.2f}h < min {args.min_hours}h) — "
                    f"starting pass {queue_pass}"
                )
                # Refill from a shuffled copy of base_queue (light variation
                # to keep each new pass slightly different).
                if queue_pass <= 4:
                    queue = list(base_queue)
                else:
                    # Diversify by reversing each subsequent pass.
                    queue = list(reversed(base_queue))

        candidate_name = f"c{cycle_idx:02d}_{op_name}"
        log(
            f"\n## cycle {cycle_idx} — {candidate_name}  "
            f"[{elapsed_h:.2f}h elapsed, {remaining_h:.2f}h left]"
        )

        records: list[MutationRecord] = []
        if op_name == "baseline":
            candidate_genome = deepcopy(best_genome)
            records.append(MutationRecord(
                operator="baseline",
                description="Reproduce current best (no mutation).",
                rationale="Sanity / archive seeding.",
            ))
        else:
            stats = aggregate(load_archive(args.archive_path))
            try:
                candidate_genome, rec = apply_operator(
                    op_name, deepcopy(best_genome), stats, **op_kwargs
                )
            except Exception as exc:
                log(f"  operator {op_name} raised: {exc}")
                continue
            records.append(rec)

        enabled_count = _enabled_skeleton_count(candidate_genome)
        log(
            f"  operator: {op_name}{op_kwargs}  "
            f"-> enabled_skeletons={enabled_count}"
        )
        for r in records:
            log(f"  · {r.operator}: {r.description}")

        # Decide per-eval timeout from remaining time.
        time_left = max(0.0, deadline_max - time.time())
        medium_to = min(args.per_medium_timeout, max(180, int(time_left)))
        medium_path, medium_metrics = _eval_one(
            candidate_genome,
            theorem_set=args.theorem_set,
            candidate_name=candidate_name + "_medium",
            out_root=eval_root,
            ckpt_dir=args.ckpt_dir,
            eval_timeout_seconds=medium_to,
            log_fn=log,
        )
        proved_medium = (
            int(medium_metrics.get("proved")) if medium_metrics else None
        )
        log(f"  medium: proved={proved_medium}")

        accepted = (
            proved_medium is not None
            and proved_medium >= args.baseline_proved_medium
        )

        # Archive ingestion.
        if medium_path is not None:
            try:
                res = update_archive_from_metrics_path(
                    medium_path,
                    run_id=run_id + "/" + candidate_name + "_medium",
                    last_seen_commit=head,
                    archive_path=args.archive_path,
                    index_path=args.archive_index,
                )
                log(f"  archive: +{res['rows_appended']} rows (total {res['rows_total']})")
            except Exception as exc:
                log(f"  archive ingest failed: {exc}")

        # Compute proved-set delta vs current best.
        newly_proved: list[str] = []
        newly_lost: list[str] = []
        if medium_metrics:
            ps = _proved_set_of(medium_metrics)
            if best_proved_set is not None:
                newly_proved = sorted(ps - best_proved_set)
                newly_lost = sorted(best_proved_set - ps)
        if newly_proved:
            log(f"  newly proved: {newly_proved}")
        if newly_lost:
            log(f"  REGRESSIONS: {newly_lost}")

        # Decide secondary (large) eval.
        proved_large: int | None = None
        large_path = None
        should_run_large = (
            accepted
            and args.secondary_theorem_set
            and large_evals_used < args.max_large_evals
            and (time.time() < deadline_max - args.per_large_timeout - 60)
        )
        # Always run large on the very first accepted cycle to establish
        # baseline for secondary; thereafter only on promotions.
        promoted = False
        if best_proved_medium is None and accepted:
            # First accepted cycle becomes incumbent.
            promoted = True
        elif accepted:
            if proved_medium > (best_proved_medium or -1):
                promoted = True
            elif (
                proved_medium == (best_proved_medium or -1)
                and enabled_count < _enabled_skeleton_count(best_genome)
            ):
                promoted = True

        if should_run_large and promoted:
            time_left = max(0.0, deadline_max - time.time())
            large_to = min(args.per_large_timeout, max(180, int(time_left)))
            large_path, large_metrics = _eval_one(
                candidate_genome,
                theorem_set=args.secondary_theorem_set,
                candidate_name=candidate_name + "_large",
                out_root=eval_root,
                ckpt_dir=args.ckpt_dir,
                eval_timeout_seconds=large_to,
                log_fn=log,
            )
            large_evals_used += 1
            proved_large = (
                int(large_metrics.get("proved")) if large_metrics else None
            )
            log(f"  large: proved={proved_large}")
            if large_path is not None:
                try:
                    res = update_archive_from_metrics_path(
                        large_path,
                        run_id=run_id + "/" + candidate_name + "_large",
                        last_seen_commit=head,
                        archive_path=args.archive_path,
                        index_path=args.archive_index,
                    )
                    log(
                        f"  archive: +{res['rows_appended']} rows "
                        f"(total {res['rows_total']})"
                    )
                except Exception as exc:
                    log(f"  archive ingest (large) failed: {exc}")

            # Tighten promotion based on large result.
            if best_proved_large is not None:
                if (
                    proved_large is not None
                    and proved_large < best_proved_large
                    and proved_medium <= (best_proved_medium or 0)
                ):
                    promoted = False
                    log(
                        "  promotion REVOKED: regressed on large without "
                        "compensating medium gain."
                    )

        if promoted:
            best_genome = candidate_genome
            best_proved_medium = proved_medium
            if proved_large is not None:
                best_proved_large = proved_large
            best_proved_set = _proved_set_of(medium_metrics) if medium_metrics else None
            best_path.write_text(
                json.dumps(
                    {
                        "cycle": cycle_idx,
                        "candidate_name": candidate_name,
                        "proved_medium": best_proved_medium,
                        "proved_large": best_proved_large,
                        "enabled_skeletons": enabled_count,
                        "genome": best_genome,
                    },
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            log(
                f"  PROMOTED → best now proved_medium={best_proved_medium}, "
                f"proved_large={best_proved_large}"
            )

        # CycleResult.
        skat = int(medium_metrics.get("skeleton_attempt_count") or 0) if medium_metrics else 0
        skad = int(medium_metrics.get("skeleton_advanced_count") or 0) if medium_metrics else 0
        skpr = int(medium_metrics.get("skeleton_proved_count") or 0) if medium_metrics else 0
        cycle_res = CycleResult(
            cycle=cycle_idx,
            name=candidate_name,
            operator=op_name,
            operator_kwargs=op_kwargs,
            proved_medium=proved_medium,
            proved_large=proved_large,
            runtime_seconds=time.time() - now,
            medium_metrics_path=str(medium_path) if medium_path else None,
            large_metrics_path=str(large_path) if large_path else None,
            accepted=accepted,
            promoted_to_best=promoted,
            notes="; ".join(r.description for r in records),
            mutation_records=[r.to_dict() for r in records],
            skeleton_attempt_count=skat,
            skeleton_advanced_count=skad,
            skeleton_proved_count=skpr,
            enabled_skeletons_count=enabled_count,
            newly_proved=newly_proved,
            newly_lost=newly_lost,
        )
        cycle_results.append(cycle_res)
        with scoreboard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cycle_res.to_dict(), ensure_ascii=False) + "\n")
        append_mutation_log(
            mutation_log_path,
            cycle=cycle_idx,
            candidate_name=candidate_name,
            records=records,
            eval_summary={
                "proved_medium": proved_medium,
                "proved_large": proved_large,
                "accepted": accepted,
                "promoted": promoted,
                "enabled_skeletons": enabled_count,
                "skeleton_attempt_count": skat,
                "skeleton_proved_count": skpr,
                "newly_proved": newly_proved,
                "newly_lost": newly_lost,
            },
        )

    # Snapshot the archive.
    try:
        rows = load_archive(args.archive_path)
        archive_snapshot_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # Final report.
    elapsed_total = time.time() - started
    final_report_path.write_text(
        _render_final_report(
            run_id=run_id,
            started=started,
            elapsed=elapsed_total,
            head=head,
            args=args,
            cycle_results=cycle_results,
            best_proved_medium=best_proved_medium,
            best_proved_large=best_proved_large,
            archive_path=args.archive_path,
        ),
        encoding="utf-8",
    )
    log(
        f"\n## run complete — {elapsed_total/3600:.2f}h, "
        f"{len(cycle_results)} cycles, "
        f"best_medium={best_proved_medium}, best_large={best_proved_large}"
    )
    log(f"final report: {final_report_path}")


def _render_final_report(
    run_id: str,
    started: float,
    elapsed: float,
    head: str,
    args: argparse.Namespace,
    cycle_results: list[CycleResult],
    best_proved_medium: int | None,
    best_proved_large: int | None,
    archive_path: str,
) -> str:
    arc_summary = archive_summarize(archive_path, top_n=15, dead_min_attempts=10)
    promoted = [r for r in cycle_results if r.promoted_to_best]
    accepted = [r for r in cycle_results if r.accepted]
    regressed = [r for r in cycle_results if r.newly_lost]
    table = ["| cycle | operator | enabled | medium | large | accepted | promoted |",
             "|------:|----------|--------:|-------:|------:|:--------:|:--------:|"]
    for r in cycle_results:
        table.append(
            f"| {r.cycle} | {r.operator}{r.operator_kwargs or ''} | "
            f"{r.enabled_skeletons_count} | "
            f"{r.proved_medium if r.proved_medium is not None else '—'} | "
            f"{r.proved_large if r.proved_large is not None else '—'} | "
            f"{'yes' if r.accepted else 'no'} | "
            f"{'yes' if r.promoted_to_best else ''} |"
        )
    return f"""# NS5 final report — {run_id}

- start: {datetime.fromtimestamp(started, tz=timezone.utc).isoformat()}
- elapsed: {elapsed/3600:.2f}h
- branch HEAD: {head}
- theorem_set: {args.theorem_set}
- secondary: {args.secondary_theorem_set or '(disabled)'}
- ckpt_dir: {args.ckpt_dir}
- cycles run: {len(cycle_results)}
- cycles accepted (medium >= {args.baseline_proved_medium}): {len(accepted)}
- cycles promoted to best: {len(promoted)}
- regressing cycles: {len(regressed)}

## Best result

- proved on medium: **{best_proved_medium}** (baseline {args.baseline_proved_medium})
- proved on large:  **{best_proved_large}** (baseline {args.baseline_proved_large})

## Cycle table

{chr(10).join(table)}

## Archive snapshot

```
{arc_summary}
```

## Promotions

{chr(10).join(
    f"- cycle {r.cycle} {r.name}: medium={r.proved_medium} large={r.proved_large}  ({r.notes})"
    for r in promoted
) or '(none)'}

## Regressions

{chr(10).join(
    f"- cycle {r.cycle} {r.name}: lost={r.newly_lost}"
    for r in regressed
) or '(none)'}

## Notes

- See `mutation_log.md` for per-cycle operator detail.
- See `scoreboard.jsonl` for per-cycle JSON rows.
- See `archive_snapshot.jsonl` for the post-run archive state.
"""


if __name__ == "__main__":
    main()
