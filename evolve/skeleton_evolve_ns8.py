"""NS8 — rank-simulation skeleton evolution.

Replaces NS7's bag-only pre-flight detector with a full
ranked-list simulator (`evolve.rank_simulator.RankSimulator`) backed
by cached `gen_v5` model outputs. The simulator builds the wrapper's
actual `last_ranked_tactics` for each protected state and rejects
mutations whose critical tactic drops out of the list.

CLI:

    python -m evolve.skeleton_evolve_ns8 \\
        --theorem-set nat_defs_medium \\
        --secondary-theorem-set nat_defs_large_v5 \\
        --seed-genome project/evolve/ns7_runs/baseline/genome.json \\
        --baseline-traces project/evolve/ns7_runs/baseline/medium/eval-X/traces.jsonl \\
        --baseline-traces project/evolve/ns7_runs/baseline/large/eval-Y/traces.jsonl \\
        --protected-states project/evolve/archive/protected_states.jsonl \\
        --model-cache project/evolve/archive/model_outputs_cache.jsonl \\
        --out-dir project/evolve/ns8_runs \\
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
from evolve.rank_coupling import (
    check_state_coupling, summarize_state_violations,
)
from evolve.rank_simulator import RankSimulator
from evolve.skeleton_archive import (
    aggregate, load_archive, update_archive_from_metrics_path,
)
from evolve.skeleton_mutator import (
    MutationRecord, apply_operator, bag_to_genome, genome_to_bag,
)


DEFAULT_QUEUE: list[tuple[str, dict]] = [
    ("baseline", {}),
    # Safe pruning — credit-aware, plus NS8 simulator pre-flight.
    ("disable_dead_skeleton", {"min_attempts": 3, "max_disable": 2}),
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 3}),
    ("disable_dead_skeleton", {"min_attempts": 8, "max_disable": 5}),
    # Credit-aware archive_seed at multiple top_n.
    ("archive_seed_credit", {"top_n": 15}),
    ("archive_seed_credit", {"top_n": 18}),
    ("archive_seed_credit", {"top_n": 20}),
    ("archive_seed_credit", {"top_n": 22}),
    ("archive_seed_credit", {"top_n": 25}),
    # Wins-only archive_seed — NS8 should reject pre-flight.
    ("archive_seed", {"top_n": 18}),
    ("archive_seed", {"top_n": 22}),
    # Scoped reorders.
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "priority_template", "scope_shape": "iff"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "fallback_tactic", "scope_shape": "any"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "tactic_template", "scope_shape": "any"}),
    ("promote_high_win_skeleton",
     {"top_n": 10, "scope_origin": "family_tactic", "scope_shape": "any"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "iff"}),
    ("demote_generic_skeleton",
     {"scope_origin": "priority_template", "scope_shape": "any"}),
    # Final aggressive prune.
    ("disable_dead_skeleton", {"min_attempts": 5, "max_disable": 8}),
    ("disable_dead_skeleton", {"min_attempts": 10, "max_disable": 12}),
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
    preflight_rejected: bool = False
    preflight_violations: int = 0
    preflight_affected_theorems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CREDIT_SEEN_PATHS: set[str] = set()


def _update_credit_from_traces(
    credit: dict[str, dict[str, int]],
    traces_path: Path | None,
) -> None:
    if traces_path is None or not traces_path.exists():
        return
    key = str(traces_path.resolve())
    if key in _CREDIT_SEEN_PATHS:
        return
    _CREDIT_SEEN_PATHS.add(key)
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


def _eval_one(
    genome: dict[str, Any], theorem_set: str, candidate_name: str,
    out_root: Path, ckpt_dir: str, eval_timeout_seconds: int, log_fn,
) -> tuple[Optional[Path], dict[str, Any], Optional[Path]]:
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
    log_fn(f"  [{candidate_name}] eval start ({theorem_set}, timeout {eval_timeout_seconds}s)")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        try:
            r = subprocess.run(
                cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
                timeout=eval_timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            log_fn(f"  [{candidate_name}] TIMEOUT after {eval_timeout_seconds}s")
            return None, {}, None
    elapsed = time.time() - started
    log_fn(f"  [{candidate_name}] eval done in {elapsed:.0f}s (rc={r.returncode})")
    if r.returncode != 0:
        return None, {}, None
    metrics_paths = sorted(eval_dir.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    traces_paths = sorted(eval_dir.glob("*/traces.jsonl"), key=lambda p: p.stat().st_mtime)
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


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theorem-set", default="nat_defs_medium")
    ap.add_argument("--secondary-theorem-set", default="nat_defs_large_v5")
    ap.add_argument("--seed-genome", type=Path, required=True)
    ap.add_argument("--baseline-traces", action="append", default=[])
    ap.add_argument("--protected-states", type=Path, required=True)
    ap.add_argument("--model-cache", type=Path, required=True)
    ap.add_argument("--ckpt-dir", default="project/models/gen_v5")
    ap.add_argument("--out-dir", type=Path, default=Path("project/evolve/ns8_runs"))
    ap.add_argument("--max-cycles", type=int, default=20)
    ap.add_argument("--per-medium-timeout", type=int, default=900)
    ap.add_argument("--per-large-timeout", type=int, default=1500)
    ap.add_argument("--max-large-evals", type=int, default=3)
    ap.add_argument("--preflight-disable", action="store_true")
    args = ap.parse_args()

    run_id = f"ns8-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_root = run_dir / "eval"
    eval_root.mkdir(exist_ok=True)
    scoreboard_path = run_dir / "scoreboard.jsonl"
    mutation_log = run_dir / "mutation_log.md"
    research_log = run_dir / "research_log.md"

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with research_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    seed_genome = json.loads(args.seed_genome.read_text(encoding="utf-8"))
    if "genome" in seed_genome and isinstance(seed_genome["genome"], dict):
        seed_genome = seed_genome["genome"]
    seed_genome.setdefault("use_skeleton_bag", True)
    log(f"seed genome: {args.seed_genome}")
    log(f"  enabled skeletons: {_enabled_skeleton_count(seed_genome)}")

    # Load protected states + model cache.
    protected_states: list[dict[str, Any]] = []
    with args.protected_states.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            protected_states.append(json.loads(line))
    log(f"protected states: {len(protected_states)} from {args.protected_states}")

    simulator = RankSimulator(args.model_cache)
    log(f"model cache: {args.model_cache}")

    credit: dict[str, dict[str, int]] = {}
    for tp in args.baseline_traces:
        _update_credit_from_traces(credit, Path(tp))
    log(f"credit index: {len(credit)} entries from {len(args.baseline_traces)} trace file(s)")

    archive_rows = load_archive()
    archive_stats = aggregate(archive_rows)
    log(f"archive: {len(archive_rows)} rows, {len(archive_stats)} distinct skeletons")

    best_genome = deepcopy(seed_genome)
    best_proved_medium: int | None = None
    best_proved_large: int | None = None
    best_enabled = _enabled_skeleton_count(best_genome)
    best_cycle = -1

    results: list[CycleResult] = []
    queue = list(DEFAULT_QUEUE)[:args.max_cycles]
    n_large_evals = 0
    preflight_rejected_count = 0

    for cycle_idx, (op_name, raw_kwargs) in enumerate(queue, start=1):
        kwargs = dict(raw_kwargs)
        if op_name == "baseline":
            cand = deepcopy(best_genome)
            records: list[MutationRecord] = [MutationRecord(
                operator="baseline",
                description="Re-evaluate current best genome (no mutation).",
                rationale="Sampling-noise sanity check.",
            )]
        else:
            if op_name in ("disable_dead_skeleton", "archive_seed_credit"):
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

        # NS8 pre-flight: full ranked-list simulation.
        preflight_violations = 0
        preflight_affected: list[str] = []
        preflight_rejected = False
        if not args.preflight_disable and op_name != "baseline":
            violations = check_state_coupling(
                best_genome, cand, protected_states, simulator,
            )
            preflight_violations = len(violations)
            summary = summarize_state_violations(violations)
            preflight_affected = summary["affected_theorems"]
            if violations:
                preflight_rejected = True
                preflight_rejected_count += 1
                log(f"    NS8 PRE-FLIGHT REJECTED: {preflight_violations} violations, "
                    f"theorems={preflight_affected[:3]}")

        proved_med: int | None = None
        proved_large: int | None = None
        mp_med = None
        mp_large = None
        notes = ""
        accepted = False
        promoted = False
        cycle_started = time.time()

        if preflight_rejected:
            notes = (
                f"NS8 pre-flight rejected: {preflight_violations} violations "
                f"(theorems={preflight_affected[:2]})"
            )
        else:
            mp_med, met_med, tp_med = _eval_one(
                cand, args.theorem_set, candidate_name + "_medium",
                eval_root, args.ckpt_dir, args.per_medium_timeout, log,
            )
            if met_med:
                proved_med = int(met_med.get("proved") or 0)
                log(f"    medium: proved={proved_med}")
                update_archive_from_metrics_path(mp_med, run_id=run_id,
                                                 last_seen_commit=_git_head())
                archive_rows = load_archive()
                archive_stats = aggregate(archive_rows)
            _update_credit_from_traces(credit, tp_med)

            proved_med_int = proved_med if isinstance(proved_med, int) else -1
            baseline_floor = best_proved_medium if best_proved_medium is not None else 37
            if proved_med_int >= baseline_floor:
                accepted = True
                if best_proved_medium is None:
                    best_proved_medium = proved_med_int
                    best_genome = deepcopy(cand)
                    best_enabled = _enabled_skeleton_count(best_genome)
                    promoted = True
                    best_cycle = cycle_idx
                else:
                    strict_improve = proved_med_int > best_proved_medium
                    enabled_count = _enabled_skeleton_count(cand)
                    strict_compact = (
                        proved_med_int == best_proved_medium
                        and enabled_count < best_enabled
                    )
                    if strict_improve or strict_compact:
                        promoted = True
                        best_proved_medium = proved_med_int
                        best_genome = deepcopy(cand)
                        best_enabled = _enabled_skeleton_count(best_genome)
                        best_cycle = cycle_idx
                        notes = "promoted: " + (
                            "strict_improve" if strict_improve else "strict_compact"
                        )
                    else:
                        notes = "accepted, no promotion"
            else:
                notes = (
                    f"Lean rejected: proved={proved_med_int} < best_floor={baseline_floor}"
                )

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
                    _update_credit_from_traces(credit, tp_large)
                    if best_proved_large is None or proved_large >= best_proved_large:
                        best_proved_large = proved_large

        result = CycleResult(
            cycle=cycle_idx, name=candidate_name,
            operator=op_name,
            operator_kwargs={k: v for k, v in kwargs.items() if k != "credit_stats"},
            scope=scope,
            proved_medium=proved_med, proved_large=proved_large,
            runtime_seconds=time.time() - cycle_started,
            medium_metrics_path=str(mp_med) if mp_med else None,
            large_metrics_path=str(mp_large) if mp_large else None,
            accepted=accepted, promoted_to_best=promoted, notes=notes,
            mutation_records=[r.to_dict() for r in records],
            enabled_skeletons_count=_enabled_skeleton_count(cand),
            preflight_rejected=preflight_rejected,
            preflight_violations=preflight_violations,
            preflight_affected_theorems=preflight_affected,
        )
        results.append(result)
        with scoreboard_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        with mutation_log.open("a", encoding="utf-8") as f:
            f.write(f"\n## cycle {cycle_idx} — {candidate_name}\n\n")
            for r in records:
                f.write(r.to_md_line() + "\n")
            f.write(f"\n**preflight_rejected**: {preflight_rejected} "
                    f"(violations={preflight_violations})\n")
            f.write(f"**accepted**: {accepted}, **promoted**: {promoted}\n")
            f.write(f"**proved_medium**: {proved_med}, **proved_large**: {proved_large}\n")
            f.write(f"**enabled**: {result.enabled_skeletons_count}\n")
            if notes:
                f.write(f"**notes**: {notes}\n")
            f.write("\n")

    best_path = run_dir / "best_candidate.json"
    best_path.write_text(json.dumps({
        "run_id": run_id, "best_cycle": best_cycle,
        "proved_medium": best_proved_medium, "proved_large": best_proved_large,
        "enabled_skeletons": best_enabled,
        "preflight_rejected_count": preflight_rejected_count,
        "genome": best_genome,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"DONE — best_cycle={best_cycle}, "
        f"medium={best_proved_medium}, large={best_proved_large}, "
        f"enabled={best_enabled}, preflight_rejected={preflight_rejected_count}")


if __name__ == "__main__":
    main()
