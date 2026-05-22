"""Evaluator — turns a SearchCandidate into EvalMetrics.

Two modes:

  dry_run=True   Deterministic fake metrics derived from the candidate's
                 config. No Lean, no subprocess. Lets the evolutionary loop
                 be developed and tested in isolation.

  dry_run=False  Calls the existing eval_rollout_all.py via subprocess, then
                 parses the metrics.json it emits.

--- Assumptions about eval_rollout_all.py (verified 2026-05-20) ---
CLI args used:   --theorem-set --policy-type --ckpt-dir --top-k --max-steps
                 --out-dir
metrics.json keys consumed: available, proved, exhausted, errored, per_theorem
                 (each per_theorem row: finished, has_error, num_steps,
                  skip_reason, error_message)

eval_rollout_all.py has NO timeout / fallback-tactic / prompt-template CLI
knob today, so candidate.timeout_per_theorem, candidate.fallback_tactics,
candidate.prompt_template and candidate.tactic_templates are NOT passed to the
real eval yet. Mapping for the EvalMetrics fields:
    attempted_count      <- metrics["available"]
    proved_count         <- metrics["proved"]
    progress_count       <- metrics["exhausted"]   (advanced but did not close)
    total_steps          <- sum of per_theorem num_steps
    timeout_count        <- per_theorem rows whose skip_reason/error_message
                            mentions "timeout"
    invalid_tactic_count <- metrics["errored"]     (search dead-ended on errors)
Revisit this mapping once a strategy-wrapper layer adds real timeout handling.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from evolve.candidate import SearchCandidate
from evolve.scoring import EvalMetrics
from evolve.strategy_wrapper import dump_strategy_config

# Policy types whose checkpoint must be a real generative-model directory.
# Both 'generative' and the v3 'hybrid_evolved' (which wraps GenerativePolicy)
# require this.
_GENERATIVE_FAMILY = {"generative", "hybrid_evolved"}

# Repo root = parent of the evolve/ package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# eval_rollout_all.py writes one subdirectory per call under --out-dir:
#   <out_dir>/<run_id>/{metrics.json, config.json, traces.jsonl}
# where run_id = "eval-<8hex>" (see eval_rollout_all.main). So
# `<out_dir>/*/metrics.json` is the right glob.

# Theorem-set sizes used by dry-run when the real `tasks` module can't be
# imported. Best-effort; the real count is preferred when available.
_FALLBACK_SET_SIZES = {
    "curriculum_all": 31,
    "nat_defs_subset": 15,
    "nat_defs_medium": 38,
}

# Default total-timeout buffer: extra seconds on top of
# (timeout_per_theorem * n_theorems * slack) to absorb startup cost (model
# load, Lean REPL init, mathlib trace lookup). Tuned to be generous; the
# caller can always override via eval_timeout_seconds.
_TIMEOUT_BUFFER_SECONDS = 60
_TIMEOUT_SLACK = 1.05


# --------------------------------------------------------------------------
# dry-run
# --------------------------------------------------------------------------
def _stable_unit(seed: str) -> float:
    """Deterministic float in [0, 1) from a string."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (int(h[:12], 16) % 1_000_000) / 1_000_000.0


def _stable_jitter(seed: str, lo: float, hi: float) -> float:
    return lo + _stable_unit(seed) * (hi - lo)


def _theorem_set_size(theorem_set: str) -> int:
    """Number of theorems in a set. Tries the real `tasks` module first
    (it has no Lean dependency), then falls back to a known-size table."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from tasks import get_theorems  # type: ignore

        return len(get_theorems(theorem_set))
    except Exception:
        return _FALLBACK_SET_SIZES.get(theorem_set, 30)
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT):
            sys.path.pop(0)


def _dry_run_metrics(candidate: SearchCandidate, theorem_set: str) -> EvalMetrics:
    """Deterministic, config-sensitive fake metrics.

    The pseudo-fitness peaks at moderate top_k / max_steps and rewards a richer
    fallback/template genome, so the evolutionary loop has a real gradient to
    climb without ever touching Lean. Same candidate -> same metrics.
    """
    attempted = _theorem_set_size(theorem_set)
    seed = candidate.name

    ratio = 0.45
    # top_k: peak near 10, penalised far from it.
    ratio += 0.12 * (1.0 - abs(candidate.top_k - 10) / 12.0)
    # max_steps: peak near 9.
    ratio += 0.10 * (1.0 - abs(candidate.max_steps - 9) / 9.0)
    # richer fallback / template genome helps, with diminishing returns.
    ratio += 0.020 * min(len(candidate.fallback_tactics), 6)
    ratio += 0.015 * min(len(candidate.tactic_templates), 6)
    # per-candidate noise so siblings differ.
    ratio += _stable_jitter(seed + ":ratio", -0.05, 0.05)
    ratio = max(0.0, min(0.95, ratio))

    proved = round(ratio * attempted)
    remaining = attempted - proved
    progress = round(remaining * (0.40 + _stable_jitter(seed + ":prog", -0.10, 0.10)))
    progress = max(0, min(remaining, progress))

    avg_steps = 3.0 + _stable_jitter(seed + ":steps", 0.0, 4.0)
    total_steps = int(proved * avg_steps + progress * candidate.max_steps)

    timeout = max(0, round(_stable_jitter(seed + ":timeout", -1.5, 2.5)))
    invalid = max(
        0,
        round(remaining * 0.30 + _stable_jitter(seed + ":invalid", -1.0, 2.0)),
    )

    return EvalMetrics(
        attempted_count=attempted,
        proved_count=proved,
        progress_count=progress,
        total_steps=total_steps,
        timeout_count=timeout,
        invalid_tactic_count=invalid,
    )


# --------------------------------------------------------------------------
# real run
# --------------------------------------------------------------------------
def _looks_like_timeout(row: dict) -> bool:
    blob = " ".join(
        str(row.get(k) or "") for k in ("skip_reason", "error_message")
    ).lower()
    return "timeout" in blob or "timed out" in blob


def _parse_eval_metrics(raw: dict) -> EvalMetrics:
    """Best-effort EvalMetrics from an eval_rollout_all.py metrics.json dict."""
    per = raw.get("per_theorem", []) or []
    attempted = int(raw.get("available", sum(1 for r in per if r.get("available"))))
    proved = int(raw.get("proved", sum(1 for r in per if r.get("finished"))))
    progress = int(
        raw.get(
            "exhausted",
            sum(
                1
                for r in per
                if r.get("available")
                and not r.get("finished")
                and not r.get("has_error")
            ),
        )
    )
    total_steps = sum(int(r.get("num_steps") or 0) for r in per)
    timeout = sum(1 for r in per if _looks_like_timeout(r))
    invalid = int(raw.get("errored", sum(1 for r in per if r.get("has_error"))))
    return EvalMetrics(
        attempted_count=attempted,
        proved_count=proved,
        progress_count=progress,
        total_steps=total_steps,
        timeout_count=timeout,
        invalid_tactic_count=invalid,
    )


def _newest_metrics_json(out_dir: Path, since_mtime: float) -> Optional[Path]:
    """Find the metrics.json written after `since_mtime` under out_dir."""
    candidates = [
        p
        for p in out_dir.glob("*/metrics.json")
        if p.stat().st_mtime >= since_mtime
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_ckpt_dir(ckpt_dir: Optional[str]) -> Optional[Path]:
    """Resolve ckpt_dir against REPO_ROOT if it is relative. Returns None if
    the caller passed no ckpt_dir."""
    if not ckpt_dir:
        return None
    p = Path(ckpt_dir)
    return p if p.is_absolute() else (REPO_ROOT / p)


def check_ckpt_exists(candidate: SearchCandidate) -> None:
    """Pre-flight check: raise FileNotFoundError if a candidate from the
    generative family ('generative', 'hybrid_evolved') has a missing
    checkpoint directory. Classifier checkpoints (e.g. clf_ckpt) are not
    validated here because the legacy default may not always be on disk
    during dry-run development."""
    if candidate.policy_type not in _GENERATIVE_FAMILY:
        return
    resolved = _resolve_ckpt_dir(candidate.ckpt_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"candidate '{candidate.name}' has policy_type={candidate.policy_type} "
            "but no ckpt_dir set; cannot run real eval."
        )
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory not found for candidate '{candidate.name}': "
            f"policy_type={candidate.policy_type}, ckpt_dir={candidate.ckpt_dir} "
            f"(resolved to {resolved})"
        )


def _derive_eval_timeout(candidate: SearchCandidate, theorem_set: str) -> int:
    """Default total-subprocess timeout: per-theorem * count * slack + buffer."""
    n = _theorem_set_size(theorem_set)
    return max(
        _TIMEOUT_BUFFER_SECONDS,
        int(candidate.timeout_per_theorem * n * _TIMEOUT_SLACK + _TIMEOUT_BUFFER_SECONDS),
    )


def _real_metrics(
    candidate: SearchCandidate,
    theorem_set: str,
    output_dir: str,
    eval_timeout_seconds: Optional[int] = None,
) -> EvalMetrics:
    """Run eval_rollout_all.py as a subprocess and parse its metrics.json.

    The subprocess's combined stdout+stderr is streamed by a background thread
    into `<output_dir>/subprocess.log` so a real run (minutes) is observable
    via `tail -f` rather than buffered in memory.

    Timeout: if the subprocess does not finish within eval_timeout_seconds (or
    the derived default of timeout_per_theorem * n_theorems * slack + buffer),
    the subprocess is killed and synthetic metrics with
    timeout_count = n_theorems are returned. The score function will then
    crush the candidate via the timeout penalty.
    """
    check_ckpt_exists(candidate)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "subprocess.log"
    before = time.time()

    if eval_timeout_seconds is None:
        eval_timeout_seconds = _derive_eval_timeout(candidate, theorem_set)
    n_theorems = _theorem_set_size(theorem_set)

    cmd = [
        sys.executable,
        "eval_rollout_all.py",
        "--theorem-set", theorem_set,
        "--policy-type", candidate.policy_type,
        "--ckpt-dir", candidate.ckpt_dir or "clf_ckpt",
        "--top-k", str(candidate.top_k),
        "--max-steps", str(candidate.max_steps),
        "--out-dir", str(out_dir),
    ]

    # The v3 strategy wrapper consumes a JSON config the candidate provides.
    # Write it next to subprocess.log so artifacts stay co-located.
    if candidate.policy_type == "hybrid_evolved":
        strategy_path = out_dir / "strategy_config.json"
        dump_strategy_config(
            strategy_path,
            candidate.fallback_tactics,
            candidate.tactic_templates,
            max_extra_tactics_per_state=candidate.max_extra_tactics_per_state,
            theorem_family_tactics=getattr(
                candidate, "theorem_family_tactics", None
            ),
            family_budgets=getattr(candidate, "family_budgets", None),
            theorem_tactic_denylist=getattr(
                candidate, "theorem_tactic_denylist", None
            ),
            retrieval_enabled=getattr(candidate, "retrieval_enabled", False),
            retrieval_top_k=getattr(candidate, "retrieval_top_k", 0),
            retrieval_tactic_forms=getattr(
                candidate, "retrieval_tactic_forms", None
            ),
        )
        cmd.extend(["--strategy-config", str(strategy_path)])

    # v3.3 per-candidate anti-loop opt-in.
    if getattr(candidate, "enable_loop_avoidance", False):
        cmd.append("--enable-loop-avoidance")

    print(
        f"      [evaluator] $ {' '.join(cmd)}\n"
        f"      [evaluator] timeout {eval_timeout_seconds}s "
        f"({candidate.timeout_per_theorem}s/thm × {n_theorems} thm + buffer)\n"
        f"      [evaluator] log: {log_path}",
        flush=True,
    )

    timed_out = False
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        def _drain() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                logf.write(line)
                logf.flush()

        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()

        try:
            returncode = proc.wait(timeout=eval_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                returncode = -9

        # Wait for the drain thread to flush remaining buffered output.
        drain.join(timeout=5)

        if timed_out:
            logf.write(
                f"\n--- TIMED OUT after {eval_timeout_seconds}s; subprocess killed ---\n"
            )
            logf.flush()

    if timed_out:
        print(
            f"      [evaluator] TIMEOUT — killed after {eval_timeout_seconds}s",
            flush=True,
        )
        return EvalMetrics(
            attempted_count=n_theorems,
            proved_count=0,
            progress_count=0,
            total_steps=0,
            timeout_count=n_theorems,
            invalid_tactic_count=0,
        )

    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-2500:]
        raise RuntimeError(
            f"eval_rollout_all.py failed (exit {returncode}).\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  log: {log_path}\n"
            f"  log tail:\n{tail}"
        )

    metrics_path = _newest_metrics_json(out_dir, before)
    if metrics_path is None:
        tail = log_path.read_text(encoding="utf-8")[-2500:]
        raise RuntimeError(
            f"eval_rollout_all.py exited 0 but wrote no metrics.json under {out_dir}.\n"
            f"  log: {log_path}\n"
            f"  log tail:\n{tail}"
        )
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    return _parse_eval_metrics(raw)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def evaluate_candidate(
    candidate: SearchCandidate,
    theorem_set: str,
    output_dir: str,
    dry_run: bool = False,
    eval_timeout_seconds: Optional[int] = None,
) -> EvalMetrics:
    """Evaluate one candidate over a theorem set.

    dry_run=True returns deterministic fake metrics (no Lean). dry_run=False
    shells out to eval_rollout_all.py and parses its metrics.json. The
    subprocess is killed and a timeout-penalised EvalMetrics is returned if
    it does not finish within eval_timeout_seconds (or the derived default
    of `candidate.timeout_per_theorem * n_theorems * 1.05 + 60`).
    """
    if dry_run:
        return _dry_run_metrics(candidate, theorem_set)
    return _real_metrics(
        candidate, theorem_set, output_dir,
        eval_timeout_seconds=eval_timeout_seconds,
    )
