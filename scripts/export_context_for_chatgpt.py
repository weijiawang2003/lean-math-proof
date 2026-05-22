"""Export a compact ChatGPT-ready snapshot of project state.

Produces project/evolve/reports/chatgpt_context.md, summarising git state,
the best evaluation result so far, the most recent experiment, key reports,
and open problems. Designed to be uploaded to ChatGPT for next-step advice.

Safe to run from the repo root:

    python scripts/export_context_for_chatgpt.py

The script touches only small JSON/JSONL/Markdown artefacts; it never reads
checkpoints, full traces, or subprocess logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "project" / "evolve" / "reports"
RUNS_DIR = REPO_ROOT / "project" / "evolve" / "autonomous_runs"
OUTPUT_PATH = REPORTS_DIR / "chatgpt_context.md"

# Reports we surface if present, in display order.
KEY_REPORT_NAMES = [
    "nat_defs_medium_summary.md",
    "V5_README.md",
    "v5_central_findings.md",
    "v5_autonomous_exploration.md",
    "v5_priority_templates_insight.md",
    "v5_evaluation_summary.md",
    "v5_alphaevolve_architecture.md",
    "v5_trace_to_training_plan.md",
    "v5_next_steps.md",
    "v5_lessons_for_contributors.md",
    "v5_failure_deep_dive_add_mod_eq_ite.md",
    "nat_defs_medium_failure_classification_v5.md",
]

# Path fragments we treat as "model/checkpoint artifacts" — flagged separately
# in the git status section. These are gitignored binaries in normal use.
CHECKPOINT_FRAGMENTS = ("models/", "checkpoint", "merged_ckpt", ".bin", ".safetensors")

# Path fragments we never want to surface in the changed-files list.
NOISY_PATH_FRAGMENTS = (
    "autonomous_runs/",
    "traces.jsonl",
    "subprocess.log",
    "_console.log",
    ".pyc",
)

MAX_CHANGED_FILES = 60
MAX_REPORT_LIST = 20
MAX_RUNS_SCANNED = 200


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def run_git(args: list[str]) -> str:
    """Run a git command from the repo root; return stdout or '' on error."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    return proc.stdout.rstrip("\n") if proc.returncode == 0 else ""


def read_json(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def section_git_state() -> str:
    out = ["## 1. Current Git State", ""]

    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "(unknown)"
    out.append(f"- **Branch:** `{branch}`")

    log = run_git(["log", "--oneline", "-n", "10"])
    out.append("- **Latest 10 commits:**")
    out.append("")
    out.append("```")
    out.append(log if log else "(no commits found)")
    out.append("```")
    out.append("")

    status = run_git(["status", "--short"])
    status_lines = [ln for ln in status.splitlines() if ln.strip()]

    code_changes: list[str] = []
    checkpoint_changes: list[str] = []
    for ln in status_lines:
        path_part = ln[3:].strip()
        if any(frag in path_part for frag in CHECKPOINT_FRAGMENTS):
            checkpoint_changes.append(ln)
        elif any(frag in path_part for frag in NOISY_PATH_FRAGMENTS):
            continue
        else:
            code_changes.append(ln)

    out.append(f"- **Working tree:** {len(status_lines)} entries in `git status --short`.")
    if code_changes:
        out.append(f"  - {len(code_changes)} non-artefact entries — WORKING TREE HAS UNCOMMITTED CODE CHANGES.")
        out.append("")
        out.append("  ```")
        for ln in code_changes[:30]:
            out.append("  " + ln)
        if len(code_changes) > 30:
            out.append(f"  ... ({len(code_changes) - 30} more)")
        out.append("  ```")
    else:
        out.append("  - 0 non-artefact entries — code working tree is clean.")

    if checkpoint_changes:
        out.append("")
        out.append(
            f"- **WARNING:** {len(checkpoint_changes)} checkpoint/model artefact "
            "paths show as modified/deleted/untracked. These should be gitignored; "
            "do not commit them."
        )
        out.append("")
        out.append("  ```")
        for ln in checkpoint_changes[:15]:
            out.append("  " + ln)
        if len(checkpoint_changes) > 15:
            out.append(f"  ... ({len(checkpoint_changes) - 15} more)")
        out.append("  ```")

    out.append("")
    return "\n".join(out)


@dataclass
class ScoreRow:
    name: str
    proved: int
    delta_vs_baseline: int | None
    newly_proved: list[str]
    newly_lost: list[str]
    runtime_seconds: float | None
    eval_dir: str
    scoreboard_path: Path
    run_dir: Path
    theorem_set: str | None = None
    available: int | None = None
    total_theorems: int | None = None


def iter_scoreboard_rows() -> Iterable[ScoreRow]:
    """Yield every scoreboard row from autonomous_runs/*/scoreboard.jsonl."""
    if not RUNS_DIR.exists():
        return
    candidates = sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:MAX_RUNS_SCANNED]
    for run_dir in candidates:
        scoreboard = run_dir / "scoreboard.jsonl"
        if not scoreboard.exists():
            continue
        run_config = read_json(run_dir / "config.json") or {}
        run_theorem_set = run_config.get("theorem_set")
        for row in read_jsonl(scoreboard):
            eval_dir = row.get("eval_dir", "")
            yield ScoreRow(
                name=row.get("name", "?"),
                proved=int(row.get("proved", 0)),
                delta_vs_baseline=row.get("delta_vs_baseline"),
                newly_proved=list(row.get("newly_proved", [])),
                newly_lost=list(row.get("newly_lost", [])),
                runtime_seconds=row.get("runtime_seconds"),
                eval_dir=eval_dir,
                scoreboard_path=scoreboard,
                run_dir=run_dir,
                theorem_set=row.get("theorem_set") or run_theorem_set,
            )


def standalone_eval_runs() -> list[tuple[Path, dict]]:
    """Pick up runs that have a metrics.json but no scoreboard.jsonl.

    These are the single-config runs (e.g. large_v5_master, demo_v1_master,
    nat_defs_medium_raw_baseline). We attach the run-level config + metrics
    for downstream sections.
    """
    out: list[tuple[Path, dict]] = []
    if not RUNS_DIR.exists():
        return out
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        if (run_dir / "scoreboard.jsonl").exists():
            continue
        # Find a metrics.json under run_dir, two levels deep at most.
        metrics_files = sorted(run_dir.glob("*/eval-*/metrics.json")) + sorted(
            run_dir.glob("eval-*/metrics.json")
        )
        for mfile in metrics_files:
            metrics = read_json(mfile)
            if metrics:
                out.append((run_dir, metrics))
                break
    return out


def best_on_theorem_set(target_set: str) -> ScoreRow | None:
    best: ScoreRow | None = None
    for row in iter_scoreboard_rows():
        if row.theorem_set and row.theorem_set != target_set:
            continue
        if best is None or row.proved > best.proved:
            best = row
    return best


def genome_path_for(row: ScoreRow) -> Path | None:
    """The genome.json sits one level above the eval-<hash>/ dir."""
    if not row.eval_dir:
        return None
    parent = Path(row.eval_dir).parent
    g = parent / "genome.json"
    return g if g.exists() else None


def section_latest_milestone() -> str:
    out = ["## 2. Latest Project Milestone", ""]

    best_medium = best_on_theorem_set("nat_defs_medium")
    best_large = best_on_theorem_set("nat_defs_large_v5")

    # Fallback: check single-config runs for nat_defs_large_v5 (large_v5_master).
    if best_large is None:
        for run_dir, metrics in standalone_eval_runs():
            if metrics.get("theorem_set") == "nat_defs_large_v5":
                eval_dirs = sorted(run_dir.glob("**/eval-*"))
                eval_dir = str(eval_dirs[0]) if eval_dirs else ""
                best_large = ScoreRow(
                    name=run_dir.name,
                    proved=int(metrics.get("proved", 0)),
                    delta_vs_baseline=None,
                    newly_proved=[],
                    newly_lost=[],
                    runtime_seconds=None,
                    eval_dir=eval_dir,
                    scoreboard_path=run_dir,
                    run_dir=run_dir,
                    theorem_set="nat_defs_large_v5",
                    available=metrics.get("available"),
                    total_theorems=metrics.get("total_theorems"),
                )
                break

    if best_medium:
        out.append(
            f"- **Best on `nat_defs_medium`:** `{best_medium.name}` — "
            f"**{best_medium.proved}** proved "
            f"(Δ vs scoreboard baseline: {best_medium.delta_vs_baseline}). "
            f"Run: `{best_medium.run_dir.relative_to(REPO_ROOT)}`."
        )
        if best_medium.newly_proved:
            out.append(
                "  - New wins captured in this row: "
                + ", ".join(f"`{n}`" for n in best_medium.newly_proved)
            )
        g = genome_path_for(best_medium)
        if g:
            out.append(f"- **Best genome:** `{g.relative_to(REPO_ROOT)}`")
    else:
        out.append("- **Best on `nat_defs_medium`:** (no scoreboard data found)")

    if best_large:
        avail = best_large.available
        rate_str = ""
        if avail:
            rate_str = f" ({best_large.proved}/{avail} = {best_large.proved / avail:.0%})"
        out.append(
            f"- **Best on `nat_defs_large_v5`:** `{best_large.name}` — "
            f"**{best_large.proved}** proved{rate_str}. "
            f"Run: `{best_large.run_dir.relative_to(REPO_ROOT)}`."
        )
    else:
        out.append("- **Best on `nat_defs_large_v5`:** (not evaluated)")

    # Latest experiment stage: read from V5_README.md if present, else from
    # recent reports filenames.
    stages = sorted(
        {
            p.name.split("_")[0]
            for p in REPORTS_DIR.glob("v*_*.md")
            if p.name[1:2].isdigit()
        }
    )
    if stages:
        out.append(f"- **Experiment stages with reports:** {', '.join(stages)}")
    out.append(
        "- **Best report (orientation):** "
        + ("`project/evolve/reports/V5_README.md`" if (REPORTS_DIR / "V5_README.md").exists() else "(none)")
    )

    out.append("")
    return "\n".join(out)


def section_recent_experiment() -> str:
    out = ["## 3. Recent Experiment Summary", ""]

    rows: list[ScoreRow] = list(iter_scoreboard_rows())
    if not rows:
        # Fall back to a standalone run.
        standalones = standalone_eval_runs()
        if standalones:
            run_dir, metrics = standalones[0]
            out.append(f"- **Run dir:** `{run_dir.relative_to(REPO_ROOT)}`")
            out.append(f"- **Theorem set:** `{metrics.get('theorem_set', '?')}`")
            out.append(f"- **Policy type:** `{metrics.get('policy_type', '?')}`")
            out.append(
                f"- **Proved / available / total:** "
                f"{metrics.get('proved', '?')} / {metrics.get('available', '?')} / "
                f"{metrics.get('total_theorems', '?')}"
            )
            out.append("")
            return "\n".join(out)
        out.append("- (no recent runs found in `project/evolve/autonomous_runs/`)")
        out.append("")
        return "\n".join(out)

    # Find the most recently-modified scoreboard, then take its last row.
    latest_run_dir = max({r.run_dir for r in rows}, key=lambda p: p.stat().st_mtime)
    latest_rows = [r for r in rows if r.run_dir == latest_run_dir]
    if not latest_rows:
        out.append("- (no rows in latest scoreboard)")
        return "\n".join(out)
    last = latest_rows[0]  # iter is mtime-desc; rows preserve scoreboard order
    # Actually pick the last-written row in the scoreboard file (final cycle).
    raw_rows = read_jsonl(latest_run_dir / "scoreboard.jsonl")
    if raw_rows:
        last_raw = raw_rows[-1]
        last = ScoreRow(
            name=last_raw.get("name", "?"),
            proved=int(last_raw.get("proved", 0)),
            delta_vs_baseline=last_raw.get("delta_vs_baseline"),
            newly_proved=list(last_raw.get("newly_proved", [])),
            newly_lost=list(last_raw.get("newly_lost", [])),
            runtime_seconds=last_raw.get("runtime_seconds"),
            eval_dir=last_raw.get("eval_dir", ""),
            scoreboard_path=latest_run_dir / "scoreboard.jsonl",
            run_dir=latest_run_dir,
            theorem_set=last_raw.get("theorem_set"),
        )

    run_config = read_json(latest_run_dir / "config.json") or {}
    theorem_set = last.theorem_set or run_config.get("theorem_set", "?")

    # Try to read the per-eval metrics.json for the last cycle for richer stats.
    metrics: dict = {}
    if last.eval_dir:
        m = read_json(Path(last.eval_dir) / "metrics.json")
        if m:
            metrics = m
    available = metrics.get("available")
    total = metrics.get("total_theorems")
    policy_type = metrics.get("policy_type", "hybrid_evolved")

    out.append(f"- **Latest run dir:** `{latest_run_dir.relative_to(REPO_ROOT)}`")
    out.append(f"- **Last cycle:** `{last.name}`")
    out.append(f"- **Theorem set:** `{theorem_set}`")
    out.append(f"- **Policy type:** `{policy_type}`")
    if available is not None and total is not None:
        out.append(
            f"- **Proved / available / total:** {last.proved} / {available} / {total}"
        )
    else:
        out.append(f"- **Proved:** {last.proved}")
    if last.delta_vs_baseline is not None:
        out.append(f"- **Δ vs scoreboard baseline:** {last.delta_vs_baseline}")
    if last.runtime_seconds is not None:
        out.append(f"- **Runtime:** {last.runtime_seconds:.0f} s")
    if last.newly_proved:
        out.append("- **New wins:** " + ", ".join(f"`{n}`" for n in last.newly_proved))
    else:
        out.append("- **New wins:** none")
    if last.newly_lost:
        out.append("- **Regressions:** " + ", ".join(f"`{n}`" for n in last.newly_lost))
    else:
        out.append("- **Regressions:** none")

    # Crash and unknown-constant counts: scan metrics.json's errored field +
    # peek at progress_summary.json if present (we only count, never store).
    if metrics:
        out.append(f"- **Errored count:** {metrics.get('errored', '?')}")
    progress_path = (
        Path(last.eval_dir).parent / "progress_summary.json" if last.eval_dir else None
    )
    if progress_path and progress_path.exists():
        progress = read_json(progress_path)
        if isinstance(progress, list):
            crash_n = sum(1 for r in progress if "DojoCrash" in json.dumps(r))
            unknown_n = sum(1 for r in progress if "unknown constant" in json.dumps(r))
            out.append(f"- **Approx. crash count:** {crash_n}")
            out.append(f"- **Approx. unknown-constant count:** {unknown_n}")

    # Brief comparison: how does the latest cycle compare to that run's own
    # first cycle (the in-run baseline)?
    if len(raw_rows) >= 2:
        first = raw_rows[0]
        out.append(
            f"- **vs first cycle in same run** "
            f"(`{first.get('name')}`, proved {first.get('proved')}): "
            f"Δ = {last.proved - int(first.get('proved', 0))}"
        )

    out.append("")
    return "\n".join(out)


def section_important_reports() -> str:
    out = ["## 4. Important Reports", ""]
    if not REPORTS_DIR.exists():
        out.append("- (no reports directory)")
        out.append("")
        return "\n".join(out)

    listed: set[str] = set()
    for name in KEY_REPORT_NAMES:
        path = REPORTS_DIR / name
        if path.exists():
            size_kb = path.stat().st_size // 1024
            out.append(f"- `{path.relative_to(REPO_ROOT)}` ({size_kb} KB)")
            listed.add(name)

    # Surface any newer v6/v7 reports that aren't in the canonical list.
    extras = []
    for path in sorted(REPORTS_DIR.glob("v[6-9]_*.md")):
        if path.name not in listed:
            extras.append(path)
    for path in sorted(REPORTS_DIR.glob("V[6-9]_*.md")):
        if path.name not in listed:
            extras.append(path)
    if extras:
        out.append("")
        out.append("**Newer stage reports:**")
        for p in extras[:MAX_REPORT_LIST]:
            size_kb = p.stat().st_size // 1024
            out.append(f"- `{p.relative_to(REPO_ROOT)}` ({size_kb} KB)")

    out.append("")
    return "\n".join(out)


def _extract_section(text: str, header_prefix: str, max_lines: int = 25) -> list[str]:
    """Pull lines from `text` starting at the first heading matching the prefix,
    until the next heading at the same or higher level. Keep `max_lines`."""
    lines = text.splitlines()
    start = None
    level = None
    for i, ln in enumerate(lines):
        if ln.startswith("#") and header_prefix.lower() in ln.lower():
            start = i
            level = len(ln) - len(ln.lstrip("#"))
            break
    if start is None:
        return []
    out: list[str] = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            if lvl <= level:
                break
        out.append(ln)
        if len(out) >= max_lines:
            out.append("...")
            break
    return out


def section_open_problems() -> str:
    out = ["## 5. Open Problems", ""]

    # Phase inferred from latest report names.
    if (REPORTS_DIR / "v5_alphaevolve_architecture.md").exists() and not any(
        REPORTS_DIR.glob("v6_*.md")
    ):
        phase = "**priority-template / skeleton-bag transition** (v5 → v6)"
    elif any(REPORTS_DIR.glob("v6_*.md")):
        phase = "**skeleton-bag architecture (v6)**"
    elif any(REPORTS_DIR.glob("v7_*.md")):
        phase = "**training-data / model-training (v7)**"
    else:
        phase = "**tactic-ordering / priority-template (v5)**"
    out.append(f"- **Current phase:** {phase}")

    # Remaining failed theorems — sourced from V5_README.md's tail section.
    readme = REPORTS_DIR / "V5_README.md"
    if readme.exists():
        text = readme.read_text(errors="replace")
        unsolved = _extract_section(text, "What didn't close", max_lines=40)
        if unsolved:
            out.append("")
            out.append("**Remaining failed theorems (from `V5_README.md`):**")
            out.append("")
            out.extend(unsolved[1:])  # skip the header itself

    next_steps = REPORTS_DIR / "v5_next_steps.md"
    if next_steps.exists():
        text = next_steps.read_text(errors="replace")
        priorities = _extract_section(text, "Priority for the v5", max_lines=30)
        if priorities:
            out.append("")
            out.append("**Proposed priorities (from `v5_next_steps.md`):**")
            out.append("")
            out.extend(priorities[1:])

    out.append("")
    return "\n".join(out)


def section_changed_files() -> str:
    out = ["## 6. Changed Files Since `main`", ""]
    diff = run_git(["diff", "--name-only", "main...HEAD"])
    if not diff:
        out.append("- (no diff against `main`, or `main` not present)")
        out.append("")
        return "\n".join(out)
    files = [
        ln for ln in diff.splitlines() if ln and not any(f in ln for f in NOISY_PATH_FRAGMENTS)
    ]
    out.append(f"- **Total files changed:** {len(files)}")
    out.append("")
    out.append("```")
    for ln in files[:MAX_CHANGED_FILES]:
        out.append(ln)
    if len(files) > MAX_CHANGED_FILES:
        out.append(f"... ({len(files) - MAX_CHANGED_FILES} more)")
    out.append("```")
    out.append("")
    return "\n".join(out)


def section_next_question() -> str:
    out = ["## 7. Suggested Next Question for ChatGPT", ""]
    out.append(
        "Given this state, ask ChatGPT to decide whether to proceed with **(X)** "
        "the cheap NS3 lemma-name retrieval pass (1–3 quick wins on the remaining "
        "failures), **(Y)** the NS4+NS5 skeleton-bag + two-tier mutator refactor "
        "(the real v6, ~1–2 weeks), or **(Z)** the NS7 training-data + gen_v5+1 "
        "fine-tune track (parallel research, tests whether the priority_templates "
        "wins generalise at the model level)."
    )
    out.append("")
    out.append(
        "Paste this file into a fresh ChatGPT thread and ask: *\"Given the "
        "headline numbers, the remaining 7 unsolved theorems, and the saturation "
        "evidence in section 3, which of X/Y/Z would you start tomorrow morning, "
        "and what's the single experiment you'd run first to de-risk it?\"*"
    )
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    parts = [
        "# ChatGPT context snapshot",
        "",
        f"_Generated by `scripts/export_context_for_chatgpt.py`. Repo root: `{REPO_ROOT}`._",
        "",
        "Paste this file into ChatGPT to bring it up to speed on the current "
        "state of the Lean AlphaEvolve-style proof-search project, then ask "
        "for next-step advice. Section 7 has a suggested opening question.",
        "",
        section_git_state(),
        section_latest_milestone(),
        section_recent_experiment(),
        section_important_reports(),
        section_open_problems(),
        section_changed_files(),
        section_next_question(),
    ]
    body = "\n".join(parts).rstrip() + "\n"

    # Word-count guardrail (informational only — we don't trim).
    word_count = len(body.split())

    OUTPUT_PATH.write_text(body)

    rel = OUTPUT_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {rel} ({len(body):,} chars, ~{word_count:,} words).")
    if word_count > 5500:
        print(
            "  Note: output exceeds the 5000-word soft target. Consider "
            "pruning long sections in the source reports if this matters."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
