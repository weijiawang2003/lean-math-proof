"""report_run — render a Markdown report from an eval run.

A run directory is one of:
  (a) An `eval_rollout_all.py` output dir containing `eval-<hex>/metrics.json`.
  (b) An `evolve.run_evolve` output dir containing `summary.json` plus
      `eval/<candidate>/eval-<hex>/metrics.json` for one or more candidates.

The script prints (or writes to --output) a self-contained Markdown report
with proved/failed lists, family activations, origin breakdown, runtime,
and an optional baseline comparison.

Usage
-----
  # Single run
  python -m evolve.report_run --hybrid-run project/evolve/runs/<run_id>

  # With baseline
  python -m evolve.report_run \\
      --hybrid-run project/evolve/runs/<run_id> \\
      --baseline-run /tmp/gen_v5_baseline_medium/eval-<hex> \\
      --output project/evolve/reports/nat_defs_medium_v3_6.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _read_metrics(run_dir: Path) -> tuple[dict[str, Any], Path]:
    """Find and load a metrics.json under run_dir.

    Search order:
      1. <run_dir>/metrics.json
      2. <run_dir>/eval-*/metrics.json (eval_rollout_all output)
      3. <run_dir>/eval/seed-baseline/eval-*/metrics.json (run_evolve output)
      4. <run_dir>/eval/*/eval-*/metrics.json (any candidate, newest wins)
    """
    direct = run_dir / "metrics.json"
    if direct.is_file():
        return json.loads(direct.read_text(encoding="utf-8")), direct

    eval_pat = sorted(run_dir.glob("eval-*/metrics.json"))
    if eval_pat:
        p = eval_pat[-1]
        return json.loads(p.read_text(encoding="utf-8")), p

    seed = sorted((run_dir / "eval" / "seed-baseline").glob("eval-*/metrics.json"))
    if seed:
        p = seed[-1]
        return json.loads(p.read_text(encoding="utf-8")), p

    any_cand = sorted(run_dir.glob("eval/*/eval-*/metrics.json"))
    if any_cand:
        p = max(any_cand, key=lambda x: x.stat().st_mtime)
        return json.loads(p.read_text(encoding="utf-8")), p

    raise FileNotFoundError(f"no metrics.json found under {run_dir}")


def _subprocess_log_seconds(metrics_path: Path) -> Optional[int]:
    """Best-effort wallclock from the parent dir's subprocess.log mtime
    delta against its creation time. Returns None when not available."""
    log = metrics_path.parent.parent / "subprocess.log"
    if not log.is_file():
        return None
    try:
        st = log.stat()
        # macOS exposes birthtime; on Linux fall back to ctime.
        start = getattr(st, "st_birthtime", st.st_ctime)
        return int(st.st_mtime - start)
    except OSError:
        return None


def _per_theorem_table(rows: list[dict]) -> str:
    """Render a Markdown table of theorem -> status -> tactic."""
    out = [
        "| Theorem | Status | Steps | Origin | Family | Winning Tactic |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = r.get("full_name", "")
        if r.get("finished"):
            status = "PROVED"
            tac = (r.get("winning_tactic") or "")
            org = r.get("winning_tactic_origin") or ""
            fam = r.get("winning_tactic_family_source") or ""
        elif r.get("has_error"):
            status = "ERROR"
            tac = (r.get("error_message") or "")[:60]
            org = fam = ""
        elif r.get("available"):
            status = "EXHAUSTED"
            tac = ""
            org = fam = ""
        else:
            status = "SKIP"
            tac = (r.get("skip_reason") or "")[:60]
            org = fam = ""
        tac_md = tac.replace("|", "\\|")
        out.append(
            f"| `{name}` | {status} | {r.get('num_steps', 0)} | "
            f"{org} | {fam} | `{tac_md}` |"
        )
    return "\n".join(out)


def _section_origin(m: dict) -> str:
    by_origin = m.get("proved_by_origin") or {}
    if not by_origin:
        return ""
    lines = ["**Proved by origin**", ""]
    for k, v in by_origin.items():
        lines.append(f"- `{k}`: {v}")
    return "\n".join(lines) + "\n"


def _section_families(m: dict) -> str:
    act = m.get("family_activation_counts") or {}
    proved = m.get("family_proved_counts") or {}
    activated = m.get("family_activated_theorems") or {}
    if not act:
        return ""
    lines = ["**Family activations**", ""]
    lines.append("| Family | Activated on | Wins | Theorems |")
    lines.append("|---|---|---|---|")
    for fam in sorted(act.keys()):
        n_act = act[fam]
        n_won = proved.get(fam, 0)
        thms = ", ".join(f"`{t}`" for t in activated.get(fam, []))
        lines.append(f"| `{fam}` | {n_act} | {n_won} | {thms} |")
    return "\n".join(lines) + "\n"


def _section_comparison(hybrid: dict, baseline: dict) -> str:
    """Side-by-side: theorems each side proved, gains, regressions."""
    h_by = {r["full_name"]: r for r in hybrid.get("per_theorem", [])}
    b_by = {r["full_name"]: r for r in baseline.get("per_theorem", [])}
    common = set(h_by) & set(b_by)
    gains = [
        name for name in sorted(common)
        if h_by[name].get("finished") and not b_by[name].get("finished")
    ]
    regressions = [
        name for name in sorted(common)
        if b_by[name].get("finished") and not h_by[name].get("finished")
    ]
    both = [
        name for name in sorted(common)
        if h_by[name].get("finished") and b_by[name].get("finished")
    ]
    lines = ["## Comparison with baseline", ""]
    h_proved = sum(1 for r in hybrid.get("per_theorem", []) if r.get("finished"))
    b_proved = sum(1 for r in baseline.get("per_theorem", []) if r.get("finished"))
    lines.append(f"- **Hybrid**:  {h_proved}/{len(hybrid.get('per_theorem', []))}")
    lines.append(f"- **Baseline**: {b_proved}/{len(baseline.get('per_theorem', []))}")
    lines.append(f"- **Δ = +{h_proved - b_proved}** (gains: {len(gains)}, regressions: {len(regressions)})")
    lines.append("")
    if gains:
        lines.append(f"### Gains over baseline ({len(gains)})")
        lines.append("")
        lines.append("| Theorem | Hybrid origin | Family | Winning tactic |")
        lines.append("|---|---|---|---|")
        for name in gains:
            r = h_by[name]
            tac = (r.get("winning_tactic") or "").replace("|", "\\|")
            lines.append(
                f"| `{name}` | {r.get('winning_tactic_origin','')} | "
                f"{r.get('winning_tactic_family_source','') or ''} | `{tac}` |"
            )
        lines.append("")
    if regressions:
        lines.append(f"### Regressions (baseline won, hybrid lost) ({len(regressions)})")
        lines.append("")
        for name in regressions:
            r = b_by[name]
            tac = (r.get("winning_tactic") or "")
            lines.append(f"- `{name}` (baseline tactic: `{tac}`)")
        lines.append("")
    else:
        lines.append("### Regressions")
        lines.append("")
        lines.append("None — every baseline win is also a hybrid win.")
        lines.append("")
    if both:
        lines.append(f"### Wins on both sides ({len(both)})")
        lines.append("")
        for name in both:
            tac = h_by[name].get("winning_tactic", "")
            lines.append(f"- `{name}` (hybrid: `{tac}`)")
        lines.append("")
    return "\n".join(lines)


def _render_report(
    hybrid_metrics: dict, hybrid_path: Path,
    baseline_metrics: Optional[dict] = None,
    baseline_path: Optional[Path] = None,
) -> str:
    per = hybrid_metrics.get("per_theorem", []) or []
    proved = [r for r in per if r.get("finished")]
    failed = [r for r in per if r.get("available") and not r.get("finished")]
    skipped = [r for r in per if not r.get("available")]

    n = len(per)
    n_proved = len(proved)
    n_avail = sum(1 for r in per if r.get("available"))
    rate = (n_proved / n_avail) if n_avail else 0.0

    wall = _subprocess_log_seconds(hybrid_path)
    per_thm = (wall / n) if (wall and n) else None

    lines = [
        f"# Evaluation report — {hybrid_metrics.get('theorem_set', '?')} / "
        f"{hybrid_metrics.get('policy_type', '?')}",
        "",
        f"**Run id**: `{hybrid_metrics.get('run_id', '?')}`  ",
        f"**Metrics**: `{hybrid_path}`  ",
        f"**Checkpoint**: `{hybrid_metrics.get('ckpt_dir', '?')}`  ",
        f"**Top-k**: {hybrid_metrics.get('top_k', '?')}, "
        f"**Max-steps**: {hybrid_metrics.get('max_steps', '?')}, "
        f"**Decode**: {hybrid_metrics.get('decode_mode', '?')}",
        "",
        "## Summary",
        "",
        f"- **Proved**: **{n_proved}/{n_avail}** "
        f"({rate:.1%}) of {n} theorems"
        f"  (errored {sum(1 for r in per if r.get('has_error'))},"
        f" exhausted {sum(1 for r in per if r.get('available') and not r.get('finished') and not r.get('has_error'))},"
        f" skipped {len(skipped)})",
    ]
    if wall is not None:
        per_thm_str = f"{per_thm:.1f}s" if per_thm is not None else "?"
        lines.append(
            f"- **Wallclock**: {wall}s ({wall // 60}m {wall % 60}s), "
            f"~{per_thm_str}/theorem"
        )
    denied = hybrid_metrics.get("denied_tactic_total")
    if denied:
        lines.append(f"- **Denied tactics filtered**: {denied} (per-theorem deny-list)")
    loop_n = hybrid_metrics.get("loop_transition_count")
    if hybrid_metrics.get("enable_loop_avoidance") or loop_n:
        lines.append(
            f"- **Anti-loop**: enabled={hybrid_metrics.get('enable_loop_avoidance')}, "
            f"loops detected={loop_n}, "
            f"skipped repeats={hybrid_metrics.get('skipped_repeated_tactic_count')}, "
            f"unseen advances={hybrid_metrics.get('unseen_progress_count')}"
        )
    lines.append("")

    org = _section_origin(hybrid_metrics)
    if org:
        lines.append(org)
    fams = _section_families(hybrid_metrics)
    if fams:
        lines.append(fams)

    if baseline_metrics is not None and baseline_path is not None:
        lines.append(_section_comparison(hybrid_metrics, baseline_metrics))

    lines.append("## Proved theorems")
    lines.append("")
    if proved:
        lines.append(_per_theorem_table(proved))
    else:
        lines.append("_(none)_")
    lines.append("")
    lines.append("## Failed theorems")
    lines.append("")
    if failed:
        lines.append(_per_theorem_table(failed))
    else:
        lines.append("_(none)_")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hybrid-run", required=True,
                   help="Path to a run dir (eval_rollout_all output or run_evolve output).")
    p.add_argument("--baseline-run", default=None,
                   help="Optional baseline run dir for side-by-side comparison.")
    p.add_argument("--output", default=None,
                   help="Write Markdown to this path instead of stdout.")
    args = p.parse_args()

    hybrid_dir = Path(args.hybrid_run).resolve()
    h_metrics, h_path = _read_metrics(hybrid_dir)
    b_metrics = b_path = None
    if args.baseline_run:
        b_metrics, b_path = _read_metrics(Path(args.baseline_run).resolve())

    md = _render_report(h_metrics, h_path, b_metrics, b_path)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Report written to {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
