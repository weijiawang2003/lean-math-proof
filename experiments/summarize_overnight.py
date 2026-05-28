"""Aggregate metrics from an overnight A+D run into SUMMARY.md.

Reads metrics.json from every run dir under <root>, groups by checkpoint
tag (v5/premise/base) and decode mode (beam vs. sample), computes mean ±
std across sampling seeds, and writes a markdown report.

Also includes the retriever-quality probe results when present.

Run:
    python experiments/summarize_overnight.py \
        --root experiments/overnight_<TIMESTAMP> \
        --out  experiments/overnight_<TIMESTAMP>/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


CKPT_LABELS = {
    "v5":      "gen_v5  (t5-small, baseline)",
    "premise": "gen_v6_premise (t5-small + premise injection)",
    "base":    "gen_v6  (t5-base, never previously evaluated)",
}
CKPT_ORDER = ["v5", "premise", "base"]


def _load_metrics(run_dir: Path) -> dict | None:
    """Find the single eval-*/metrics.json under a phase-output directory."""
    candidates = sorted(run_dir.glob("eval-*/metrics.json"))
    if not candidates:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _parse_dirname(name: str) -> tuple[str, str, int | None] | None:
    """Map dir name -> (tag, mode, seed_or_None) or None if not a run dir."""
    m = re.match(r"^(v5|premise|base)_beam$", name)
    if m:
        return m.group(1), "beam", None
    m = re.match(r"^(v5|premise|base)_sample_seed(\d+)$", name)
    if m:
        return m.group(1), "sample", int(m.group(2))
    return None


def _stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description="Summarize an overnight A+D run.")
    parser.add_argument("--root", required=True,
                        help="Path to experiments/overnight_<TIMESTAMP>")
    parser.add_argument("--out", required=True,
                        help="Path to write SUMMARY.md")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    # ---- Walk run dirs ----
    # Per-tag bookkeeping: beam_proved (single int) and sample_proved (list of ints)
    beam_proved: dict[str, int | None] = {}
    beam_avail: dict[str, int | None] = {}
    sample_proved: dict[str, list[int]] = defaultdict(list)
    sample_avail: dict[str, list[int]] = defaultdict(list)
    sample_seeds: dict[str, list[int]] = defaultdict(list)
    failed_runs: list[str] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_dirname(child.name)
        if parsed is None:
            continue
        tag, mode, seed = parsed
        m = _load_metrics(child)
        if m is None:
            failed_runs.append(f"{child.name} (no metrics.json — likely crashed)")
            continue
        proved = int(m.get("proved") or 0)
        avail = int(m.get("available") or 0)
        if mode == "beam":
            beam_proved[tag] = proved
            beam_avail[tag] = avail
        else:
            sample_proved[tag].append(proved)
            sample_avail[tag].append(avail)
            if seed is not None:
                sample_seeds[tag].append(seed)

    # ---- Retriever probe ----
    probe_path = root / "retriever_probe.json"
    probe = None
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failed_runs.append(f"retriever_probe.json (parse error: {exc})")

    # ---- Render markdown ----
    lines: list[str] = []
    lines.append(f"# Overnight A+D — Summary")
    lines.append("")
    lines.append(f"Run directory: `{root}`")
    lines.append("")
    lines.append("## Curriculum eval (curriculum_all, 30 theorems)")
    lines.append("")
    lines.append("Headline: deterministic beam-k=8 anchor + sampling variance "
                 "(temp=0.8, top-p=0.95, k=8) over 5 seeds.")
    lines.append("")
    lines.append("| Checkpoint | Beam (anchor) | Sample mean ± std | Sample range | N seeds |")
    lines.append("|---|---|---|---|---|")
    for tag in CKPT_ORDER:
        label = CKPT_LABELS[tag]
        bp = beam_proved.get(tag)
        ba = beam_avail.get(tag)
        beam_cell = f"{bp}/{ba}" if bp is not None else "—"
        samples = sample_proved.get(tag, [])
        n_seeds = len(samples)
        if samples:
            mean, std = _stats([float(x) for x in samples])
            sample_cell = f"{mean:.1f} ± {std:.1f}"
            range_cell = f"{min(samples)} – {max(samples)}"
        else:
            sample_cell = "—"
            range_cell = "—"
        lines.append(f"| {label} | {beam_cell} | {sample_cell} | {range_cell} | {n_seeds} |")
    lines.append("")

    # Per-seed table for transparency
    lines.append("### Per-seed sample results")
    lines.append("")
    lines.append("| Checkpoint | " + " | ".join(f"seed {s}" for s in sorted(set(
        s for ss in sample_seeds.values() for s in ss
    ))) + " |")
    seed_headers = sorted(set(s for ss in sample_seeds.values() for s in ss))
    if seed_headers:
        lines.append("|" + "---|" * (len(seed_headers) + 1))
        for tag in CKPT_ORDER:
            row = [CKPT_LABELS[tag]]
            seeds_for_tag = sample_seeds.get(tag, [])
            results_for_tag = sample_proved.get(tag, [])
            seed_to_result = dict(zip(seeds_for_tag, results_for_tag))
            for s in seed_headers:
                row.append(str(seed_to_result.get(s, "—")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- Headline interpretations ----
    lines.append("## Reading the result")
    lines.append("")
    bp_v5 = beam_proved.get("v5")
    bp_pr = beam_proved.get("premise")
    bp_bs = beam_proved.get("base")

    if bp_v5 is not None and bp_bs is not None:
        delta = bp_bs - bp_v5
        sign = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(f"- **Capacity scaling (t5-small → t5-base):** "
                     f"{bp_v5} → {bp_bs} (Δ = {sign} {abs(delta)} theorems on the curriculum, beam).")
    if bp_v5 is not None and bp_pr is not None:
        delta = bp_pr - bp_v5
        sign = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(f"- **Premise injection at t5-small:** "
                     f"{bp_v5} → {bp_pr} (Δ = {sign} {abs(delta)}, beam). "
                     f"Reproducing the v5/v6_premise contrast.")

    # Sampling-vs-beam interpretation
    samples_v5 = sample_proved.get("v5", [])
    samples_pr = sample_proved.get("premise", [])
    samples_bs = sample_proved.get("base", [])
    if samples_v5 and samples_pr:
        m_v5, s_v5 = _stats([float(x) for x in samples_v5])
        m_pr, s_pr = _stats([float(x) for x in samples_pr])
        # Two-sample comparison: gap robust if |m1-m2| > 2*sqrt(s1^2+s2^2)
        pooled = math.sqrt(s_v5 ** 2 + s_pr ** 2)
        gap = m_v5 - m_pr
        robustness = ("ROBUST"
                      if pooled == 0 or abs(gap) > 2 * pooled
                      else "NOT robust within sampling noise")
        lines.append(f"- **v5 vs premise gap under sampling:** "
                     f"{m_v5:.1f} ± {s_v5:.1f}  vs  {m_pr:.1f} ± {s_pr:.1f}.  "
                     f"Gap = {gap:+.1f}, pooled sd = {pooled:.2f}.  → {robustness}.")
    if samples_v5 and samples_bs:
        m_v5, s_v5 = _stats([float(x) for x in samples_v5])
        m_bs, s_bs = _stats([float(x) for x in samples_bs])
        pooled = math.sqrt(s_v5 ** 2 + s_bs ** 2)
        gap = m_bs - m_v5
        robustness = ("ROBUST"
                      if pooled == 0 or abs(gap) > 2 * pooled
                      else "NOT robust within sampling noise")
        lines.append(f"- **t5-small vs t5-base gap under sampling:** "
                     f"{m_v5:.1f} ± {s_v5:.1f}  vs  {m_bs:.1f} ± {s_bs:.1f}.  "
                     f"Gap = {gap:+.1f}, pooled sd = {pooled:.2f}.  → {robustness}.")

    lines.append("")

    # ---- Retriever probe section ----
    lines.append("## Retriever quality probe (D)")
    lines.append("")
    if probe is None:
        lines.append("_No probe results found at_ `retriever_probe.json` _._")
    else:
        counts = probe.get("counts", {})
        lines.append(f"- Total proven theorems in `project_state.json`: "
                     f"**{counts.get('total_proved', 0)}**")
        lines.append(f"- Evaluable (named premises + first-state in traces): "
                     f"**{counts.get('evaluated', 0)}**")
        lines.append(f"- Skipped — tactic had no named premises (e.g. plain "
                     f"`aesop`): {counts.get('skipped_no_named_premises', 0)}")
        lines.append(f"- Skipped — no first-state in traces: "
                     f"{counts.get('skipped_no_state_in_traces', 0)}")
        lines.append("")
        summary = probe.get("summary_by_bucket", {})
        if summary:
            lines.append("| File bucket | N | Recall@1 | Recall@5 | Recall@10 | Recall@15 |")
            lines.append("|---|---|---|---|---|---|")
            order = ["Set.Basic", "Finset.Basic", "Nat.Defs", "Nat.Basic", "other", "__OVERALL__"]
            for bucket in order:
                if bucket not in summary:
                    continue
                agg = summary[bucket]
                pretty = "**OVERALL**" if bucket == "__OVERALL__" else bucket
                lines.append(f"| {pretty} | {agg.get('n_theorems', 0)} | "
                             f"{agg.get('mean_recall@1', 0):.1%} | "
                             f"{agg.get('mean_recall@5', 0):.1%} | "
                             f"{agg.get('mean_recall@10', 0):.1%} | "
                             f"{agg.get('mean_recall@15', 0):.1%} |")
        lines.append("")
        lines.append("**Reading this:** Recall@k = of the lemma names that actually "
                     "appeared in the winning tactic, what fraction did the retriever "
                     "place in its top-k?  Low Recall@5 with high Recall@15 means the "
                     "retriever is finding the right premise but ranking it poorly. "
                     "Low Recall@15 means the premise isn't in the index at all, which "
                     "is the bottleneck regardless of model size.")
        lines.append("")

    # ---- Failures ----
    if failed_runs:
        lines.append("## Run failures (these did NOT produce metrics)")
        lines.append("")
        for name in failed_runs:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.append("## Raw artifacts")
    lines.append("")
    lines.append(f"- Per-run metrics: `{root}/<tag>_<mode>_<seed>/eval-*/metrics.json`")
    lines.append(f"- Run log: `{root}/run.log`")
    lines.append(f"- Retriever probe JSON: `{root}/retriever_probe.json`")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summarize] Wrote {out}")
    print()
    # Console preview of the headline table
    for line in lines[:30]:
        print(line)


if __name__ == "__main__":
    main()
