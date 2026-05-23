"""NS6 — assist-credit analysis over per-step traces.

NS5's archive aggregated `wins`, `advances`, and `attempts` per skeleton,
but credited *wins* only to the closing tactic. NS5 Stage-4 confirmed
this missed an important class of skeletons: zero-win skeletons that
advance the goal into a form a *later* tactic closes. Pruning them via
`disable_dead_skeleton` regressed `Nat.div_lt_iff_lt_mul'` 60+ times.

This script computes a credit-aware view per skeleton:

    direct_wins   = skeleton emitted the closing tactic
    advances      = skeleton's tactic produced an advance (new state,
                    not a close)
    assist_wins@K = skeleton advanced, and within the next K accepted
                    proof steps a *different* tactic closed the proof

Input: one or more `traces.jsonl` files written by `eval_rollout_all.py`
       *after* the NS6 patch that records `skeleton_name` per step.

Output: a per-skeleton table plus three derived lists:
    - top assist-credit zero-win skeletons (must-protect)
    - truly dead (no wins / advances / assists)
    - protected vs. prunable summary

Usage:
    python scripts/ns6_assist_credit.py \
        --traces project/evolve/skeleton_runs/<run>/eval/<eval>/traces.jsonl \
        [--traces ...] \
        --k 1,2,3 \
        --out project/evolve/reports/ns6_assist_credit_analysis.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------- helpers
def _is_close(r: dict[str, Any]) -> bool:
    return bool(r.get("proof_finished"))


def _is_advance(r: dict[str, Any]) -> bool:
    """Returns True iff this trace row is the accepted *advance* on its step.

    An advance row has `state_hash_after` set, was not a Lean error,
    was not skipped, and was not a loop-detected deferred entry that the
    inner loop continued past.
    """
    if r.get("proof_finished"):
        return False
    kind = r.get("result_kind") or ""
    if kind == "LeanError":
        return False
    if kind in {"SkippedBloatingApply", "SkippedKnownError"}:
        return False
    if r.get("loop_detected"):
        return False
    if r.get("bloat_rejected"):
        return False
    return r.get("state_hash_after") is not None


def _step_key(r: dict[str, Any]) -> tuple[str, int]:
    return (str(r.get("episode_id") or ""), int(r.get("step") or 0))


_ORIGIN_PREFIX = (
    ("pt_", "priority_template"),
    ("fam_", "family_tactic"),
    ("tb_", "term_builder"),
    ("fb_", "fallback_tactic"),
    ("tt_", "tactic_template"),
    ("retrieved:", "retrieved_premise"),
    ("gen_", "generative_topk"),
)


def _origin_from_name(name: str | None) -> str | None:
    """The wrapper records `tactic_origin` per emit-path, which conflates
    priority_template with tactic_template. The skeleton_name prefix is
    the authoritative source for the bag's own notion of origin."""
    if not name:
        return None
    for prefix, origin in _ORIGIN_PREFIX:
        if name.startswith(prefix):
            return origin
    return None


# ---------------------------------------------------------------------- core
@dataclass
class CreditStats:
    skeleton_name: str
    skeleton_shape: str | None = None
    skeleton_family: str | None = None
    origin: str | None = None
    attempts: int = 0          # any rank where this skeleton was tried
    direct_wins: int = 0       # rows with proof_finished=True
    advances: int = 0          # rows that produced an advance
    assist_wins_k1: int = 0
    assist_wins_k2: int = 0
    assist_wins_k3: int = 0
    theorems_assisted: set[str] = field(default_factory=set)
    theorems_won: set[str] = field(default_factory=set)
    theorems_advanced: set[str] = field(default_factory=set)

    def total_credit(self) -> int:
        return self.direct_wins + self.assist_wins_k3

    def to_row(self) -> dict[str, Any]:
        return {
            "skeleton_name": self.skeleton_name,
            "shape": self.skeleton_shape,
            "family": self.skeleton_family,
            "origin": self.origin,
            "attempts": self.attempts,
            "direct_wins": self.direct_wins,
            "advances": self.advances,
            "assist_wins_k1": self.assist_wins_k1,
            "assist_wins_k2": self.assist_wins_k2,
            "assist_wins_k3": self.assist_wins_k3,
            "theorems_won": sorted(self.theorems_won),
            "theorems_assisted_k3": sorted(self.theorems_assisted),
            "theorems_advanced": sorted(self.theorems_advanced),
        }


def load_traces(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def compute_credit(
    rows: Iterable[dict[str, Any]],
    ks: tuple[int, ...] = (1, 2, 3),
) -> dict[str, CreditStats]:
    """Per-skeleton credit accounting.

    Walk traces grouped by episode. Per episode, build the ordered list
    of *accepted* step rows (one per `step` index, in order). For each
    accepted advance row whose skeleton emitted, look forward at the
    next K accepted rows: if any is a close, credit assist@K to that
    skeleton.
    """
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        eid = r.get("episode_id")
        if not eid:
            continue
        by_episode[eid].append(r)

    stats: dict[str, CreditStats] = {}

    def _get(name: str, r: dict[str, Any]) -> CreditStats:
        st = stats.get(name)
        if st is None:
            st = CreditStats(
                skeleton_name=name,
                skeleton_shape=r.get("skeleton_shape"),
                skeleton_family=r.get("skeleton_family"),
                origin=_origin_from_name(name),
            )
            stats[name] = st
        if st.skeleton_shape is None and r.get("skeleton_shape"):
            st.skeleton_shape = r["skeleton_shape"]
        if st.skeleton_family is None and r.get("skeleton_family"):
            st.skeleton_family = r["skeleton_family"]
        if st.origin is None:
            st.origin = _origin_from_name(name)
        return st

    # First: count *attempts*. Every row whose `skeleton_name` is set
    # counts as an attempt, regardless of whether it was the accepted
    # advance/close on its step.
    for rows_e in by_episode.values():
        for r in rows_e:
            name = r.get("skeleton_name")
            if name:
                _get(name, r).attempts += 1

    # Second: per episode, build the accepted-step sequence and credit
    # closes/advances/assists.
    for eid, rows_e in by_episode.items():
        # Group by step, pick the accepted row (close > advance > none).
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows_e:
            try:
                s = int(r.get("step"))
            except (TypeError, ValueError):
                continue
            by_step[s].append(r)

        accepted: list[tuple[int, dict[str, Any], str]] = []  # (step, row, role)
        for s in sorted(by_step.keys()):
            close = next((r for r in by_step[s] if _is_close(r)), None)
            if close is not None:
                accepted.append((s, close, "close"))
                continue
            adv = next((r for r in by_step[s] if _is_advance(r)), None)
            if adv is not None:
                accepted.append((s, adv, "advance"))
        # Direct credit per accepted row.
        for _, r, role in accepted:
            name = r.get("skeleton_name")
            thm = r.get("full_name")
            if not name:
                continue
            st = _get(name, r)
            if role == "close":
                st.direct_wins += 1
                if thm:
                    st.theorems_won.add(thm)
            elif role == "advance":
                st.advances += 1
                if thm:
                    st.theorems_advanced.add(thm)

        # Assist credit: for each advance-row, look ahead up to max(ks)
        # accepted rows for a close. If found at distance d, credit
        # assist@K for every K in ks where K >= d.
        max_k = max(ks)
        for i, (_, r, role) in enumerate(accepted):
            if role != "advance":
                continue
            name = r.get("skeleton_name")
            if not name:
                continue
            thm = r.get("full_name")
            # Closing-tactic skeleton must be different from the assister
            # only if explicitly different; same name still gets credit
            # as long as it's a *separate* later tactic. The spec says
            # "closed by another tactic" — we interpret tactic as the
            # specific tac string; same skeleton emitting a different
            # closing tactic still counts as assist.
            for j in range(i + 1, min(len(accepted), i + 1 + max_k)):
                step_idx, fr, frole = accepted[j]
                if frole != "close":
                    continue
                d = j - i  # distance in accepted-step count
                close_tac = fr.get("tactic")
                advance_tac = r.get("tactic")
                if close_tac is not None and close_tac == advance_tac:
                    # Same exact tactic doesn't count as assist — would
                    # be the same step type / no real handoff.
                    continue
                st = _get(name, r)
                if d <= 1 and 1 in ks:
                    st.assist_wins_k1 += 1
                if d <= 2 and 2 in ks:
                    st.assist_wins_k2 += 1
                if d <= 3 and 3 in ks:
                    st.assist_wins_k3 += 1
                if thm:
                    st.theorems_assisted.add(thm)
                break  # first close in window — single assist credit per advance
    return stats


# ---------------------------------------------------------------------- report
def _fmt_row(st: CreditStats) -> str:
    return (
        f"| {st.skeleton_name} | {st.skeleton_shape or '-'} | "
        f"{st.skeleton_family or '-'} | {st.origin or '-'} | "
        f"{st.attempts} | {st.direct_wins} | {st.advances} | "
        f"{st.assist_wins_k1} | {st.assist_wins_k2} | {st.assist_wins_k3} |"
    )


def render_report(
    stats: dict[str, CreditStats],
    sources: list[Path],
    dead_threshold: int = 5,
) -> str:
    rows = sorted(
        stats.values(),
        key=lambda s: (-s.direct_wins, -s.assist_wins_k3, -s.advances, s.skeleton_name),
    )
    lines: list[str] = []
    lines.append("# NS6 — assist-credit analysis\n")
    lines.append("Per-skeleton credit accounting over the per-step traces below.")
    lines.append("`direct_wins` = closed the proof; `advances` = produced a new")
    lines.append("state without closing; `assist_wins_kN` = advanced, and within")
    lines.append("the next N accepted proof steps a *different* tactic closed the")
    lines.append("proof.\n")
    lines.append("## Sources\n")
    for p in sources:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append(f"- skeletons observed: **{len(rows)}**")
    total_direct = sum(s.direct_wins for s in rows)
    total_assist3 = sum(s.assist_wins_k3 for s in rows)
    total_adv = sum(s.advances for s in rows)
    lines.append(f"- total direct wins: **{total_direct}**")
    lines.append(f"- total advances: **{total_adv}**")
    lines.append(f"- total assist@k3: **{total_assist3}**")
    lines.append("")

    lines.append("## Per-skeleton credit table\n")
    lines.append("| skeleton | shape | family | origin | attempts | direct_wins | advances | assist@1 | assist@2 | assist@3 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for st in rows:
        lines.append(_fmt_row(st))
    lines.append("")

    zero_win_assist = [s for s in rows if s.direct_wins == 0 and s.assist_wins_k3 > 0]
    zero_win_assist.sort(key=lambda s: (-s.assist_wins_k3, -s.assist_wins_k2, s.skeleton_name))
    lines.append("## Zero-win skeletons with assist credit (MUST-PROTECT)\n")
    if not zero_win_assist:
        lines.append("_None observed — no zero-win skeleton produced an assist within K≤3._\n")
    else:
        lines.append("These skeletons never closed a proof but advanced state into a form a")
        lines.append("later tactic closed within K≤3 steps. NS5's wins-only `disable_dead_skeleton`")
        lines.append("would prune them — NS6's safe pruning rule must protect them.\n")
        lines.append("| skeleton | shape | origin | advances | assist@1 | assist@2 | assist@3 | assisted theorems |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for st in zero_win_assist:
            thms = ", ".join(sorted(st.theorems_assisted)[:5])
            if len(st.theorems_assisted) > 5:
                thms += f", … (+{len(st.theorems_assisted) - 5} more)"
            lines.append(
                f"| {st.skeleton_name} | {st.skeleton_shape or '-'} | {st.origin or '-'} | "
                f"{st.advances} | {st.assist_wins_k1} | {st.assist_wins_k2} | "
                f"{st.assist_wins_k3} | {thms} |"
            )
        lines.append("")

    truly_dead = [
        s for s in rows
        if s.direct_wins == 0 and s.advances == 0 and s.assist_wins_k3 == 0
        and s.attempts >= dead_threshold
    ]
    truly_dead.sort(key=lambda s: (-s.attempts, s.skeleton_name))
    lines.append("## Truly dead skeletons (safe to prune)\n")
    lines.append(f"`attempts >= {dead_threshold}` AND `direct_wins = advances = assist_wins_k3 = 0`.\n")
    if not truly_dead:
        lines.append("_None._\n")
    else:
        lines.append("| skeleton | shape | origin | attempts |")
        lines.append("|---|---|---|---:|")
        for st in truly_dead:
            lines.append(
                f"| {st.skeleton_name} | {st.skeleton_shape or '-'} | "
                f"{st.origin or '-'} | {st.attempts} |"
            )
        lines.append("")

    # Summary table of protection categories.
    n_direct_win = sum(1 for s in rows if s.direct_wins > 0)
    n_assist_only = len(zero_win_assist)
    n_advance_only = sum(
        1 for s in rows
        if s.direct_wins == 0 and s.assist_wins_k3 == 0 and s.advances > 0
    )
    n_truly_dead = len(truly_dead)
    n_attempt_only = sum(
        1 for s in rows
        if s.direct_wins == 0 and s.advances == 0 and s.assist_wins_k3 == 0
        and s.attempts > 0 and s.attempts < dead_threshold
    )
    lines.append("## Protection summary\n")
    lines.append("| category | count |")
    lines.append("|---|---:|")
    lines.append(f"| direct-win skeletons (protected) | {n_direct_win} |")
    lines.append(f"| zero-win assist@3 skeletons (must-protect) | {n_assist_only} |")
    lines.append(f"| advance-only skeletons (review) | {n_advance_only} |")
    lines.append(f"| low-attempt skeletons (insufficient signal) | {n_attempt_only} |")
    lines.append(f"| truly dead (attempts≥{dead_threshold}, no signal) | {n_truly_dead} |")
    lines.append("")
    return "\n".join(lines)


def write_index(
    stats: dict[str, CreditStats],
    out_path: Path,
) -> None:
    payload = {
        "skeleton_count": len(stats),
        "skeletons": [s.to_row() for s in stats.values()],
    }
    # serialize sets as lists
    for sk in payload["skeletons"]:
        for k in ("theorems_won", "theorems_assisted_k3", "theorems_advanced"):
            v = sk.get(k)
            if isinstance(v, set):
                sk[k] = sorted(v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", action="append", required=True,
                    help="path to traces.jsonl (repeatable)")
    ap.add_argument("--k", default="1,2,3",
                    help="comma-separated assist-window sizes; default 1,2,3")
    ap.add_argument("--out", type=Path, default=None,
                    help="write markdown report to this path")
    ap.add_argument("--index", type=Path, default=None,
                    help="write JSON credit-index to this path")
    ap.add_argument("--dead-threshold", type=int, default=5,
                    help="attempts threshold for `truly dead` classification")
    args = ap.parse_args()

    paths = [Path(p) for p in args.traces]
    rows = load_traces(paths)
    if not rows:
        print(f"no rows loaded from {paths}")
        return
    print(f"loaded {len(rows)} trace rows from {len(paths)} file(s)")
    ks = tuple(int(x) for x in args.k.split(",") if x.strip())
    stats = compute_credit(rows, ks=ks)
    n_with_skel = sum(1 for r in rows if r.get("skeleton_name"))
    print(f"  rows with skeleton_name: {n_with_skel}")
    print(f"  distinct skeletons observed: {len(stats)}")
    report = render_report(stats, paths, dead_threshold=args.dead_threshold)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote report to {args.out}")
    else:
        print(report)
    if args.index:
        write_index(stats, args.index)
        print(f"wrote index to {args.index}")


if __name__ == "__main__":
    main()
