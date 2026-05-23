"""NS5 skeleton archive.

A JSONL file at `project/evolve/archive/skeletons.jsonl` that records
per-skeleton performance across evolution runs. Each row is the record
of a single skeleton observed on a single theorem within a single eval
run:

    {
      "skeleton_name": "pt_iff_8",
      "skeleton_shape": "iff",
      "skeleton_family": null,
      "skeleton_specificity": 1,
      "skeleton_priority": 0,
      "origin": "priority_template",
      "template": "exact ⟨fun h => by omega, fun h => by omega⟩",
      "theorem": "Nat.lt_succ_iff",
      "theorem_set": "nat_defs_medium",
      "result_kind": "proved",         # proved / advanced / attempted
      "tactic": "exact ⟨fun h => by omega, fun h => by omega⟩",
      "run_id": "ns5-20260522-234500-abcdef",
      "first_seen_run": "ns5-20260522-...",
      "last_seen_commit": "4a61ea1",
      "metadata": {...}
    }

The archive is append-only at the row level; aggregation
(`top_skeletons_by_wins`, `dead_skeletons`, …) is recomputed each time
from the row stream.

We also write a compact derived index at
`project/evolve/archive/skeletons_index.json` after every archive
update, so the runner can do `O(1)` lookups for `dead`/`top` without
re-parsing the JSONL.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_PATH = REPO_ROOT / "project/evolve/archive/skeletons.jsonl"
DEFAULT_INDEX_PATH = REPO_ROOT / "project/evolve/archive/skeletons_index.json"

# A skeleton is "dead" when it has been *attempted* this many times
# across the archive without producing a single win.
DEFAULT_DEAD_ATTEMPT_THRESHOLD = 10

# Origins that should NEVER be auto-disabled by `dead_skeletons` — they
# are dynamic or have known side effects.
PROTECTED_ORIGINS = {"retrieved_premise", "generative_topk"}


# ---------------------------------------------------------------------- aggregator
@dataclass
class SkeletonStats:
    """Aggregated stats for one skeleton_name across the whole archive."""

    skeleton_name: str
    skeleton_shape: str | None = None
    skeleton_family: str | None = None
    skeleton_specificity: int | None = None
    skeleton_priority: int | None = None
    origin: str | None = None
    template: str | None = None
    wins: int = 0
    advances: int = 0
    attempts: int = 0
    theorems_won: list[str] = field(default_factory=list)
    theorems_advanced: list[str] = field(default_factory=list)
    theorem_sets: list[str] = field(default_factory=list)
    first_seen_run: str | None = None
    last_seen_run: str | None = None
    last_seen_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton_name": self.skeleton_name,
            "skeleton_shape": self.skeleton_shape,
            "skeleton_family": self.skeleton_family,
            "skeleton_specificity": self.skeleton_specificity,
            "skeleton_priority": self.skeleton_priority,
            "origin": self.origin,
            "template": self.template,
            "wins": self.wins,
            "advances": self.advances,
            "attempts": self.attempts,
            "theorems_won": list(self.theorems_won),
            "theorems_advanced": list(self.theorems_advanced),
            "theorem_sets": list(self.theorem_sets),
            "first_seen_run": self.first_seen_run,
            "last_seen_run": self.last_seen_run,
            "last_seen_commit": self.last_seen_commit,
        }


# ---------------------------------------------------------------------- IO
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_rows(
    rows: Iterable[dict[str, Any]],
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
) -> int:
    """Append rows to the JSONL archive. Returns the number written."""
    path = Path(archive_path)
    _ensure_dir(path)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_archive(
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
) -> list[dict[str, Any]]:
    """Load the JSONL archive into a list of dicts. Returns [] if absent."""
    path = Path(archive_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupt lines but do not crash the runner.
                continue
    return rows


def aggregate(
    rows: Iterable[dict[str, Any]],
) -> dict[str, SkeletonStats]:
    """Aggregate per-skeleton stats from raw rows."""
    out: dict[str, SkeletonStats] = {}
    for r in rows:
        name = r.get("skeleton_name")
        if not name:
            continue
        st = out.get(name)
        if st is None:
            st = SkeletonStats(
                skeleton_name=name,
                skeleton_shape=r.get("skeleton_shape"),
                skeleton_family=r.get("skeleton_family"),
                skeleton_specificity=r.get("skeleton_specificity"),
                skeleton_priority=r.get("skeleton_priority"),
                origin=r.get("origin"),
                template=r.get("template"),
                first_seen_run=r.get("run_id"),
            )
            out[name] = st
        # Always update last-seen / mutable identity fields. Shape and
        # family may legitimately drift across mutator runs; we keep the
        # most recent observation.
        st.last_seen_run = r.get("run_id") or st.last_seen_run
        st.last_seen_commit = r.get("last_seen_commit") or st.last_seen_commit
        if r.get("skeleton_shape") is not None:
            st.skeleton_shape = r["skeleton_shape"]
        if r.get("skeleton_family") is not None:
            st.skeleton_family = r["skeleton_family"]
        if r.get("skeleton_specificity") is not None:
            st.skeleton_specificity = r["skeleton_specificity"]
        if r.get("skeleton_priority") is not None:
            st.skeleton_priority = r["skeleton_priority"]
        if r.get("origin") is not None:
            st.origin = r["origin"]
        if r.get("template") is not None:
            st.template = r["template"]
        kind = (r.get("result_kind") or "").lower()
        thm = r.get("theorem")
        tset = r.get("theorem_set")
        if tset and tset not in st.theorem_sets:
            st.theorem_sets.append(tset)
        if kind == "proved":
            st.wins += 1
            if thm and thm not in st.theorems_won:
                st.theorems_won.append(thm)
            # Wins imply advance + attempt.
            st.advances += 1
            st.attempts += 1
            if thm and thm not in st.theorems_advanced:
                st.theorems_advanced.append(thm)
        elif kind == "advanced":
            st.advances += 1
            st.attempts += 1
            if thm and thm not in st.theorems_advanced:
                st.theorems_advanced.append(thm)
        elif kind == "attempted":
            st.attempts += 1
        # Other kinds (regressed, etc.) are recorded but not summed.
    return out


def write_index(
    stats: dict[str, SkeletonStats],
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> None:
    path = Path(index_path)
    _ensure_dir(path)
    payload = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "skeleton_count": len(stats),
        "skeletons": [s.to_dict() for s in stats.values()],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------- query
def top_skeletons_by_wins(
    stats: dict[str, SkeletonStats],
    n: int = 20,
    origin: str | None = None,
    shape: str | None = None,
) -> list[SkeletonStats]:
    """Return the n best-winning skeletons, optionally filtered."""
    out = list(stats.values())
    if origin is not None:
        out = [s for s in out if s.origin == origin]
    if shape is not None:
        out = [s for s in out if s.skeleton_shape == shape]
    out.sort(key=lambda s: (-s.wins, -s.advances, s.attempts, s.skeleton_name))
    return out[:n]


def dead_skeletons(
    stats: dict[str, SkeletonStats],
    min_attempts: int = DEFAULT_DEAD_ATTEMPT_THRESHOLD,
    exclude_origins: Iterable[str] = PROTECTED_ORIGINS,
) -> list[SkeletonStats]:
    """Return skeletons with 0 wins and at least `min_attempts` attempts."""
    excl = set(exclude_origins)
    out = [
        s for s in stats.values()
        if s.wins == 0
        and s.attempts >= min_attempts
        and (s.origin not in excl)
    ]
    out.sort(key=lambda s: (-s.attempts, s.skeleton_name))
    return out


def skeletons_by_shape(
    stats: dict[str, SkeletonStats],
) -> dict[str, list[SkeletonStats]]:
    out: dict[str, list[SkeletonStats]] = defaultdict(list)
    for s in stats.values():
        out[s.skeleton_shape or "unknown"].append(s)
    for k in out:
        out[k].sort(key=lambda s: (-s.wins, -s.advances))
    return dict(out)


def skeletons_by_family(
    stats: dict[str, SkeletonStats],
) -> dict[str, list[SkeletonStats]]:
    out: dict[str, list[SkeletonStats]] = defaultdict(list)
    for s in stats.values():
        out[s.skeleton_family or "_none_"].append(s)
    for k in out:
        out[k].sort(key=lambda s: (-s.wins, -s.advances))
    return dict(out)


# ---------------------------------------------------------------------- update from a run
def _rows_from_metrics(
    metrics: dict[str, Any],
    run_id: str,
    last_seen_commit: str | None,
) -> list[dict[str, Any]]:
    """Synthesize archive rows from a metrics.json blob.

    For each per-theorem result `r`, emit:
      - one `proved` row if `r.winning_tactic_skeleton_name` is set,
      - one `advanced` row for every skeleton in `r.skeletons_seen` that
        is not the winner (no per-skeleton attempt count yet, so we
        coarse-grain by `result_kind="advanced"` if the skeleton appeared,
        otherwise nothing). When `skeleton_attempt_count > 0` but
        `skeletons_seen` is empty we degrade gracefully and emit an
        `attempted` row keyed by a synthetic name.

    The metrics.json schema is what NS4.1 / NS4.2 produces.
    """
    theorem_set = metrics.get("theorem_set") or "unknown"
    per = metrics.get("per_theorem") or []
    skeleton_wins = metrics.get("skeleton_wins") or []
    win_template = {
        (w.get("theorem"), w.get("skeleton_name")): w.get("tactic")
        for w in skeleton_wins
    }
    rows: list[dict[str, Any]] = []
    for r in per:
        thm = r.get("full_name")
        win_name = r.get("winning_tactic_skeleton_name")
        seen = list(r.get("skeletons_seen") or [])
        win_tactic = r.get("winning_tactic")
        # PROVED row for the winner.
        if win_name:
            rows.append({
                "skeleton_name": win_name,
                "skeleton_shape": r.get("winning_tactic_skeleton_shape"),
                "skeleton_family": r.get("winning_tactic_skeleton_family"),
                "skeleton_specificity": r.get(
                    "winning_tactic_skeleton_specificity"
                ),
                "skeleton_priority": r.get(
                    "winning_tactic_skeleton_priority"
                ),
                "origin": r.get("winning_tactic_origin"),
                "template": win_template.get(
                    (thm, win_name), win_tactic
                ),
                "theorem": thm,
                "theorem_set": theorem_set,
                "result_kind": "proved",
                "tactic": win_tactic,
                "run_id": run_id,
                "last_seen_commit": last_seen_commit,
            })
        # ADVANCED rows for every other skeleton attempted on this theorem.
        # We don't have per-skeleton advanced flags in metrics.json (only
        # totals), so we conservatively log every seen skeleton as
        # "attempted". If it advanced or proved the runner would have a
        # per-step trace; without that we cannot distinguish, but
        # `attempted` is the correct safe label.
        for name in seen:
            if name == win_name:
                continue
            rows.append({
                "skeleton_name": name,
                "skeleton_shape": None,
                "skeleton_family": None,
                "skeleton_specificity": None,
                "skeleton_priority": None,
                "origin": None,
                "template": None,
                "theorem": thm,
                "theorem_set": theorem_set,
                "result_kind": "attempted",
                "tactic": None,
                "run_id": run_id,
                "last_seen_commit": last_seen_commit,
            })
    return rows


def update_archive_from_metrics(
    metrics: dict[str, Any],
    run_id: str,
    last_seen_commit: str | None = None,
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, int]:
    """Append rows derived from a metrics.json blob.

    Returns `{"rows_appended": N, "skeleton_count_after": K}`.
    """
    rows = _rows_from_metrics(metrics, run_id, last_seen_commit)
    n = append_rows(rows, archive_path)
    all_rows = load_archive(archive_path)
    stats = aggregate(all_rows)
    write_index(stats, index_path)
    return {
        "rows_appended": n,
        "rows_total": len(all_rows),
        "skeleton_count_after": len(stats),
    }


def update_archive_from_metrics_path(
    metrics_path: Path | str,
    run_id: str | None = None,
    last_seen_commit: str | None = None,
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, int]:
    p = Path(metrics_path)
    metrics = json.loads(p.read_text(encoding="utf-8"))
    rid = run_id or metrics.get("run_id") or p.parent.name
    return update_archive_from_metrics(
        metrics, rid, last_seen_commit, archive_path, index_path,
    )


def update_archive_from_run(
    run_dir: Path | str,
    theorem_set: str | None = None,
    last_seen_commit: str | None = None,
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> dict[str, int]:
    """Scan `run_dir` for `metrics.json` files and ingest each one.

    `theorem_set` is informational only — it lets the caller filter to a
    specific theorem set; in practice metrics.json carries this field
    and the row builder uses *that* value, so passing it here is for
    documentation / future filtering.
    """
    base = Path(run_dir)
    total = {"rows_appended": 0, "rows_total": 0, "skeleton_count_after": 0}
    for mp in sorted(base.rglob("metrics.json")):
        run_id = mp.parent.name
        try:
            res = update_archive_from_metrics_path(
                mp,
                run_id=run_id,
                last_seen_commit=last_seen_commit,
                archive_path=archive_path,
                index_path=index_path,
            )
        except Exception as exc:
            print(f"  [archive] skip {mp}: {exc}")
            continue
        total["rows_appended"] += res["rows_appended"]
        total["rows_total"] = res["rows_total"]
        total["skeleton_count_after"] = res["skeleton_count_after"]
    return total


# ---------------------------------------------------------------------- CLI helper
def summarize(
    archive_path: Path | str = DEFAULT_ARCHIVE_PATH,
    top_n: int = 15,
    dead_min_attempts: int = DEFAULT_DEAD_ATTEMPT_THRESHOLD,
) -> str:
    """Human-readable summary string (used by reports)."""
    rows = load_archive(archive_path)
    stats = aggregate(rows)
    top = top_skeletons_by_wins(stats, n=top_n)
    dead = dead_skeletons(stats, min_attempts=dead_min_attempts)

    lines: list[str] = []
    lines.append(f"archive_rows: {len(rows)}")
    lines.append(f"distinct_skeletons: {len(stats)}")
    lines.append(f"top_{top_n}_by_wins:")
    for s in top:
        lines.append(
            f"  {s.skeleton_name:25s} wins={s.wins:3d} adv={s.advances:3d} "
            f"att={s.attempts:4d} shape={s.skeleton_shape!s:8s} "
            f"family={s.skeleton_family!s:8s} origin={s.origin}"
        )
    lines.append(f"dead_skeletons (attempts>={dead_min_attempts}, wins==0): {len(dead)}")
    for s in dead[:top_n]:
        lines.append(
            f"  {s.skeleton_name:25s} att={s.attempts:4d} "
            f"shape={s.skeleton_shape!s:8s} origin={s.origin}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skeleton archive CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub_summary = sub.add_parser("summary")
    sub_summary.add_argument(
        "--archive", default=str(DEFAULT_ARCHIVE_PATH)
    )
    sub_summary.add_argument("--top-n", type=int, default=15)
    sub_summary.add_argument("--dead-min-attempts", type=int, default=10)
    sub_ingest = sub.add_parser("ingest")
    sub_ingest.add_argument("metrics", help="path to a metrics.json")
    sub_ingest.add_argument("--archive", default=str(DEFAULT_ARCHIVE_PATH))
    sub_ingest.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    sub_ingest.add_argument("--commit", default=None)
    sub_ingest.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.cmd == "summary":
        print(
            summarize(
                args.archive,
                top_n=args.top_n,
                dead_min_attempts=args.dead_min_attempts,
            )
        )
    elif args.cmd == "ingest":
        res = update_archive_from_metrics_path(
            args.metrics,
            run_id=args.run_id,
            last_seen_commit=args.commit,
            archive_path=args.archive,
            index_path=args.index,
        )
        print(json.dumps(res, indent=2))
