"""NS7 — protected skeleton set.

Walk per-step traces and emit a JSON file listing skeletons that MUST
NOT be removed (or rank-displaced) from the wrapper's ranked tactic
list, with the (theorem, state_hash) where each protection applies.

Three protection categories:

    direct_win       — the skeleton emitted the closing tactic on this
                       theorem at this state
    assist_win       — the skeleton advanced state, and a different
                       tactic closed the proof within K accepted steps
    critical_advance — the skeleton advanced state on a step that was
                       on the *successful* path (followed by another
                       advance/close; not a dead end)

We key by `skeleton_stable_id` (NS7 stable identifier from
`Skeleton.stable_id`). For older traces missing the field we derive
the stable_id retrospectively from origin/shape/family/specificity +
template_source.

Output schema:

    {
      "k_window": 3,
      "skeleton_count": N,
      "entries": [
        {
          "skeleton_stable_id": "abc123def",
          "skeleton_name": "pt_iff_8",
          "origin": "priority_template",
          "shape": "iff",
          "family": null,
          "theorem": "Nat.add_eq_left",
          "state_hash": "5d4b659ed8d5bfc4",
          "reason": "direct_win",         # or assist_win, critical_advance
          "required_rank_max": 5,         # observed rank of the tactic
                                          # in the wrapper's emit order;
                                          # mutations must not push the
                                          # skeleton past this index.
          "source_run": "eval-77302f0c"
        },
        ...
      ]
    }

Usage:
    python scripts/ns7_protected_set.py \\
        --traces project/evolve/ns7_runs/baseline/medium/eval-XXX/traces.jsonl \\
        --traces project/evolve/ns7_runs/baseline/large/eval-YYY/traces.jsonl \\
        --out project/evolve/archive/protected_skeletons.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


def _derive_stable_id(row: dict[str, Any]) -> str | None:
    """Reconstruct stable_id for traces that pre-date NS7."""
    sid = row.get("skeleton_stable_id")
    if sid:
        return sid
    name = row.get("skeleton_name")
    if not name:
        return None
    origin = None
    # Derive origin from name prefix (same scheme as ns6_assist_credit).
    for prefix, o in (
        ("pt_", "priority_template"),
        ("fam_", "family_tactic"),
        ("tb_", "term_builder"),
        ("fb_", "fallback_tactic"),
        ("tt_", "tactic_template"),
        ("retrieved:", "retrieved_premise"),
    ):
        if name.startswith(prefix):
            origin = o
            break
    if origin is None:
        return None
    shape = row.get("skeleton_shape") or ""
    family = row.get("skeleton_family") or ""
    specificity = row.get("skeleton_specificity")
    template = (row.get("tactic_template_source") or "").strip()
    if origin == "retrieved_premise":
        # Retrieved skeletons use a different canonical form (see
        # SkeletonBag.emit_retrieved_tactics).
        premise = row.get("tactic_retrieved_premise") or ""
        form = row.get("tactic_retrieved_form") or ""
        canonical = "|".join((
            "retrieved_premise", shape, family, "0",
            template, premise, form,
        ))
    else:
        canonical = "|".join((
            origin, shape, family, str(specificity if specificity is not None else 1),
            template,
        ))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _is_close(r: dict[str, Any]) -> bool:
    return bool(r.get("proof_finished"))


def _is_advance(r: dict[str, Any]) -> bool:
    if r.get("proof_finished"):
        return False
    kind = r.get("result_kind") or ""
    if kind == "LeanError":
        return False
    if kind in {"SkippedBloatingApply", "SkippedKnownError"}:
        return False
    if r.get("loop_detected") or r.get("bloat_rejected"):
        return False
    return r.get("state_hash_after") is not None


@dataclass
class ProtectionEntry:
    skeleton_stable_id: str
    skeleton_name: str
    origin: str | None
    shape: str | None
    family: str | None
    theorem: str
    state_hash: str | None
    reason: str  # direct_win | assist_win | critical_advance
    required_rank_max: int | None  # observed rank in trace
    source_run: str | None


def build_protected_set(
    rows: list[dict[str, Any]],
    k: int = 3,
) -> dict[str, list[ProtectionEntry]]:
    """Return mapping stable_id → list[ProtectionEntry]."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        eid = r.get("episode_id")
        if eid:
            by_episode[eid].append(r)

    out: dict[str, list[ProtectionEntry]] = defaultdict(list)

    for eid, ep_rows in by_episode.items():
        # Group by step.
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in ep_rows:
            try:
                s = int(r.get("step"))
            except (TypeError, ValueError):
                continue
            by_step[s].append(r)
        # Walk steps in order. The "accepted" row per step is the
        # close or first advance. Track rank within each step (the
        # rank IS the position in ep_rows for that step — eval_rollout_all
        # appends in rank order).
        accepted: list[tuple[int, dict[str, Any], str, int]] = []  # (step, row, role, rank)
        for s in sorted(by_step.keys()):
            step_rows = by_step[s]
            close_idx = next(
                (i for i, r in enumerate(step_rows) if _is_close(r)), None
            )
            if close_idx is not None:
                accepted.append((s, step_rows[close_idx], "close", close_idx))
                continue
            adv_idx = next(
                (i for i, r in enumerate(step_rows) if _is_advance(r)), None
            )
            if adv_idx is not None:
                accepted.append((s, step_rows[adv_idx], "advance", adv_idx))

        # Direct/assist/critical credit per accepted row.
        for i, (_, r, role, rank) in enumerate(accepted):
            name = r.get("skeleton_name")
            if not name:
                continue
            sid = _derive_stable_id(r)
            if not sid:
                continue
            entry_template = dict(
                skeleton_stable_id=sid,
                skeleton_name=name,
                origin=None,  # filled below
                shape=r.get("skeleton_shape"),
                family=r.get("skeleton_family"),
                theorem=r.get("full_name") or "",
                state_hash=r.get("state_hash_before"),
                required_rank_max=rank,
                source_run=r.get("run_id"),
            )
            # Derive origin from skeleton_name prefix.
            for prefix, o in (
                ("pt_", "priority_template"),
                ("fam_", "family_tactic"),
                ("tb_", "term_builder"),
                ("fb_", "fallback_tactic"),
                ("tt_", "tactic_template"),
                ("retrieved:", "retrieved_premise"),
            ):
                if name.startswith(prefix):
                    entry_template["origin"] = o
                    break

            if role == "close":
                out[sid].append(ProtectionEntry(reason="direct_win", **entry_template))
            elif role == "advance":
                # Check assist within K accepted steps for a close.
                closes_within = False
                next_advance_within = False
                for j in range(i + 1, min(len(accepted), i + 1 + k)):
                    _, _, frole, _ = accepted[j]
                    if frole == "close":
                        closes_within = True
                        break
                    if frole == "advance":
                        next_advance_within = True
                if closes_within:
                    out[sid].append(ProtectionEntry(reason="assist_win", **entry_template))
                elif next_advance_within:
                    out[sid].append(ProtectionEntry(reason="critical_advance", **entry_template))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", action="append", required=True,
                    help="traces.jsonl path (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.traces:
        path = Path(p)
        if not path.exists():
            print(f"skipping missing: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"loaded {len(rows)} rows from {len(args.traces)} file(s)")

    protected = build_protected_set(rows, k=args.k)
    entries: list[dict[str, Any]] = []
    for sid, eps in protected.items():
        for e in eps:
            entries.append(asdict(e))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "k_window": args.k,
        "skeleton_count": len(protected),
        "entry_count": len(entries),
        "entries": entries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"protected stable_ids: {len(protected)}")
    print(f"total entries:        {len(entries)}")
    print(f"by reason: ", {
        r: sum(1 for e in entries if e["reason"] == r)
        for r in ("direct_win", "assist_win", "critical_advance")
    })
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
