"""NS8 — extract protected states from per-step traces.

For each protected entry in the NS7 protected_skeletons.json (or
freshly computed from traces), find the actual state_pp from the
trace and the *critical tactic* the skeleton emitted that produced
the credit. Write a per-state JSONL file that the NS8 model-output
cache and rank-simulator consume.

Output schema (one row per (theorem, state_hash, skeleton_stable_id, reason)):

    {
      "theorem": "Nat.div_lt_iff_lt_mul'",
      "state_hash": "abc123...",
      "state_pp": "n m k : ℕ\\n⊢ ...",
      "full_name": "Nat.div_lt_iff_lt_mul'",
      "step": 1,
      "critical_skeleton_stable_id": "deadbeef1234",
      "critical_skeleton_name": "retrieved:Nat.div_lt_iff_lt_mul:rw",
      "critical_tactic": "rw [Nat.div_lt_iff_lt_mul]",
      "critical_origin": "retrieved_premise",
      "critical_role": "advance",     # close | advance
      "reason": "assist_win",         # protection reason
      "observed_rank_in_trace": 16    # rank in wrapper-merged list
    }

Usage:
    python scripts/ns8_protected_states.py \\
        --traces project/evolve/ns7_runs/baseline/medium/eval-XXX/traces.jsonl \\
        --traces project/evolve/ns7_runs/baseline/large/eval-YYY/traces.jsonl \\
        --protected project/evolve/ns7_runs/baseline/protected_skeletons.json \\
        --out project/evolve/archive/protected_states.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ns7_protected_set import _derive_stable_id


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


def build_protected_states(
    rows: list[dict[str, Any]],
    protected_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each protected entry, find the actual trace step that
    earned its credit and return a row with state_pp + critical_tactic.
    """
    # Index trace rows by (theorem, state_hash) and by episode-step.
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        eid = r.get("episode_id")
        if eid:
            by_episode[eid].append(r)

    # Build per-episode accepted-step sequence (close > advance, with rank
    # in wrapper-merged list = position in step_rows).
    # The protected entry has (theorem, state_hash, skeleton_stable_id, reason).
    # We need to locate the trace row that matches.
    out_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    # Group entries by theorem for fast lookup.
    by_theorem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in protected_entries:
        by_theorem[e.get("theorem") or ""].append(e)

    for eid, ep_rows in by_episode.items():
        # Episode id format: "<full_name>:<run_suffix>"
        theorem = eid.split(":")[0] if ":" in eid else eid
        entries = by_theorem.get(theorem, [])
        if not entries:
            continue

        # Group by step + rank.
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in ep_rows:
            try:
                s = int(r.get("step"))
            except (TypeError, ValueError):
                continue
            by_step[s].append(r)

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

        for entry in entries:
            sid_target = entry.get("skeleton_stable_id")
            state_hash_target = entry.get("state_hash")
            reason = entry.get("reason")
            # Find matching accepted step.
            match = None
            for s, r, role, rank in accepted:
                if r.get("state_hash_before") != state_hash_target:
                    continue
                if _derive_stable_id(r) != sid_target:
                    continue
                # For assist/critical_advance, the accepted row's role is
                # `advance`. For direct_win, `close`. Match accordingly.
                if reason == "direct_win" and role != "close":
                    continue
                if reason in ("assist_win", "critical_advance") and role != "advance":
                    continue
                match = (s, r, role, rank)
                break
            if match is None:
                continue
            s, r, role, rank = match
            key = (theorem, state_hash_target, sid_target, reason)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out_rows.append({
                "theorem": theorem,
                "state_hash": state_hash_target,
                "state_pp": r.get("state_pp"),
                "full_name": r.get("full_name") or theorem,
                "step": s,
                "critical_skeleton_stable_id": sid_target,
                "critical_skeleton_name": r.get("skeleton_name"),
                "critical_tactic": r.get("tactic"),
                "critical_origin": r.get("tactic_origin"),
                "critical_role": role,
                "reason": reason,
                "observed_rank_in_trace": rank,
                "source_run": r.get("run_id"),
            })
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", action="append", required=True)
    ap.add_argument("--protected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.traces:
        path = Path(p)
        if not path.exists():
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
    protected = json.loads(args.protected.read_text(encoding="utf-8"))
    protected_entries = protected["entries"]
    print(f"loaded {len(rows)} trace rows, {len(protected_entries)} protected entries")
    out_rows = build_protected_states(rows, protected_entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} protected states to {args.out}")
    by_reason: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    for r in out_rows:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        by_origin[r.get("critical_origin") or "?"] = by_origin.get(r.get("critical_origin") or "?", 0) + 1
    print(f"by reason: {by_reason}")
    print(f"by origin: {by_origin}")


if __name__ == "__main__":
    main()
