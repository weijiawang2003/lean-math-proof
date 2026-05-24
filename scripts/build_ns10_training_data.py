"""NS10 — build supervised-training data from NS9/v5 verified traces.

Builds on `scripts/build_v5_training_data.py`. Differences from v5:

  - Reads from the NS9 trace directories
    (`project/evolve/ns9_runs/`, `project/evolve/ns7_runs/`, etc.)
    in addition to `project/evolve/autonomous_runs/`. Pass each as
    `--traces-dir`.
  - Carries NS7 `skeleton_stable_id` + NS6 attribution fields when
    present.
  - Includes `assist_within_k` rows: advancing transitions whose
    proof closes within K accepted steps later.
  - Retrieved-premise tactics are gated by `--include-retrieved`
    (default off; the v5 plan recommends excluding them because
    they aren't reproducible without the retriever).

Held-out theorems by default include the 5 v5 priority-template
new wins so a later v5+1 eval can honestly test whether the new
model learned to emit them natively.

Output (default paths):

  - `project/data/ns10_evolve_train.jsonl` — one row per
    (prompt, completion) pair.
  - `project/data/ns10_evolve_train_meta.json` — counts +
    provenance + filter summary.

Each row:

    {
      "prompt": "Theorem: <full_name>\\n\\nProof state:\\n<state_pp>\\n",
      "completion": "<tactic>",
      "theorem": "<full_name>",
      "theorem_set": "<set_name or unknown>",
      "origin": "<tactic_template | family_tactic | ...>",
      "source_run": "<run_id>",
      "state_hash": "<state_hash_before>",
      "tactic_hash": "<sha1(tactic)[:12]>",
      "skeleton_name": "<name or null>",
      "skeleton_stable_id": "<sid or null>",
      "role": "close" | "advance_assist" | "advance",
      "assist_distance": <int>   # only for advance_assist
    }

Usage:
    python scripts/build_ns10_training_data.py \\
        --traces-dir project/evolve/ns9_runs \\
        --traces-dir project/evolve/ns7_runs \\
        --out project/data/ns10_evolve_train.jsonl \\
        --k 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Any


DEFAULT_HELD_OUT = {
    # NS9 single residual + NS6/NS7/NS8 brittle theorem
    "Nat.AM_GM",
    "Nat.div_lt_iff_lt_mul'",
    # v5 priority-template new wins — kept out so v5+1 eval is honest
    "Nat.div_lt_one_iff",
    "Nat.div_pos",
    "Nat.div_pos_iff",
    "Nat.mul_eq_left",
    "Nat.mul_eq_right",
    # additional structurally-tricky theorems
    "Nat.dvd_iff_div_mul_eq",
    "Nat.sqrt_lt",
    "Nat.pow_lt_pow_iff_left",
}

# Default-allowed origins (matches v5 plan; retrieved_premise opt-in).
DEFAULT_ALLOWED = {
    "fallback_tactic", "family_tactic", "generative_topk",
    "term_builder", "tactic_template",
}

MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500


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


def iter_trace_paths(traces_dirs: list[Path]) -> Iterable[Path]:
    """Yield every traces.jsonl under any of the given dirs."""
    seen: set[str] = set()
    for d in traces_dirs:
        for p in d.rglob("traces.jsonl"):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield p


def load_episodes(
    trace_paths: list[Path],
) -> dict[str, list[dict[str, Any]]]:
    """Group every (episode_id, step, row) into episodes."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in trace_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_source_run"] = p.parent.name
            r["_source_trace_path"] = str(p)
            eid = r.get("episode_id")
            if not eid:
                continue
            by_episode[eid].append(r)
    return by_episode


def accepted_sequence(
    rows: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any], str, int]]:
    """Per-step accepted (close|advance) row + rank (position in step)."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            s = int(r.get("step"))
        except (TypeError, ValueError):
            continue
        by_step[s].append(r)
    out: list[tuple[int, dict[str, Any], str, int]] = []
    for s in sorted(by_step.keys()):
        step_rows = by_step[s]
        cidx = next((i for i, r in enumerate(step_rows) if _is_close(r)), None)
        if cidx is not None:
            out.append((s, step_rows[cidx], "close", cidx))
            continue
        aidx = next((i for i, r in enumerate(step_rows) if _is_advance(r)), None)
        if aidx is not None:
            out.append((s, step_rows[aidx], "advance", aidx))
    return out


def is_admissible(
    r: dict[str, Any],
    held_out: set[str],
    allowed: set[str],
) -> bool:
    tac = r.get("tactic") or ""
    state = r.get("state_pp") or ""
    name = r.get("full_name") or ""
    if not tac:
        return False
    if len(tac) > MAX_TACTIC_LEN or len(state) > MAX_STATE_LEN:
        return False
    if name in held_out:
        return False
    if name and name in tac:
        return False  # self-reference
    origin = r.get("tactic_origin")
    if origin not in allowed:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", action="append", required=True, type=Path)
    p.add_argument("--out", default="project/data/ns10_evolve_train.jsonl", type=Path)
    p.add_argument("--meta", default="project/data/ns10_evolve_train_meta.json",
                   type=Path)
    p.add_argument("--k", type=int, default=3,
                   help="advance is included if a close lands within K accepted steps")
    p.add_argument("--include-retrieved", action="store_true",
                   help="also include retrieved_premise wins/assists "
                        "(NOT reproducible without the retriever)")
    p.add_argument("--held-out", nargs="*", default=None)
    args = p.parse_args()

    held_out = set(args.held_out) if args.held_out else set(DEFAULT_HELD_OUT)
    allowed = set(DEFAULT_ALLOWED)
    if args.include_retrieved:
        allowed.add("retrieved_premise")

    trace_paths = list(iter_trace_paths(args.traces_dir))
    print(f"trace files: {len(trace_paths)}")
    episodes = load_episodes(trace_paths)
    print(f"episodes:    {len(episodes)}")

    pairs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    by_origin: dict[str, int] = defaultdict(int)
    by_theorem: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    by_skeleton: dict[str, int] = defaultdict(int)
    runs_seen: set[str] = set()
    held_out_seen: dict[str, int] = defaultdict(int)
    tactic_lens: list[int] = []

    for eid, ep_rows in episodes.items():
        acc = accepted_sequence(ep_rows)
        if not acc:
            continue
        for i, (step, r, role, rank) in enumerate(acc):
            if r.get("full_name") in held_out:
                held_out_seen[r["full_name"]] += 1
                continue
            if not is_admissible(r, held_out, allowed):
                continue
            classify_role = role
            assist_distance: int | None = None
            if role == "advance":
                # Only keep advance if a close occurs within K accepted steps.
                close_in_window = None
                for j in range(i + 1, min(len(acc), i + 1 + args.k)):
                    if acc[j][2] == "close":
                        close_in_window = j - i
                        break
                if close_in_window is None:
                    continue
                classify_role = "advance_assist"
                assist_distance = close_in_window
            # Build the training pair.
            state = r["state_pp"]
            tactic = r["tactic"]
            key = (state, tactic)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            tactic_hash = hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12]
            pair = {
                "prompt": f"Theorem: {r.get('full_name','')}\n\nProof state:\n{state}\n",
                # Trainer (`train_tactic_generator.py`) reads `tactic`.
                # We also keep `completion` for tools/specs that use the
                # OpenAI-style name. They are identical.
                "tactic": tactic,
                "completion": tactic,
                "theorem": r.get("full_name", ""),
                "theorem_set": r.get("_source_run", "unknown"),
                "origin": r.get("tactic_origin"),
                "source_run": r.get("_source_run", ""),
                "state_hash": r.get("state_hash_before"),
                "tactic_hash": tactic_hash,
                "skeleton_name": r.get("skeleton_name"),
                "skeleton_stable_id": r.get("skeleton_stable_id"),
                "skeleton_shape": r.get("skeleton_shape"),
                "skeleton_family": r.get("skeleton_family"),
                "role": classify_role,
                "assist_distance": assist_distance,
            }
            pairs.append(pair)
            by_origin[pair["origin"] or "?"] += 1
            by_theorem[pair["theorem"] or "?"] += 1
            by_role[classify_role] += 1
            by_skeleton[pair["skeleton_name"] or "(none)"] += 1
            runs_seen.add(pair["source_run"])
            tactic_lens.append(len(tactic))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    md5 = hashlib.md5(args.out.read_bytes()).hexdigest()
    meta = {
        "v": "ns10",
        "out_path": str(args.out),
        "md5": md5,
        "n_pairs": len(pairs),
        "n_unique_theorems": len(by_theorem),
        "n_runs_seen": len(runs_seen),
        "runs_seen": sorted(runs_seen),
        "held_out": sorted(held_out),
        "held_out_seen_in_traces": dict(held_out_seen),
        "by_origin": dict(by_origin),
        "by_role": dict(by_role),
        "top_theorems": dict(sorted(by_theorem.items(), key=lambda x: -x[1])[:15]),
        "top_skeletons": dict(sorted(by_skeleton.items(), key=lambda x: -x[1])[:15]),
        "tactic_length_stats": {
            "n": len(tactic_lens),
            "min": min(tactic_lens) if tactic_lens else None,
            "max": max(tactic_lens) if tactic_lens else None,
            "mean": (sum(tactic_lens) / len(tactic_lens)) if tactic_lens else None,
        },
        "filters": {
            "max_tactic_len": MAX_TACTIC_LEN,
            "max_state_len": MAX_STATE_LEN,
            "allowed_origins": sorted(allowed),
            "include_retrieved": args.include_retrieved,
            "k_assist_window": args.k,
            "rejects_self_reference": True,
            "rejects_held_out": True,
            "dedup_by_state_tactic": True,
        },
    }
    args.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"wrote {len(pairs)} pairs to {args.out}")
    print(f"  origins: {dict(by_origin)}")
    print(f"  roles: {dict(by_role)}")
    print(f"  unique theorems: {len(by_theorem)}")
    print(f"  held-out trace rows skipped: {sum(held_out_seen.values())}")
    print(f"  meta: {args.meta}")


if __name__ == "__main__":
    main()
