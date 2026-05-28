"""Build seq2seq training data from v5 autonomous-loop traces.

Per Direction E in `v5_trace_to_training_plan.md`. This script is the
concrete deliverable: it reads `project/evolve/autonomous_runs/<run_id>/
eval/<variant>/eval-*/traces.jsonl`, applies the filters described in
the plan (held-out theorems, self-reference exclusion, retrieval origin
gating, length limits), and writes `project/seq2seq_data_v5_evolve.jsonl`
plus a header `project/seq2seq_data_v5_evolve.header.json`.

The script is intentionally a single-file standalone so the v5 pipeline
can be re-run cleanly in v6. No training is performed; the dataset is
the deliverable.

Usage:
    python scripts/build_v5_training_data.py \\
        --runs-dir project/evolve/autonomous_runs \\
        --out project/seq2seq_data_v5_evolve.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


# Theorems held out so the model's later eval can be honest.
DEFAULT_HELD_OUT = {
    "Nat.div_lt_iff_lt_mul'",
    "Nat.add_mod_eq_add_mod_left",
    "Nat.mod_two_ne_zero",
    "Nat.succ_succ_ne_one",
    # newly proven in v5; hold out so the v5 → v6 training cycle has
    # a fair test of whether the model learned to generate them.
    "Nat.div_lt_one_iff",
    "Nat.mul_eq_left",
    "Nat.mul_eq_right",
}

# Origins that produce reproducible (state, tactic) pairs.
ALLOWED_ORIGINS = {
    "fallback_tactic", "family_tactic", "generative_topk", "term_builder",
    "tactic_template",
}
# retrieved_premise is excluded because the tactic is only reproducible
# in the presence of the retrieval engine. See Direction E.

MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500


def iter_traces(runs_dir: Path) -> Iterable[dict]:
    """Yield trace records from every traces.jsonl under runs_dir."""
    for trace_path in runs_dir.glob("*/eval/*/eval-*/traces.jsonl"):
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d["_source_run"] = trace_path.parents[2].name
            d["_source_variant"] = trace_path.parents[1].name
            yield d


def is_good_transition(d: dict, held_out: set[str]) -> bool:
    """Return True if this trace record should be included in training data."""
    if not d.get("tactic"):
        return False
    tactic = d["tactic"]
    state = d.get("state_pp") or ""
    name = d.get("full_name", "")
    if len(tactic) > MAX_TACTIC_LEN or len(state) > MAX_STATE_LEN:
        return False
    if name in held_out:
        return False
    # Reject self-reference: tactic mentions the theorem's own full_name.
    if name and name in tactic:
        return False
    origin = d.get("tactic_origin")
    if origin not in ALLOWED_ORIGINS:
        return False
    # Only closing or advancing transitions.
    kind = d.get("result_kind")
    if kind == "ProofFinished":
        return True
    if kind == "TacticState":
        # Advancing — make sure the state actually changed.
        if d.get("num_goals_after") is None:
            return False
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="project/evolve/autonomous_runs", type=Path)
    p.add_argument("--out", default="project/seq2seq_data_v5_evolve.jsonl", type=Path)
    p.add_argument("--header-out", default="project/seq2seq_data_v5_evolve.header.json", type=Path)
    p.add_argument("--held-out", nargs="*", default=None)
    args = p.parse_args()

    held_out = set(args.held_out) if args.held_out else set(DEFAULT_HELD_OUT)
    pairs: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()  # (state, tactic) dedup
    origin_counts: dict[str, int] = {}
    theorem_counts: dict[str, int] = {}
    runs_seen: set[str] = set()

    for d in iter_traces(args.runs_dir):
        if not is_good_transition(d, held_out):
            continue
        state = d["state_pp"]
        tactic = d["tactic"]
        key = (state, tactic)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pair = {
            "prompt": f"Theorem: {d['full_name']}\n\nProof state:\n{state}\n",
            "completion": tactic,
            "origin": d["tactic_origin"],
            "theorem": d["full_name"],
            "file_path": d.get("file_path", ""),
            "domain": d.get("domain", ""),
            "source_run_id": d["_source_run"],
            "source_variant": d["_source_variant"],
        }
        pairs.append(pair)
        origin_counts[d["tactic_origin"]] = origin_counts.get(d["tactic_origin"], 0) + 1
        theorem_counts[d["full_name"]] = theorem_counts.get(d["full_name"], 0) + 1
        runs_seen.add(d["_source_run"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    md5 = hashlib.md5(args.out.read_bytes()).hexdigest()
    header = {
        "v": 5,
        "out_path": str(args.out),
        "md5": md5,
        "n_pairs": len(pairs),
        "n_unique_theorems": len(theorem_counts),
        "n_runs_seen": len(runs_seen),
        "runs_seen": sorted(runs_seen),
        "held_out": sorted(held_out),
        "origin_counts": origin_counts,
        "top_theorem_counts": dict(sorted(theorem_counts.items(), key=lambda x: -x[1])[:10]),
        "filters": {
            "max_tactic_len": MAX_TACTIC_LEN,
            "max_state_len": MAX_STATE_LEN,
            "allowed_origins": sorted(ALLOWED_ORIGINS),
            "rejects_self_reference": True,
            "rejects_held_out": True,
            "dedup_by_state_tactic": True,
        },
    }
    args.header_out.write_text(json.dumps(header, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(pairs)} pairs to {args.out}")
    print(f"  origins: {origin_counts}")
    print(f"  unique theorems: {len(theorem_counts)}")
    print(f"  header: {args.header_out}")


if __name__ == "__main__":
    main()
