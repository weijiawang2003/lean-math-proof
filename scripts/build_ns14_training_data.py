"""NS14 — wider-trace pair extraction.

Walks the NS14 eval-run trace directories produced in Stage 2
(``project/evolve/eval_runs/ns14_*``) and emits a deduplicated
list of supervised (prompt, tactic) pairs.

Inherits the NS11 filtering pipeline (close transitions +
advance_assist within K=3) and adds:

  - origin labeling (``raw_routed`` / ``raw_gen_v5`` /
    ``wrapper_routed`` / ``wrapper_gen_v5``) propagated from the
    eval run name when the trace row lacks ``tactic_origin``.
  - namespace stratification — the meta JSON reports per-namespace
    counts (Nat / Set / Finset / other).
  - retrieved-premise + skeleton transitions are kept (origins
    ``retrieved_premise``, ``skeleton_emitted``) since downstream
    can decide whether to train on them.

Output: a single JSONL plus a meta JSON. The JSONL is *not*
committed (the .gitignore excludes ``project/data/ns14_*.jsonl``);
only the meta JSON is committed.

Usage::

    python scripts/build_ns14_training_data.py \
        --out project/data/ns14_train_combined.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500

# Origins we trust by default (NS11 default-allowed set, plus
# retrieved_premise and skeleton-derived emissions which NS14
# explicitly opts into).
DEFAULT_ALLOWED_ORIGINS = {
    "fallback_tactic", "family_tactic", "generative_topk",
    "term_builder", "tactic_template", "priority_template",
    "retrieved_premise", "skeleton_emitted",
}

# Origin tags inferred from the eval-run dir name when a row's
# tactic_origin is None (raw generative rollouts have no origin
# attribution).
DIR_ORIGIN_FALLBACK = {
    "ns14_routed_wrapper":    "wrapper_routed",
    "ns14_routed_raw":        "raw_routed",
    "ns14_gen_v5_wrapper":    "wrapper_gen_v5",
    "ns14_gen_v5_raw":        "raw_gen_v5",
}


def namespace_of(full_name: str) -> str:
    if not full_name or "." not in full_name:
        return "?"
    ns = full_name.split(".", 1)[0]
    if ns in ("Nat", "Set", "Finset"):
        return ns
    return "other"


def is_close(r: dict) -> bool:
    return bool(r.get("proof_finished"))


def is_advance(r: dict) -> bool:
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


_BAD_TAC_TOKENS = ("error:", "unknown constant", "sorry", "admit")


def looks_bad_tactic(tactic: str, full_name: str) -> bool:
    if not tactic:
        return True
    if len(tactic) > MAX_TACTIC_LEN:
        return True
    lo = tactic.lower()
    for tok in _BAD_TAC_TOKENS:
        if tok in lo:
            return True
    # No self-reference: tactic must not mention the theorem name.
    if full_name and full_name in tactic:
        return True
    return False


def infer_origin_for_dir(dir_name: str) -> str:
    for prefix, tag in DIR_ORIGIN_FALLBACK.items():
        if dir_name.startswith(prefix):
            return tag
    return "unknown"


def build_prompt_from_state(state_pp: str, full_name: str) -> str:
    return f"Theorem: {full_name}\n\nProof state:\n{state_pp}\n"


def extract_episodes(trace_path: Path) -> list[list[dict]]:
    """Group raw trace rows into episodes preserving order."""
    eps: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = r.get("episode_id") or r.get("full_name") or "?"
        if eid not in eps:
            order.append(eid)
        eps[eid].append(r)
    return [eps[k] for k in order]


def extract_rows(
    trace_path: Path,
    *,
    dir_origin: str,
    k_assist_window: int = 3,
) -> list[dict]:
    """Stream one traces.jsonl and produce supervised pairs."""
    rows: list[dict] = []
    episodes = extract_episodes(trace_path)

    for ep in episodes:
        # Mark accepted positions (close or advance) and stash close idx
        # so we can compute assist_distance for advance rows.
        accepted_idx: list[int] = []
        close_idx: int | None = None
        for i, r in enumerate(ep):
            if is_close(r):
                accepted_idx.append(i)
                if close_idx is None:
                    close_idx = i
            elif is_advance(r):
                accepted_idx.append(i)

        for i, r in enumerate(ep):
            role: str | None = None
            assist_distance: int | None = None
            if is_close(r):
                role = "close"
            elif is_advance(r) and close_idx is not None and i < close_idx:
                # advance with downstream close within K accepted positions.
                # Count accepted indices strictly after i, up to close.
                after = [j for j in accepted_idx if j > i and j <= close_idx]
                if 1 <= len(after) <= k_assist_window:
                    role = "advance_assist"
                    assist_distance = len(after)
            if role is None:
                continue

            state_pp = r.get("state_pp") or r.get("state_pp_before") or ""
            tactic = r.get("tactic") or ""
            full_name = r.get("full_name") or ""

            if not state_pp or len(state_pp) > MAX_STATE_LEN:
                continue
            if looks_bad_tactic(tactic, full_name):
                continue

            origin = r.get("tactic_origin") or dir_origin or "unknown"
            if (origin not in DEFAULT_ALLOWED_ORIGINS
                    and origin not in DIR_ORIGIN_FALLBACK.values()):
                # not on our allow-list
                continue

            prompt = build_prompt_from_state(state_pp, full_name)
            state_hash = hashlib.sha1(state_pp.encode("utf-8")).hexdigest()[:16]
            tactic_hash = hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12]
            rows.append({
                "prompt": prompt,
                "tactic": tactic,
                "completion": tactic,
                "theorem": full_name,
                "theorem_set": r.get("theorem_set") or trace_path.parent.name,
                "origin": origin,
                "source_run": trace_path.parent.name,
                "state_hash": state_hash,
                "tactic_hash": tactic_hash,
                "namespace": namespace_of(full_name),
                "role": role,
                "assist_distance": assist_distance,
                "skeleton_stable_id": r.get("skeleton_stable_id"),
                "skeleton_name": r.get("skeleton_name"),
                "_variant": "ns14",
                "_prompt_style": "vanilla",
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace-dir",
        default="project/evolve/eval_runs",
        type=Path,
        help="Root containing the ns14_* eval-run trees.",
    )
    ap.add_argument("--prefix", default="ns14_",
                    help="Restrict to subdirectories with this prefix.")
    ap.add_argument("--out", required=True, type=Path,
                    help="JSONL output path (NOT committed).")
    ap.add_argument("--k-assist-window", type=int, default=3)
    args = ap.parse_args()

    trace_files = sorted(
        p for p in args.trace_dir.glob(f"{args.prefix}*/*/traces.jsonl")
    )
    if not trace_files:
        print(f"no NS14 traces.jsonl under {args.trace_dir} with prefix "
              f"{args.prefix!r} — run Stage 2 evals first.")

    all_rows: list[dict] = []
    per_run_counts: dict[str, int] = {}
    for tp in trace_files:
        # Run dir is parent.parent (e.g. ns14_routed_wrapper_set_finset/<eval-id>/traces.jsonl
        # → run = ns14_routed_wrapper_set_finset).
        run_dir = tp.parent.parent.name
        dir_origin = infer_origin_for_dir(run_dir)
        rows = extract_rows(
            tp, dir_origin=dir_origin,
            k_assist_window=args.k_assist_window,
        )
        per_run_counts[run_dir] = per_run_counts.get(run_dir, 0) + len(rows)
        all_rows.extend(rows)

    # Dedup by (state_hash, tactic_hash).
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in all_rows:
        key = (r["state_hash"], r["tactic_hash"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_ns: dict[str, int] = defaultdict(int)
    by_origin: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    thms: set[str] = set()
    for r in deduped:
        by_ns[r["namespace"]] += 1
        by_origin[r["origin"]] += 1
        by_role[r["role"]] += 1
        thms.add(r["theorem"])

    meta = {
        "trace_files_scanned": len(trace_files),
        "total_rows_pre_dedup": len(all_rows),
        "total_rows_post_dedup": len(deduped),
        "unique_theorems": len(thms),
        "per_run_rows_pre_dedup": per_run_counts,
        "namespace_distribution": dict(by_ns),
        "origin_distribution": dict(by_origin),
        "role_distribution": dict(by_role),
        "filters": {
            "max_tactic_len": MAX_TACTIC_LEN,
            "max_state_len": MAX_STATE_LEN,
            "k_assist_window": args.k_assist_window,
            "allowed_origins": sorted(DEFAULT_ALLOWED_ORIGINS),
            "dir_origin_fallback": DIR_ORIGIN_FALLBACK,
        },
    }
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"trace files scanned: {len(trace_files)}")
    print(f"rows pre-dedup     : {len(all_rows)}")
    print(f"rows post-dedup    : {len(deduped)}")
    print(f"unique theorems    : {len(thms)}")
    print(f"by namespace       : {dict(by_ns)}")
    print(f"by origin          : {dict(by_origin)}")
    print(f"by role            : {dict(by_role)}")
    print(f"wrote {args.out} and {meta_path}")


if __name__ == "__main__":
    main()
