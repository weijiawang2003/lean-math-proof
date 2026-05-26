"""NS21 — Finset/aesop imitation training-data builder.

The CX1 combined wrapper-only pool surfaced 6 unique theorems where
the variant wrapper emits ``aesop`` and aesop closes the proof, but
the NS15-routed raw model fails to emit aesop:

  Finset.coe_insert            (NS18, ns17_finset_extra)
  Finset.cons_eq_insert        (NS18, ns17_finset_extra)
  Finset.disjUnion_singleton   (NS18, ns17_finset_extra)
  Finset.coe_cons              (NS19, ns19_finset_aesop_surface)
  Finset.card_insert_eq_ite    (CX1,  cx1_finset_image_filter)
  Finset.image_id              (CX1,  cx1_finset_image_filter)

This script extracts the wrapper-only close-rows for those 6
theorems and combines them with NS12 balanced replay (which
preserves demo/Set/Finset behavior). Three dataset variants are
produced: 10x oversample, 20x oversample, minimal.

Hard exclusions per NS21 spec:
- NO Nat simp_all wrapper-only rows (sub-gate, 3 unique only)
- NO Int iff_omega_pair / fallback_omega rows (sub-gate)
- NO heterogeneous wrapper-only rows

Outputs (JSONL gitignored, meta committed):
  project/data/ns21_finset_aesop_10x.jsonl
  project/data/ns21_finset_aesop_10x_meta.json
  project/data/ns21_finset_aesop_20x.jsonl
  project/data/ns21_finset_aesop_20x_meta.json
  project/data/ns21_finset_aesop_minimal.jsonl
  project/data/ns21_finset_aesop_minimal_meta.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500

ALLOWED_ORIGINS = {
    "fallback_tactic", "family_tactic", "generative_topk",
    "term_builder", "tactic_template", "priority_template",
    "retrieved_premise", "skeleton_emitted",
}

# Per-set wrapper-traces glob and the wrapper-only theorems
# from that trace that belong in the Finset/aesop pool.
POOL_SOURCES = [
    {
        "set": "ns17_finset_extra",
        "wrap_traces": (
            "project/evolve/eval_runs/"
            "ns18_aesop_wrapper_wrapper_ns17_finset_extra/eval-*/traces.jsonl"
        ),
        "theorems": [
            "Finset.coe_insert",
            "Finset.cons_eq_insert",
            "Finset.disjUnion_singleton",
        ],
        "first_seen_arc": "NS18",
    },
    {
        "set": "ns19_finset_aesop_surface",
        "wrap_traces": (
            "project/evolve/eval_runs/"
            "ns19_finset_aesop_only_wrapper_ns19_finset_aesop_surface/"
            "eval-*/traces.jsonl"
        ),
        "theorems": ["Finset.coe_cons"],
        "first_seen_arc": "NS19",
    },
    {
        "set": "cx1_finset_image_filter",
        "wrap_traces": (
            "project/evolve/eval_runs/"
            "cx1_finset_aesop_only_wrapper_cx1_finset_image_filter/"
            "eval-*/traces.jsonl"
        ),
        "theorems": [
            "Finset.card_insert_eq_ite",
            "Finset.image_id",
        ],
        "first_seen_arc": "CX1",
    },
]

NS12_BALANCED_PATH = Path("project/data/ns12_train_balanced.jsonl")


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def hash_state(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def hash_tactic(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]


def extract_pool_rows() -> list[dict]:
    """Pull the close-row (state_pp, tactic="aesop") for each pool theorem."""
    rows: list[dict] = []
    seen: set[str] = set()
    for src in POOL_SOURCES:
        trace_path = first_match(src["wrap_traces"])
        if not trace_path:
            print(f"WARNING: no trace for {src['set']}")
            continue
        wanted = set(src["theorems"])
        for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("proof_finished"):
                continue
            full = r.get("full_name") or ""
            if full not in wanted:
                continue
            if full in seen:
                continue  # one row per theorem (the close row)
            tac = r.get("tactic") or ""
            if tac.strip() != "aesop":
                continue  # only the homogeneous aesop close
            origin = r.get("tactic_origin") or "unknown"
            if origin not in ALLOWED_ORIGINS:
                continue
            state = r.get("state_pp") or r.get("state_pp_before") or ""
            if not state or len(state) > MAX_STATE_LEN:
                continue
            prompt = f"Theorem: {full}\n\nProof state:\n{state}\n"
            rows.append({
                "prompt": prompt,
                "tactic": "aesop",
                "completion": "aesop",
                "theorem": full,
                "theorem_set": src["set"],
                "origin": origin,
                "source_run": Path(trace_path).parent.name,
                "state_hash": hash_state(state),
                "tactic_hash": hash_tactic("aesop"),
                "namespace": "Finset",
                "role": "close",
                "assist_distance": None,
                "skeleton_stable_id": r.get("skeleton_stable_id"),
                "skeleton_name": r.get("skeleton_name"),
                "skeleton_shape": r.get("skeleton_shape"),
                "skeleton_family": r.get("skeleton_family"),
                "wrapper_only": True,
                "first_seen_arc": src["first_seen_arc"],
                "_variant": "ns21",
                "_prompt_style": "vanilla",
            })
            seen.add(full)
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_variant(
    *,
    variant: str,
    pool_rows: list[dict],
    replay_rows: list[dict],
    oversample: int,
    out_path: Path,
    seed: int = 42,
) -> dict:
    out_rows: list[dict] = []

    # Oversample pool rows.
    for i in range(oversample):
        for r in pool_rows:
            rr = dict(r)
            rr["_oversample_idx"] = i
            out_rows.append(rr)

    # Mix in replay (full for 10x/20x; sampled for minimal).
    out_rows.extend(replay_rows)

    # Shuffle so pool rows aren't all at the top.
    rng = random.Random(seed)
    rng.shuffle(out_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_role: dict[str, int] = defaultdict(int)
    by_origin: dict[str, int] = defaultdict(int)
    by_namespace: dict[str, int] = defaultdict(int)
    by_tactic: dict[str, int] = defaultdict(int)
    pool_thms: set[str] = set()
    n_pool_rows = 0
    for r in out_rows:
        by_role[r.get("role") or "unknown"] += 1
        by_origin[r.get("origin") or "unknown"] += 1
        by_namespace[r.get("namespace") or "?"] += 1
        # Tactic distribution truncated to first 40 chars for a histogram.
        t = (r.get("tactic") or "")[:40]
        by_tactic[t] += 1
        if r.get("_variant") == "ns21":
            n_pool_rows += 1
            pool_thms.add(r["theorem"])

    # Top-20 tactics only in meta (full histogram pollutes commit diffs).
    top_tactics = dict(sorted(
        by_tactic.items(), key=lambda kv: -kv[1]
    )[:20])

    meta = {
        "variant": variant,
        "out_path": str(out_path),
        "n_rows": len(out_rows),
        "n_pool_rows": n_pool_rows,
        "n_pool_unique_theorems": len(pool_thms),
        "n_pool_source_rows": len(pool_rows),
        "oversample_factor": oversample,
        "n_replay_rows": len(replay_rows),
        "replay_source": str(NS12_BALANCED_PATH),
        "init_from_recommended": "project/models/gen_v5_ns12_balanced",
        "exclusions": [
            "Nat simp_all wrapper-only (sub-gate)",
            "Int iff_omega_pair / fallback_omega (sub-gate)",
            "heterogeneous wrapper-only rows",
        ],
        "by_role": dict(by_role),
        "by_origin": dict(by_origin),
        "by_namespace": dict(by_namespace),
        "top_tactics_truncated": top_tactics,
        "pool_theorems": sorted(pool_thms),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        required=True,
        choices=["10x", "20x", "minimal", "all"],
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool_rows = extract_pool_rows()
    print(f"pool rows extracted: {len(pool_rows)} from "
          f"{len({r['theorem'] for r in pool_rows})} theorems")
    if len(pool_rows) != 6:
        print(
            f"WARNING: expected 6 pool rows, got {len(pool_rows)}."
            " Continuing — check trace globs and pool meta."
        )

    if not NS12_BALANCED_PATH.exists():
        raise SystemExit(
            f"missing {NS12_BALANCED_PATH} — required for replay."
        )
    replay_full = load_jsonl(NS12_BALANCED_PATH)
    rng = random.Random(args.seed)

    # Minimal replay = random sample of 500 rows from the NS12 balanced
    # corpus. NS12 is already 50% Set+Finset by construction (3432 Set +
    # 3752 Finset + 261 Nat by domain meta), so a random sample preserves
    # that anti-forgetting mix without forcing a brittle field filter on
    # heterogeneous v5-base rows that lack a namespace column.
    minimal_replay_pool = list(replay_full)
    rng.shuffle(minimal_replay_pool)
    minimal_replay = minimal_replay_pool[:500]
    print(f"minimal replay: {len(minimal_replay)} rows "
          f"(random sample from {len(replay_full)} NS12 balanced rows)")

    variants_to_run = (
        ["10x", "20x", "minimal"] if args.variant == "all"
        else [args.variant]
    )

    for v in variants_to_run:
        if v == "10x":
            out = Path("project/data/ns21_finset_aesop_10x.jsonl")
            meta = build_variant(
                variant=v, pool_rows=pool_rows,
                replay_rows=list(replay_full), oversample=10,
                out_path=out, seed=args.seed,
            )
        elif v == "20x":
            out = Path("project/data/ns21_finset_aesop_20x.jsonl")
            meta = build_variant(
                variant=v, pool_rows=pool_rows,
                replay_rows=list(replay_full), oversample=20,
                out_path=out, seed=args.seed,
            )
        elif v == "minimal":
            out = Path("project/data/ns21_finset_aesop_minimal.jsonl")
            meta = build_variant(
                variant=v, pool_rows=pool_rows,
                replay_rows=list(minimal_replay), oversample=20,
                out_path=out, seed=args.seed,
            )
        else:
            raise SystemExit(f"unknown variant {v}")
        print(f"\n=== variant {v} ===")
        print(f"out                = {meta['out_path']}")
        print(f"total rows         = {meta['n_rows']}")
        print(f"pool rows          = {meta['n_pool_rows']}")
        print(f"pool unique thms   = {meta['n_pool_unique_theorems']}")
        print(f"oversample         = {meta['oversample_factor']}")
        print(f"replay rows        = {meta['n_replay_rows']}")
        print(f"by_namespace       = {meta['by_namespace']}")


if __name__ == "__main__":
    main()
