"""NS22 — Int iff_omega + (ablation) fallback_omega imitation training data.

CX2's combined Int wrapper-only-vs-NS9 pool, per
`project/data/cx2_int_iff_omega_pool_meta.json`:

  - iff_omega_pair / Int (10 unique). All emit:
      exact ⟨fun h => by omega, fun h => by omega⟩
  - fallback_omega / Int (13 unique). All emit: omega

NS22-A trains on iff_omega_pair only (homogeneous, lower-risk).
NS22-B is the fallback_omega ablation. NS22-A and NS22-B are
**never mixed** in the same training dataset per NS22 hard
constraint.

Variants:

  - `ns22_int_iff_omega_5x`   — 10 iff_omega rows × 5 + NS12 replay
  - `ns22_int_iff_omega_10x`  — 10 iff_omega rows × 10 + NS12 replay
  - `ns22_int_fallback_omega_5x` — ablation, 13 omega rows × 5 + NS12 replay

Outputs:
  project/data/ns22_int_iff_omega_5x_meta.json   (committed)
  project/data/ns22_int_iff_omega_5x.jsonl       (gitignored)
  project/data/ns22_int_iff_omega_10x_meta.json
  project/data/ns22_int_iff_omega_10x.jsonl
  project/data/ns22_int_fallback_omega_5x_meta.json
  project/data/ns22_int_fallback_omega_5x.jsonl
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

POOL_META = Path("project/data/cx2_int_iff_omega_pool_meta.json")
NS12_BALANCED_PATH = Path("project/data/ns12_train_balanced.jsonl")

WRAPPER_TRACE_GLOBS = [
    ("CX1", "cx1_bool_option_int",
     "project/evolve/eval_runs/cx1_ns9wrap_cx1_bool_option_int/eval-*/traces.jsonl"),
    ("CX2", "cx2_int_iff_omega_easy",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_iff_omega_easy/eval-*/traces.jsonl"),
    ("CX2", "cx2_int_iff_omega_medium",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_iff_omega_medium/eval-*/traces.jsonl"),
    ("CX2", "cx2_int_order_arith",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_order_arith/eval-*/traces.jsonl"),
    ("CX2", "cx2_int_mixed",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_mixed/eval-*/traces.jsonl"),
]


def hash_state(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def hash_tactic(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]


def load_pool_theorems() -> tuple[set[str], set[str]]:
    meta = json.load(open(POOL_META))
    iff = set(meta["families"]["iff_omega_pair|Int"]["theorems"].keys())
    omega = set(meta["families"]["fallback_omega|Int"]["theorems"].keys())
    return iff, omega


def extract_close_rows(
    want_theorems: set[str],
    want_tactic_substring: str,
) -> list[dict]:
    """Find the close (proof_finished) row for each wanted theorem.

    Returns at most one row per theorem (first match across globs).
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for arc, set_name, glob_pat in WRAPPER_TRACE_GLOBS:
        for p in sorted(glob.glob(glob_pat)):
            for line in Path(p).read_text(encoding="utf-8").splitlines():
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
                if full not in want_theorems or full in seen:
                    continue
                tac = (r.get("tactic") or "").strip()
                if want_tactic_substring and want_tactic_substring not in tac:
                    continue
                origin = r.get("tactic_origin") or "unknown"
                if origin not in ALLOWED_ORIGINS:
                    continue
                state = r.get("state_pp") or r.get("state_pp_before") or ""
                if not state or len(state) > MAX_STATE_LEN:
                    continue
                if len(tac) > MAX_TACTIC_LEN:
                    continue
                prompt = f"Theorem: {full}\n\nProof state:\n{state}\n"
                rows.append({
                    "prompt": prompt,
                    "tactic": tac,
                    "completion": tac,
                    "theorem": full,
                    "theorem_set": set_name,
                    "origin": origin,
                    "source_run": Path(p).parent.name,
                    "state_hash": hash_state(state),
                    "tactic_hash": hash_tactic(tac),
                    "namespace": "Int",
                    "role": "close",
                    "assist_distance": None,
                    "skeleton_stable_id": r.get("skeleton_stable_id"),
                    "skeleton_name": r.get("skeleton_name"),
                    "skeleton_shape": r.get("skeleton_shape"),
                    "skeleton_family": r.get("skeleton_family"),
                    "wrapper_only": True,
                    "first_seen_arc": arc,
                    "_variant": "ns22",
                    "_prompt_style": "vanilla",
                })
                seen.add(full)
            if len(seen) == len(want_theorems):
                break
        if len(seen) == len(want_theorems):
            break
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build(
    *,
    variant_name: str,
    pool_rows: list[dict],
    pool_family_label: str,
    replay_rows: list[dict],
    oversample: int,
    out_path: Path,
    seed: int = 42,
) -> dict:
    out_rows: list[dict] = []
    for i in range(oversample):
        for r in pool_rows:
            rr = dict(r)
            rr["_oversample_idx"] = i
            out_rows.append(rr)
    out_rows.extend(replay_rows)

    rng = random.Random(seed)
    rng.shuffle(out_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_namespace: dict[str, int] = defaultdict(int)
    by_tactic_kind: dict[str, int] = defaultdict(int)
    pool_thms: set[str] = set()
    n_pool_rows = 0
    for r in out_rows:
        by_namespace[r.get("namespace") or "?"] += 1
        t = (r.get("tactic") or "")
        if "fun h => by omega" in t and t.count("by omega") >= 2:
            by_tactic_kind["iff_omega_pair"] += 1
        elif t == "omega":
            by_tactic_kind["fallback_omega"] += 1
        else:
            by_tactic_kind["other"] += 1
        if r.get("_variant") == "ns22":
            n_pool_rows += 1
            pool_thms.add(r["theorem"])

    meta = {
        "variant": variant_name,
        "pool_family_label": pool_family_label,
        "out_path": str(out_path),
        "n_rows": len(out_rows),
        "n_pool_rows": n_pool_rows,
        "n_pool_unique_theorems": len(pool_thms),
        "n_pool_source_rows": len(pool_rows),
        "oversample_factor": oversample,
        "n_replay_rows": len(replay_rows),
        "replay_source": str(NS12_BALANCED_PATH),
        "init_from_recommended": "project/models/gen_v5_ns12_balanced",
        "hard_exclusions": (
            "no fallback_omega mixed into iff_omega corpus; "
            "no Nat simp_all rows; no Set/Finset wrapper-only rows"
        ),
        "by_namespace": dict(by_namespace),
        "by_tactic_kind": dict(by_tactic_kind),
        "pool_theorems": sorted(pool_thms),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", required=True,
        choices=["iff_5x", "iff_10x", "omega_5x", "all"],
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    iff_thms, omega_thms = load_pool_theorems()
    print(f"iff_omega pool: {len(iff_thms)} theorems")
    print(f"fallback_omega pool: {len(omega_thms)} theorems")

    iff_rows = extract_close_rows(iff_thms, "fun h => by omega")
    omega_rows = extract_close_rows(omega_thms, "")  # any close row
    # Filter omega_rows to exact "omega" only (not nested iff calls).
    omega_rows = [r for r in omega_rows if r["tactic"].strip() == "omega"]
    print(f"iff close-rows extracted: {len(iff_rows)}")
    print(f"omega close-rows extracted: {len(omega_rows)}")
    if len(iff_rows) != len(iff_thms):
        print(f"WARNING: iff missing "
              f"{iff_thms - {r['theorem'] for r in iff_rows}}")
    if len(omega_rows) != len(omega_thms):
        print(f"WARNING: omega missing "
              f"{omega_thms - {r['theorem'] for r in omega_rows}}")

    if not NS12_BALANCED_PATH.exists():
        raise SystemExit(f"missing {NS12_BALANCED_PATH}")
    replay_full = load_jsonl(NS12_BALANCED_PATH)

    todo = (["iff_5x", "iff_10x", "omega_5x"]
            if args.variant == "all" else [args.variant])
    for v in todo:
        if v == "iff_5x":
            out = Path("project/data/ns22_int_iff_omega_5x.jsonl")
            meta = build(
                variant_name=v, pool_rows=iff_rows,
                pool_family_label="iff_omega_pair",
                replay_rows=list(replay_full),
                oversample=5, out_path=out, seed=args.seed,
            )
        elif v == "iff_10x":
            out = Path("project/data/ns22_int_iff_omega_10x.jsonl")
            meta = build(
                variant_name=v, pool_rows=iff_rows,
                pool_family_label="iff_omega_pair",
                replay_rows=list(replay_full),
                oversample=10, out_path=out, seed=args.seed,
            )
        elif v == "omega_5x":
            out = Path("project/data/ns22_int_fallback_omega_5x.jsonl")
            meta = build(
                variant_name=v, pool_rows=omega_rows,
                pool_family_label="fallback_omega",
                replay_rows=list(replay_full),
                oversample=5, out_path=out, seed=args.seed,
            )
        else:
            raise SystemExit(f"unknown variant {v}")
        print(f"\n=== variant {v} ===")
        print(f"out                = {meta['out_path']}")
        print(f"total rows         = {meta['n_rows']}")
        print(f"pool rows          = {meta['n_pool_rows']}")
        print(f"pool unique thms   = {meta['n_pool_unique_theorems']}")
        print(f"by_tactic_kind     = {meta['by_tactic_kind']}")


if __name__ == "__main__":
    main()
