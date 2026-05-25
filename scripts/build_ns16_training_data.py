"""NS16 Stage 3 — wrapper-only Nat trace-to-training extraction.

Walks the wrapper eval traces produced in NS15 Stage 6 + NS16
Stage 2 (``project/evolve/eval_runs/{gen_v5_ns15_routed,ns16_ns15routed}_wrapper_*``)
and extracts supervised (state, tactic) pairs.

A pair is *wrapper-only* if:
  - the theorem was proved by the wrapper variant, AND
  - the corresponding raw variant did NOT prove it.

These are the patterns the raw NS15 model has not yet learned to
emit natively. Training on them is the NS16 hypothesis: 10×–20×
oversampling reproduces the NS15-style transfer on wrapper-only
Nat wins.

The script also accepts a ``--include-all-wrapper`` flag which
keeps wrapper-template emissions (priority_template, family_tactic,
tactic_template, etc.) from any wrapper-successful trace, not just
wrapper-only. This gives a larger corpus but at the risk of
training the model on patterns it already emits.

Outputs:

  - ``project/data/ns16_nat_wrapper_only.jsonl`` (gitignored)
  - ``project/data/ns16_nat_wrapper_only_meta.json`` (committed)

Usage::

    python scripts/build_ns16_training_data.py \\
        --out project/data/ns16_nat_wrapper_only.jsonl
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path


MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500

DEFAULT_ALLOWED_ORIGINS = {
    "fallback_tactic", "family_tactic", "generative_topk",
    "term_builder", "tactic_template", "priority_template",
    "retrieved_premise", "skeleton_emitted",
}

# Map (set_name, raw_metrics_glob, wrapper_metrics_glob,
# wrapper_traces_glob) — these are the (raw, wrapper) eval-run
# pairs to mine. Wrapper traces are read; raw is only used to
# identify wrapper-only theorems.
EVAL_PAIRS = [
    {
        "set": "nat_defs_medium",
        "raw": "project/evolve/eval_runs/gen_v5_ns15_routed_raw_nat_defs_medium/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_medium/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_medium/eval-*/traces.jsonl",
    },
    {
        "set": "nat_defs_large_v5",
        "raw": "project/evolve/eval_runs/gen_v5_ns15_routed_raw_nat_defs_large_v5/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_large_v5/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_nat_defs_large_v5/eval-*/traces.jsonl",
    },
    {
        "set": "ns16_nat_iff_extra",
        "raw": "project/evolve/eval_runs/ns16_ns15routed_raw_ns16_nat_iff_extra/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_iff_extra/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_iff_extra/eval-*/traces.jsonl",
    },
    {
        "set": "ns16_nat_div_mod_extra",
        "raw": "project/evolve/eval_runs/ns16_ns15routed_raw_ns16_nat_div_mod_extra/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_div_mod_extra/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_div_mod_extra/eval-*/traces.jsonl",
    },
    {
        "set": "ns16_nat_order_extra",
        "raw": "project/evolve/eval_runs/ns16_ns15routed_raw_ns16_nat_order_extra/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_order_extra/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_order_extra/eval-*/traces.jsonl",
    },
    {
        "set": "ns16_nat_mixed_extra",
        "raw": "project/evolve/eval_runs/ns16_ns15routed_raw_ns16_nat_mixed_extra/eval-*/metrics.json",
        "wrap_metrics": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_mixed_extra/eval-*/metrics.json",
        "wrap_traces": "project/evolve/eval_runs/ns16_ns15routed_wrapper_ns16_nat_mixed_extra/eval-*/traces.jsonl",
    },
]

# NS11 held-out theorems — flagged in the meta because training on
# them contaminates the held-out eval signal. The user gets to
# choose what to do (NS15 already trained on some of these via
# NS14, so the honest-eval guarantee is already eroded).
NS11_HELD_OUT = {
    "Nat.AM_GM", "Nat.div_lt_iff_lt_mul'", "Nat.div_lt_one_iff",
    "Nat.div_pos", "Nat.div_pos_iff", "Nat.mul_eq_left",
    "Nat.mul_eq_right", "Nat.dvd_iff_div_mul_eq", "Nat.sqrt_lt",
    "Nat.pow_lt_pow_iff_left",
}


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
    if full_name and full_name in tactic:
        return True
    return False


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


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_solved(metrics_path: str) -> set[str]:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", [])
            if t.get("finished")}


def hash_state(state_pp: str) -> str:
    return hashlib.sha1(state_pp.encode("utf-8")).hexdigest()[:16]


def hash_tactic(tactic: str) -> str:
    return hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12]


def extract_episodes(trace_path: Path) -> list[list[dict]]:
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
    wrapper_only_theorems: set[str],
    eval_set_name: str,
    include_all_wrapper: bool,
    k_assist_window: int = 3,
) -> list[dict]:
    rows: list[dict] = []
    eps = extract_episodes(trace_path)
    for ep in eps:
        if not ep:
            continue
        full_name = ep[0].get("full_name") or "?"
        if full_name not in wrapper_only_theorems and not include_all_wrapper:
            continue

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
                after = [j for j in accepted_idx if j > i and j <= close_idx]
                if 1 <= len(after) <= k_assist_window:
                    role = "advance_assist"
                    assist_distance = len(after)
            if role is None:
                continue

            state_pp = r.get("state_pp") or r.get("state_pp_before") or ""
            tactic = r.get("tactic") or ""
            full = r.get("full_name") or full_name
            if not state_pp or len(state_pp) > MAX_STATE_LEN:
                continue
            if looks_bad_tactic(tactic, full):
                continue
            origin = r.get("tactic_origin") or "unknown"
            if origin not in DEFAULT_ALLOWED_ORIGINS:
                continue

            prompt = f"Theorem: {full}\n\nProof state:\n{state_pp}\n"
            row = {
                "prompt": prompt,
                "tactic": tactic,
                "completion": tactic,
                "theorem": full,
                "theorem_set": eval_set_name,
                "origin": origin,
                "source_run": trace_path.parent.name,
                "state_hash": hash_state(state_pp),
                "tactic_hash": hash_tactic(tactic),
                "namespace": "Nat",
                "role": role,
                "assist_distance": assist_distance,
                "skeleton_stable_id": r.get("skeleton_stable_id"),
                "skeleton_name": r.get("skeleton_name"),
                "skeleton_shape": r.get("skeleton_shape"),
                "skeleton_family": r.get("skeleton_family"),
                "wrapper_only": full in wrapper_only_theorems,
                "is_ns11_heldout": full in NS11_HELD_OUT,
                "_variant": "ns16",
                "_prompt_style": "vanilla",
            }
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include-all-wrapper", action="store_true",
                    help="Mine every wrapper-successful theorem's trace, "
                         "not just wrapper-only theorems.")
    ap.add_argument("--k-assist-window", type=int, default=3)
    args = ap.parse_args()

    all_rows: list[dict] = []
    per_set_meta: dict[str, dict] = {}

    for cfg in EVAL_PAIRS:
        set_name = cfg["set"]
        raw_path = first_match(cfg["raw"])
        wrap_path = first_match(cfg["wrap_metrics"])
        if not raw_path or not wrap_path:
            per_set_meta[set_name] = {"status": "missing", "raw": raw_path,
                                       "wrap": wrap_path}
            continue
        raw_solved = load_solved(raw_path)
        wrap_solved = load_solved(wrap_path)
        wrapper_only = wrap_solved - raw_solved

        trace_path_str = first_match(cfg["wrap_traces"])
        if not trace_path_str:
            per_set_meta[set_name] = {
                "status": "no_traces",
                "wrapper_only_theorems": sorted(wrapper_only),
            }
            continue
        rows = extract_rows(
            Path(trace_path_str),
            wrapper_only_theorems=wrapper_only,
            eval_set_name=set_name,
            include_all_wrapper=args.include_all_wrapper,
            k_assist_window=args.k_assist_window,
        )
        per_set_meta[set_name] = {
            "raw_solved": len(raw_solved),
            "wrap_solved": len(wrap_solved),
            "wrapper_only_count": len(wrapper_only),
            "wrapper_only_theorems": sorted(wrapper_only),
            "rows_extracted": len(rows),
        }
        all_rows.extend(rows)

    # Dedup by (state_hash, tactic_hash); keep first occurrence.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    n_dup = 0
    for r in all_rows:
        key = (r["state_hash"], r["tactic_hash"])
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        deduped.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_role: dict[str, int] = defaultdict(int)
    by_origin: dict[str, int] = defaultdict(int)
    by_set: dict[str, int] = defaultdict(int)
    thms: set[str] = set()
    n_wrapper_only_rows = 0
    n_heldout_rows = 0
    for r in deduped:
        by_role[r["role"]] += 1
        by_origin[r["origin"]] += 1
        by_set[r["theorem_set"]] += 1
        thms.add(r["theorem"])
        if r["wrapper_only"]:
            n_wrapper_only_rows += 1
        if r["is_ns11_heldout"]:
            n_heldout_rows += 1

    meta = {
        "include_all_wrapper": args.include_all_wrapper,
        "k_assist_window": args.k_assist_window,
        "total_rows_pre_dedup": len(all_rows),
        "total_rows_post_dedup": len(deduped),
        "n_dup_dropped": n_dup,
        "unique_theorems": len(thms),
        "by_role": dict(by_role),
        "by_origin": dict(by_origin),
        "by_set": dict(by_set),
        "n_wrapper_only_rows": n_wrapper_only_rows,
        "n_ns11_heldout_rows": n_heldout_rows,
        "per_set": per_set_meta,
    }
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"out                 = {args.out}")
    print(f"rows pre-dedup      = {len(all_rows)}")
    print(f"rows post-dedup     = {len(deduped)}")
    print(f"unique theorems     = {len(thms)}")
    print(f"wrapper-only rows   = {n_wrapper_only_rows}")
    print(f"ns11_held_out rows  = {n_heldout_rows}")
    print(f"by role             = {dict(by_role)}")
    print(f"by origin           = {dict(by_origin)}")
    print(f"by set              = {dict(by_set)}")


if __name__ == "__main__":
    main()
