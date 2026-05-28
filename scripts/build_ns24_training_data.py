"""NS24 — Int minimal-sufficient omega aggregate imitation training data.

NS23 (`project/data/ns23_minimal_family_pools_meta.json`) repaired the
wrapper-attributed labels by re-running every wrapper-only-vs-NS9 win
through a minimal-tactic battery. The Int `omega_aggregate` pool unifies
`fallback_omega + iff_omega_pair + constructor_omega + split_ifs_omega`
under their *minimal* labels:

  - 21 theorems are `omega`-minimal (12 originally fallback_omega + 9
    relabeled from iff_omega_pair).
  - 1 theorem (`Int.zero_le_ofNat`) is `constructor <;> omega`-minimal
    (originally fallback_omega; the wrapper closed it with `omega` only
    after a coercion-normalising lead-in step, so plain `omega` does not
    close it from the *initial* state).
  - Total: 22 unique (the `Int.lt_toNat` iff outlier is unresolved and
    excluded).

The NS24 hypothesis: training on the **shortest sufficient tactic**
(minimal label) beats training on the wrapper-attributed tactic. We
build from the repaired labels, not the wrapper tactic.

Key construction detail vs NS22: NS24 pairs each theorem's INITIAL
proof state (the minimum-`step` `state_pp` in the wrapper trace) with
its NS23 minimal tactic — NOT the wrapper close-row state. This matters
for `Int.zero_le_ofNat`, whose wrapper close happened at a post-lead-in
subgoal; the model at inference time sees the initial state.

Variants:
  A. ns24_int_minimal_omega_5x      — 21 omega rows x5  + NS12 replay  (init: ns22)
  B. ns24_int_minimal_omega_10x     — 21 omega rows x10 + NS12 replay  (init: ns22)
  C. ns24_int_minimal_omega_plus_constructor_5x
                                    — 21 omega + 1 constructor_omega x5 + replay (init: ns22)
  D. ns24_int_minimal_omega_5x_from_ns12 (optional ablation)
                                    — same rows as A; init from ns12_balanced
                                      (dataset identical to A; only --model differs)

Outputs (committed metas; JSONLs gitignored):
  project/data/ns24_int_minimal_omega_5x_meta.json / .jsonl
  project/data/ns24_int_minimal_omega_10x_meta.json / .jsonl
  project/data/ns24_int_minimal_omega_plus_constructor_5x_meta.json / .jsonl
  project/data/ns24_int_minimal_omega_5x_from_ns12_meta.json  (points at A's jsonl)
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

POOL_META = Path("project/data/ns23_minimal_family_pools_meta.json")
NS12_BALANCED_PATH = Path("project/data/ns12_train_balanced.jsonl")

# All Int NS9-wrapper eval-run trace globs that hold the pool theorems.
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


def load_omega_aggregate_pool() -> list[dict]:
    """Return the 22 NS23 omega_aggregate/Int pool entries."""
    meta = json.load(open(POOL_META))
    pool = meta["omega_aggregate_by_namespace"]["Int"]["theorems"]
    return pool


def extract_initial_states(want: set[str]) -> dict[str, dict]:
    """Map theorem -> {state_pp, step, source_run} for the INITIAL state.

    The initial state is the minimum-`step` trace row for the theorem.
    For single-step proofs this equals the close-row state; for
    multi-step proofs (e.g. Int.zero_le_ofNat) it is the true root state
    the model sees at inference time.
    """
    best: dict[str, dict] = {}
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
                fn = r.get("full_name")
                if fn not in want:
                    continue
                step = r.get("step")
                state = r.get("state_pp")
                if step is None or not state:
                    continue
                if len(state) > MAX_STATE_LEN:
                    continue
                cur = best.get(fn)
                if cur is None or step < cur["step"]:
                    best[fn] = {
                        "state_pp": state,
                        "step": step,
                        "source_run": Path(p).parent.name,
                        "first_seen_arc": arc,
                    }
    return best


def make_row(entry: dict, init: dict, target_tactic: str) -> dict:
    full = entry["theorem"]
    state = init["state_pp"]
    prompt = f"Theorem: {full}\n\nProof state:\n{state}\n"
    orig = entry["original_family"]
    return {
        "prompt": prompt,
        "tactic": target_tactic,
        "completion": target_tactic,
        "theorem": full,
        "namespace": "Int",
        "minimal_tactic": entry["minimal_tactic"],
        "minimal_family": (
            "constructor_omega" if entry["minimal_tactic"].startswith("constructor")
            else "fallback_omega"
        ),
        "original_family": orig,
        "wrapper_tactic": entry["wrapper_tactic"],
        "first_seen_arc": entry.get("first_seen_arc") or init["first_seen_arc"],
        "in_ns22_omega_training": (orig == "fallback_omega"),
        "relabeled_from_iff": (orig == "iff_omega_pair"),
        "source_run": init["source_run"],
        "init_step": init["step"],
        "state_hash": hash_state(state),
        "tactic_hash": hash_tactic(target_tactic),
        "role": "close",
        "wrapper_only": True,
        "_variant": "ns24",
        "_prompt_style": "vanilla",
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build(
    *,
    variant_name: str,
    pool_rows: list[dict],
    replay_rows: list[dict],
    oversample: int,
    out_path: Path,
    init_from: str,
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
    by_target: dict[str, int] = defaultdict(int)
    by_source_family: dict[str, int] = defaultdict(int)
    pool_thms: set[str] = set()
    n_pool_rows = 0
    n_in_ns22 = 0
    n_relabeled = 0
    for r in out_rows:
        by_namespace[r.get("namespace") or "?"] += 1
        if r.get("_variant") == "ns24":
            n_pool_rows += 1
            by_target[r["tactic"]] += 1
            by_source_family[r["original_family"]] += 1
            pool_thms.add(r["theorem"])
            if r.get("in_ns22_omega_training"):
                n_in_ns22 += 1
            if r.get("relabeled_from_iff"):
                n_relabeled += 1

    # Per-unique-theorem provenance (one entry per theorem).
    thm_prov = {}
    for r in pool_rows:
        thm_prov[r["theorem"]] = {
            "minimal_tactic": r["minimal_tactic"],
            "original_family": r["original_family"],
            "first_seen_arc": r["first_seen_arc"],
            "in_ns22_omega_training": r["in_ns22_omega_training"],
            "relabeled_from_iff": r["relabeled_from_iff"],
        }

    meta = {
        "variant": variant_name,
        "out_path": str(out_path),
        "init_from": init_from,
        "n_rows": len(out_rows),
        "n_pool_rows_after_oversample": n_pool_rows,
        "n_pool_unique_theorems": len(pool_thms),
        "n_pool_source_rows_before_oversample": len(pool_rows),
        "oversample_factor": oversample,
        "n_replay_rows": len(replay_rows),
        "replay_source": str(NS12_BALANCED_PATH),
        "int_omega_rows_before_oversample": len(pool_rows),
        "int_omega_rows_after_oversample": n_pool_rows,
        "target_tactic_distribution": dict(by_target),
        "by_namespace": dict(by_namespace),
        "source_original_family_rows": dict(by_source_family),
        "n_rows_in_ns22_omega_training": n_in_ns22,
        "n_rows_relabeled_from_iff": n_relabeled,
        "label_source": "ns23 minimal-sufficient tactic (NOT wrapper-attributed)",
        "state_source": "initial (min-step) state_pp from NS9-wrapper Int traces",
        "contamination_risk": (
            "none: only Int omega-aggregate pool rows + NS12 balanced replay; "
            "no Nat/Set/Finset wrapper-only rows; no long iff-pair tactic targets"
        ),
        "per_theorem_provenance": thm_prov,
        "pool_theorems": sorted(pool_thms),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", default="all",
        choices=["omega_5x", "omega_10x", "plus_constructor_5x",
                 "from_ns12", "all"],
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_omega_aggregate_pool()
    want = {e["theorem"] for e in pool}
    print(f"omega_aggregate/Int pool: {len(pool)} theorems")

    inits = extract_initial_states(want)
    missing = want - set(inits)
    if missing:
        raise SystemExit(f"FATAL: no initial state found for {sorted(missing)}")
    print(f"initial states extracted for all {len(inits)} theorems")

    # Split pool into omega-minimal and constructor-minimal.
    omega_rows: list[dict] = []
    constructor_rows: list[dict] = []
    for e in pool:
        init = inits[e["theorem"]]
        mt = e["minimal_tactic"]
        if mt == "omega":
            omega_rows.append(make_row(e, init, "omega"))
        elif mt.startswith("constructor"):
            constructor_rows.append(make_row(e, init, mt))
        else:
            raise SystemExit(f"unexpected minimal_tactic {mt!r} for {e['theorem']}")
    print(f"omega-minimal rows: {len(omega_rows)}; "
          f"constructor-minimal rows: {len(constructor_rows)}")

    if not NS12_BALANCED_PATH.exists():
        raise SystemExit(f"missing {NS12_BALANCED_PATH}")
    replay_full = load_jsonl(NS12_BALANCED_PATH)
    print(f"replay rows: {len(replay_full)}")

    INIT_NS22 = "project/models/gen_v5_ns22_int_fallback_omega_5x"
    INIT_NS12 = "project/models/gen_v5_ns12_balanced"

    todo = (["omega_5x", "omega_10x", "plus_constructor_5x", "from_ns12"]
            if args.variant == "all" else [args.variant])

    for v in todo:
        if v == "omega_5x":
            out = Path("project/data/ns24_int_minimal_omega_5x.jsonl")
            meta = build(variant_name=v, pool_rows=omega_rows,
                         replay_rows=list(replay_full), oversample=5,
                         out_path=out, init_from=INIT_NS22, seed=args.seed)
        elif v == "omega_10x":
            out = Path("project/data/ns24_int_minimal_omega_10x.jsonl")
            meta = build(variant_name=v, pool_rows=omega_rows,
                         replay_rows=list(replay_full), oversample=10,
                         out_path=out, init_from=INIT_NS22, seed=args.seed)
        elif v == "plus_constructor_5x":
            out = Path("project/data/ns24_int_minimal_omega_plus_constructor_5x.jsonl")
            meta = build(variant_name=v, pool_rows=omega_rows + constructor_rows,
                         replay_rows=list(replay_full), oversample=5,
                         out_path=out, init_from=INIT_NS22, seed=args.seed)
        elif v == "from_ns12":
            # Dataset identical to omega_5x; only the init checkpoint differs.
            # Reuse the omega_5x JSONL and emit a meta recording the ns12 init.
            shared = Path("project/data/ns24_int_minimal_omega_5x.jsonl")
            if not shared.exists():
                # build it if omega_5x not yet emitted in this run
                build(variant_name="omega_5x", pool_rows=omega_rows,
                      replay_rows=list(replay_full), oversample=5,
                      out_path=shared, init_from=INIT_NS22, seed=args.seed)
            meta_path = Path("project/data/ns24_int_minimal_omega_5x_from_ns12_meta.json")
            meta = {
                "variant": "from_ns12",
                "out_path": str(shared),
                "init_from": INIT_NS12,
                "note": ("ablation: identical dataset to ns24_int_minimal_omega_5x; "
                         "trained from gen_v5_ns12_balanced instead of the NS22 "
                         "Int specialist, to isolate continued-training effect"),
                "oversample_factor": 5,
                "n_pool_unique_theorems": len(omega_rows),
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"\n=== variant from_ns12 ===")
            print(f"reuses {shared} with init {INIT_NS12}")
            continue
        else:
            raise SystemExit(f"unknown variant {v}")

        print(f"\n=== variant {v} ===")
        print(f"out                = {meta['out_path']}")
        print(f"init_from          = {meta['init_from']}")
        print(f"total rows         = {meta['n_rows']}")
        print(f"pool unique thms   = {meta['n_pool_unique_theorems']}")
        print(f"target dist        = {meta['target_tactic_distribution']}")
        print(f"in NS22 / relabel  = {meta['n_rows_in_ns22_omega_training']} / "
              f"{meta['n_rows_relabeled_from_iff']} (per-row, x{meta['oversample_factor']})")


if __name__ == "__main__":
    main()
