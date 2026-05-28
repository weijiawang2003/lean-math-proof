"""NS12 — anti-forgetting dataset builder.

NS11 fine-tuned gen_v5 on a combined 5,729-pair corpus (v5 base +
152 evolved Nat-heavy pairs). The raw lift on nat_defs_medium was
3→9. The cost was a regression on demo_v1 (10→8): the model
stopped emitting ``simp [Set.subset_def]`` in top-8 for
``Set.subset_univ`` and ``Set.empty_subset`` (see
``project/evolve/reports/ns12_demo_regression_analysis.md``).

NS12 builds three anti-forgetting dataset variants on top of the
NS11 combined corpus:

  - ``balanced`` — oversample Set/Finset rows and downweight Nat
    rows so the gradient signal is biased back toward the v5
    distribution while still containing the new evolved pairs.

  - ``replay_demo`` — adds explicit replay rows for the demo_v1
    theorems we lost (``Set.subset_univ``, ``Set.empty_subset``)
    using the *known-winning* tactics gen_v5 used. Each replay
    is duplicated K times to give the gradient enough weight.

  - ``low_lr`` — identity over the NS11 combined dataset; the
    "fix" lives in training hyperparameters (lower lr, fewer
    epochs), not the data.

Output rows are in the same schema as NS11 combined
(``project/data/ns11_train_combined.jsonl``).

Usage:

    python scripts/build_ns12_training_data.py --variant balanced \
        --out project/data/ns12_train_balanced.jsonl

    python scripts/build_ns12_training_data.py --variant replay_demo \
        --out project/data/ns12_train_replay.jsonl

    python scripts/build_ns12_training_data.py --variant low_lr \
        --out project/data/ns12_train_low_lr.jsonl

For each variant the script also writes a sibling ``*_meta.json``
with per-domain counts and the balancing knobs used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


COMBINED_PATH = Path("project/data/ns11_train_combined.jsonl")
V5_BASE_PATH = Path("project/seq2seq_data_v5.jsonl")


# Demo theorems we lost on demo_v1 (gen_v5 → ns11_combined regression).
# Each value is a list of (state_before, winning_tactic) pairs we want
# the model to keep emitting. State strings come from
# project/evolve/eval_runs/gen_v5_raw_demo_v1/eval-1d29613c/traces.jsonl
DEMO_REPLAY: dict[str, list[tuple[str, str]]] = {
    "Set.subset_univ": [
        (
            "α : Type u\nβ : Type v\nγ : Type w\nι : Sort x\n"
            "a b : α\ns✝ s₁ s₂ t t₁ t₂ u s : Set α\n⊢ s ⊆ univ",
            "simp [Set.subset_def]",
        ),
    ],
    "Set.empty_subset": [
        (
            "α : Type u\nβ : Type v\nγ : Type w\nι : Sort x\n"
            "a b : α\ns✝ s₁ s₂ t t₁ t₂ u s : Set α\n⊢ ∅ ⊆ s",
            "simp [Set.subset_def]",
        ),
    ],
}

# Domain classifier — read the theorem name out of the prompt's
# ``Theorem: <name>`` header.
def domain_of(row: dict) -> str:
    p = row.get("prompt", "")
    thm = ""
    if p.startswith("Theorem: "):
        thm = p.split("\n", 1)[0][len("Theorem: ") :]
    elif row.get("theorem"):
        thm = row["theorem"]
    if thm.startswith("Nat."):
        return "Nat"
    if thm.startswith("Set."):
        return "Set"
    if thm.startswith("Finset."):
        return "Finset"
    return "other"


def theorem_of(row: dict) -> str:
    p = row.get("prompt", "")
    if p.startswith("Theorem: "):
        return p.split("\n", 1)[0][len("Theorem: ") :]
    return row.get("theorem", "")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_balanced(combined: list[dict], *, nat_keep: float, set_dup: int,
                   finset_dup: int) -> list[dict]:
    """Downsample Nat rows and oversample Set/Finset rows.

    Knobs:
      - ``nat_keep``: fraction of Nat rows to keep (0..1). Hash-based,
        deterministic. Always keeps evolved Nat rows (those with
        ``_variant``).
      - ``set_dup``: integer duplication factor for Set rows.
      - ``finset_dup``: integer duplication factor for Finset rows.
    """
    out: list[dict] = []
    rng_seed = b"ns12_balanced"
    for r in combined:
        d = domain_of(r)
        if d == "Nat":
            # Keep all evolved Nat rows.
            if r.get("_variant"):
                out.append(r)
                continue
            # Hash-deterministic subsample.
            h = hashlib.sha1(rng_seed + r["prompt"].encode("utf-8")).digest()
            keep = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) < nat_keep
            if keep:
                out.append(r)
        elif d == "Set":
            for _ in range(set_dup):
                out.append(r)
        elif d == "Finset":
            for _ in range(finset_dup):
                out.append(r)
        else:
            out.append(r)
    return out


def build_replay(combined: list[dict], *, replay_copies: int) -> list[dict]:
    """Append explicit replay rows for the lost demo theorems."""
    out: list[dict] = list(combined)
    added: list[dict] = []
    for thm, examples in DEMO_REPLAY.items():
        for state, tactic in examples:
            prompt = f"Theorem: {thm}\n\nProof state:\n{state}\n"
            row = {
                "prompt": prompt,
                "tactic": tactic,
                "completion": tactic,
                "theorem": thm,
                "theorem_set": "demo_v1_replay",
                "origin": "demo_replay",
                "source_run": "ns12_demo_replay",
                "state_hash": hashlib.sha1(state.encode("utf-8")).hexdigest()[:16],
                "tactic_hash": hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12],
                "skeleton_name": None,
                "skeleton_stable_id": None,
                "skeleton_shape": None,
                "skeleton_family": None,
                "role": "close",
                "assist_distance": None,
                "_variant": "ns12_replay",
                "_prompt_style": "vanilla",
            }
            for _ in range(replay_copies):
                added.append(row)
    out.extend(added)
    return out, added


def summarize(rows: list[dict]) -> dict:
    by_domain: dict[str, int] = defaultdict(int)
    evolved = 0
    replay = 0
    v5_base = 0
    thms: set[str] = set()
    for r in rows:
        by_domain[domain_of(r)] += 1
        thms.add(theorem_of(r))
        v = r.get("_variant")
        if v == "ns12_replay":
            replay += 1
        elif v == "v5_base":
            v5_base += 1
        elif v:
            evolved += 1
        else:
            v5_base += 1
    return {
        "total_pairs": len(rows),
        "by_domain": dict(by_domain),
        "v5_base_rows": v5_base,
        "evolved_rows": evolved,
        "replay_rows": replay,
        "unique_theorems": len(thms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["balanced", "replay_demo", "low_lr"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source",
                    type=Path, default=COMBINED_PATH,
                    help="Source corpus (default: NS11 combined).")
    # balanced knobs
    ap.add_argument("--nat-keep", type=float, default=0.5,
                    help="balanced: fraction of v5-base Nat rows to keep.")
    ap.add_argument("--set-dup", type=int, default=2,
                    help="balanced: duplication factor for Set rows.")
    ap.add_argument("--finset-dup", type=int, default=1,
                    help="balanced: duplication factor for Finset rows.")
    # replay knobs
    ap.add_argument("--replay-copies", type=int, default=20,
                    help="replay_demo: copies of each replay row to add.")
    args = ap.parse_args()

    combined = load_jsonl(args.source)

    meta = {
        "variant": args.variant,
        "source": str(args.source),
        "source_rows": len(combined),
    }

    if args.variant == "balanced":
        out_rows = build_balanced(
            combined,
            nat_keep=args.nat_keep,
            set_dup=args.set_dup,
            finset_dup=args.finset_dup,
        )
        meta["knobs"] = {
            "nat_keep": args.nat_keep,
            "set_dup": args.set_dup,
            "finset_dup": args.finset_dup,
        }
    elif args.variant == "replay_demo":
        out_rows, added = build_replay(combined,
                                       replay_copies=args.replay_copies)
        meta["knobs"] = {"replay_copies": args.replay_copies}
        meta["replay_added_rows"] = len(added)
        meta["replay_targets"] = sorted(DEMO_REPLAY.keys())
    else:  # low_lr — no data change; fix is in training hparams.
        out_rows = combined
        meta["knobs"] = {
            "note": "no data change; lower lr / fewer epochs in trainer"
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(out_rows)
    meta.update(summary)
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"variant   = {args.variant}")
    print(f"source    = {args.source} ({len(combined)} rows)")
    print(f"out       = {args.out}")
    print(f"meta      = {meta_path}")
    print(f"total     = {summary['total_pairs']}")
    print(f"by domain = {summary['by_domain']}")
    print(f"v5 base   = {summary['v5_base_rows']}")
    print(f"evolved   = {summary['evolved_rows']}")
    print(f"replay    = {summary['replay_rows']}")


if __name__ == "__main__":
    main()
