"""NS15 — wider-corpus training-data builder.

Stage 1 of the NS15 plan. Builds four dataset variants by merging
the v5 base corpus with the NS11-evolved pairs and the NS14
fresh-surface pairs:

  - ``combined_all``        — v5 base + NS11 evolved + NS14 evolved,
                              deduplicated by (state_hash, tactic_hash).
                              Baseline NS15 corpus.

  - ``nat_oversample``      — ``combined_all`` plus extra copies of
                              the NS14 wrapper-only Nat rows
                              (``tactic_template`` iff-omega pattern,
                              ``fallback_tactic`` omega). Goal: teach
                              the raw model the patterns that NS14
                              showed it never learned.

  - ``balanced_namespace``  — namespace-balanced (Nat/Set/Finset)
                              version of ``combined_all`` using the
                              NS12 trick (hash-deterministic Nat
                              subsample + Set/Finset duplication),
                              plus the NS12 demo_replay rows so the
                              model retains ``simp [Set.subset_def]``
                              on ``Set.subset_univ`` / ``Set.empty_subset``.

  - ``curriculum``          — re-emits the combined corpus in two
                              ordered "stages": stage-1 = v5 base,
                              stage-2 = NS11 + NS14 evolved (with
                              mild Nat oversampling). The training
                              script does single-pass shuffling so
                              the staged ordering is documentation
                              only; this is what we have until
                              ``train_tactic_generator.py`` learns
                              about phased fine-tunes.

Inputs:

  - ``project/seq2seq_data_v5.jsonl`` (5,577 rows, no metadata)
  - ``project/data/ns11_train_combined.jsonl`` (5,729 rows = v5 + 152)
  - ``project/data/ns14_train_combined.jsonl`` (30 rows, fresh)

The v5 rows in the NS11 combined corpus are tagged with
``_variant="v5_base"``; the NS11 evolved rows are tagged
``_variant in {"medium","conservative","coverage"}``. We can
therefore separate the three buckets by ``_variant`` and source.

Usage::

    python scripts/build_ns15_training_data.py --variant combined_all \\
        --out project/data/ns15_combined_all.jsonl

    python scripts/build_ns15_training_data.py --variant nat_oversample \\
        --out project/data/ns15_nat_oversample.jsonl --nat-oversample 10

    python scripts/build_ns15_training_data.py --variant balanced_namespace \\
        --out project/data/ns15_balanced_namespace.jsonl

    python scripts/build_ns15_training_data.py --variant curriculum \\
        --out project/data/ns15_curriculum.jsonl

Each invocation emits a sibling ``*_meta.json``. Only the metas
are committed; the JSONL data is .gitignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

V5_BASE_PATH = Path("project/seq2seq_data_v5.jsonl")
NS11_COMBINED_PATH = Path("project/data/ns11_train_combined.jsonl")
NS14_COMBINED_PATH = Path("project/data/ns14_train_combined.jsonl")

# Demo theorems we lost in the NS11 → NS12 regression. Replaying
# these tactic-state pairs in any NS15 variant keeps the model
# emitting ``simp [Set.subset_def]`` for ⊆-shaped Set goals.
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


def domain_of(row: dict) -> str:
    p = row.get("prompt", "")
    thm = ""
    if p.startswith("Theorem: "):
        thm = p.split("\n", 1)[0][len("Theorem: "):]
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
        return p.split("\n", 1)[0][len("Theorem: "):]
    return row.get("theorem", "")


def hash_state(state_pp: str) -> str:
    return hashlib.sha1(state_pp.encode("utf-8")).hexdigest()[:16]


def hash_tactic(tactic: str) -> str:
    return hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12]


def normalize_row(r: dict, *, default_variant: str) -> dict:
    """Guarantee state_hash + tactic_hash + _variant on every row."""
    out = dict(r)
    # v5 rows have only prompt/tactic — synthesize hashes.
    if "state_hash" not in out or out["state_hash"] is None:
        # Recover state from prompt body ("...Proof state:\n<state>\n").
        prompt = out.get("prompt", "")
        if "Proof state:\n" in prompt:
            state = prompt.split("Proof state:\n", 1)[1].rstrip("\n")
        else:
            state = prompt
        out["state_hash"] = hash_state(state)
    if "tactic_hash" not in out or out["tactic_hash"] is None:
        out["tactic_hash"] = hash_tactic(out.get("tactic", ""))
    if "_variant" not in out or out["_variant"] is None:
        out["_variant"] = default_variant
    out.setdefault("completion", out.get("tactic", ""))
    out.setdefault("_prompt_style", "vanilla")
    return out


def load_jsonl(path: Path, *, default_variant: str) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(normalize_row(json.loads(line), default_variant=default_variant))
    return rows


def split_ns11(combined_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """NS11 combined contains v5 rows (variant=v5_base) and evolved Nat rows.

    Return (v5_rows, ns11_evolved_rows).
    """
    v5_rows: list[dict] = []
    evolved: list[dict] = []
    for r in combined_rows:
        v = r.get("_variant")
        if v == "v5_base":
            v5_rows.append(r)
        else:
            evolved.append(r)
    return v5_rows, evolved


def dedup_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Dedup by (state_hash, tactic_hash); keep first occurrence."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["state_hash"], r["tactic_hash"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out, len(rows) - len(out)


def make_replay_rows(*, copies: int) -> list[dict]:
    """Demo_replay rows for Set.subset_univ / Set.empty_subset."""
    out: list[dict] = []
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
                "source_run": "ns15_demo_replay",
                "state_hash": hash_state(state),
                "tactic_hash": hash_tactic(tactic),
                "namespace": "Set",
                "role": "close",
                "assist_distance": None,
                "_variant": "ns15_replay",
                "_prompt_style": "vanilla",
            }
            for _ in range(copies):
                out.append(dict(row))
    return out


def is_wrapper_nat_pattern(r: dict) -> bool:
    """A wrapper-only NS14 Nat row: iff-omega or bare omega."""
    if domain_of(r) != "Nat":
        return False
    if r.get("_variant") != "ns14":
        return False
    t = (r.get("tactic") or "").strip()
    # The two patterns NS14 found wrapper-only Nat wins coming from.
    if "fun h => by omega" in t:
        return True
    if t == "omega":
        return True
    # The general family: any tactic_template / fallback origin NS14 Nat row.
    if r.get("origin") in {"tactic_template", "fallback_tactic", "family_tactic"}:
        return True
    return False


def build_combined_all(
    v5_rows: list[dict],
    ns11_evolved: list[dict],
    ns14_rows: list[dict],
) -> tuple[list[dict], dict]:
    all_rows = v5_rows + ns11_evolved + ns14_rows
    deduped, n_dropped = dedup_rows(all_rows)
    return deduped, {
        "n_v5": len(v5_rows),
        "n_ns11_evolved": len(ns11_evolved),
        "n_ns14": len(ns14_rows),
        "n_dup_dropped": n_dropped,
    }


def build_nat_oversample(
    v5_rows: list[dict],
    ns11_evolved: list[dict],
    ns14_rows: list[dict],
    *,
    nat_oversample: int,
) -> tuple[list[dict], dict]:
    """combined_all + extra copies of NS14 wrapper-only Nat rows."""
    base, base_meta = build_combined_all(v5_rows, ns11_evolved, ns14_rows)

    wrapper_nat_rows = [r for r in ns14_rows if is_wrapper_nat_pattern(r)]
    extras: list[dict] = []
    # We *deliberately* re-insert duplicates here (training set, not eval set).
    # Each extra copy gets a tweaked tactic_hash via a counter suffix so
    # dedup-by-(state,tactic) keeps the original + the oversample.
    # NOTE: For seq2seq SFT the duplicates are useful only if we DO NOT
    # dedup them again. We append them AFTER dedup of `base`.
    for _ in range(max(0, nat_oversample - 1)):
        for r in wrapper_nat_rows:
            extras.append(dict(r))
    out_rows = base + extras
    meta = dict(base_meta)
    meta.update({
        "nat_oversample_factor": nat_oversample,
        "n_wrapper_nat_source_rows": len(wrapper_nat_rows),
        "n_oversample_extras": len(extras),
    })
    return out_rows, meta


def build_balanced_namespace(
    v5_rows: list[dict],
    ns11_evolved: list[dict],
    ns14_rows: list[dict],
    *,
    nat_keep: float,
    set_dup: int,
    finset_dup: int,
    replay_copies: int,
) -> tuple[list[dict], dict]:
    """Balanced like NS12 + replay rows + NS14 mixed in.

    All ns11_evolved (Nat-heavy) rows are kept regardless of subsample.
    All NS14 rows are kept regardless of subsample.
    """
    base, base_meta = build_combined_all(v5_rows, ns11_evolved, ns14_rows)

    seed = b"ns15_balanced"
    out: list[dict] = []
    pre: dict[str, int] = defaultdict(int)
    for r in base:
        d = domain_of(r)
        pre[d] += 1
        # Keep evolved (ns11/ns14) and demo_replay rows unconditionally.
        is_protected = (
            r.get("_variant") in {"medium", "conservative", "coverage", "ns14"}
            or r.get("origin") == "demo_replay"
        )
        if d == "Nat":
            if is_protected:
                out.append(r)
                continue
            h = hashlib.sha1(seed + r["prompt"].encode("utf-8")).digest()
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

    replay = make_replay_rows(copies=replay_copies)
    out += replay

    meta = dict(base_meta)
    meta.update({
        "knobs": {
            "nat_keep": nat_keep,
            "set_dup": set_dup,
            "finset_dup": finset_dup,
            "replay_copies": replay_copies,
        },
        "pre_balance_by_domain": dict(pre),
        "n_replay_rows_added": len(replay),
    })
    return out, meta


def build_curriculum(
    v5_rows: list[dict],
    ns11_evolved: list[dict],
    ns14_rows: list[dict],
    *,
    nat_oversample: int,
) -> tuple[list[dict], dict]:
    """Stage-1 = v5 base; stage-2 = NS11 + NS14 evolved (mild Nat oversample).

    The trainer shuffles single-pass; the order matters as documentation
    only. The metadata records the staged composition so a future
    phased-finetune trainer can split on a marker.
    """
    deduped_v5, n_v5_dropped = dedup_rows(v5_rows)

    stage2_rows = ns11_evolved + ns14_rows
    wrapper_nat = [r for r in ns14_rows if is_wrapper_nat_pattern(r)]
    for _ in range(max(0, nat_oversample - 1)):
        for r in wrapper_nat:
            stage2_rows.append(dict(r))

    # Mark each row with curriculum_stage so the metadata is informative.
    s1_rows = []
    for r in deduped_v5:
        rr = dict(r)
        rr["_curriculum_stage"] = 1
        s1_rows.append(rr)
    s2_rows = []
    for r in stage2_rows:
        rr = dict(r)
        rr["_curriculum_stage"] = 2
        s2_rows.append(rr)

    out_rows = s1_rows + s2_rows
    meta = {
        "n_v5": len(deduped_v5),
        "n_v5_dropped_dup": n_v5_dropped,
        "n_ns11_evolved": len(ns11_evolved),
        "n_ns14": len(ns14_rows),
        "n_oversample_extras": len(stage2_rows) - len(ns11_evolved) - len(ns14_rows),
        "nat_oversample_factor": nat_oversample,
        "stage1_rows": len(s1_rows),
        "stage2_rows": len(s2_rows),
    }
    return out_rows, meta


def summarize(rows: list[dict]) -> dict:
    by_domain: dict[str, int] = defaultdict(int)
    by_origin: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    by_variant: dict[str, int] = defaultdict(int)
    thms: set[str] = set()
    n_close = 0
    n_advance = 0
    n_replay = 0
    n_wrapper_nat = 0
    for r in rows:
        by_domain[domain_of(r)] += 1
        by_origin[r.get("origin") or "unknown"] += 1
        by_role[r.get("role") or "unknown"] += 1
        by_variant[r.get("_variant") or "unknown"] += 1
        thms.add(theorem_of(r))
        role = r.get("role")
        if role == "close":
            n_close += 1
        elif role == "advance_assist":
            n_advance += 1
        if r.get("origin") == "demo_replay" or r.get("_variant") == "ns15_replay":
            n_replay += 1
        if is_wrapper_nat_pattern(r):
            n_wrapper_nat += 1
    return {
        "total_pairs": len(rows),
        "by_domain": dict(by_domain),
        "by_origin": dict(by_origin),
        "by_role": dict(by_role),
        "by_variant": dict(by_variant),
        "unique_theorems": len(thms),
        "n_close_role": n_close,
        "n_advance_role": n_advance,
        "n_replay_rows": n_replay,
        "n_wrapper_nat_pattern_rows": n_wrapper_nat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["combined_all", "nat_oversample",
                             "balanced_namespace", "curriculum"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--v5-path", type=Path, default=V5_BASE_PATH)
    ap.add_argument("--ns11-path", type=Path, default=NS11_COMBINED_PATH)
    ap.add_argument("--ns14-path", type=Path, default=NS14_COMBINED_PATH)
    # variant knobs
    ap.add_argument("--nat-oversample", type=int, default=10,
                    help="nat_oversample / curriculum: copies of each "
                         "wrapper-only NS14 Nat row.")
    ap.add_argument("--nat-keep", type=float, default=0.6,
                    help="balanced_namespace: fraction of *v5-base* Nat "
                         "rows to keep (evolved Nat rows always kept).")
    ap.add_argument("--set-dup", type=int, default=2)
    ap.add_argument("--finset-dup", type=int, default=1)
    ap.add_argument("--replay-copies", type=int, default=20,
                    help="balanced_namespace: copies of each demo_replay row.")
    args = ap.parse_args()

    # The NS11 combined corpus already contains the v5 base + the NS11
    # evolved rows. We use it as the single canonical source for both
    # buckets — load it once and split rather than re-loading v5 separately.
    ns11_combined = load_jsonl(args.ns11_path, default_variant="medium")
    v5_rows, ns11_evolved = split_ns11(ns11_combined)
    ns14_rows = load_jsonl(args.ns14_path, default_variant="ns14")

    if args.variant == "combined_all":
        out_rows, build_meta = build_combined_all(v5_rows, ns11_evolved, ns14_rows)
    elif args.variant == "nat_oversample":
        out_rows, build_meta = build_nat_oversample(
            v5_rows, ns11_evolved, ns14_rows,
            nat_oversample=args.nat_oversample,
        )
    elif args.variant == "balanced_namespace":
        out_rows, build_meta = build_balanced_namespace(
            v5_rows, ns11_evolved, ns14_rows,
            nat_keep=args.nat_keep,
            set_dup=args.set_dup,
            finset_dup=args.finset_dup,
            replay_copies=args.replay_copies,
        )
    elif args.variant == "curriculum":
        out_rows, build_meta = build_curriculum(
            v5_rows, ns11_evolved, ns14_rows,
            nat_oversample=args.nat_oversample,
        )
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(out_rows)
    meta = {
        "v": "ns15",
        "variant": args.variant,
        "out_path": str(args.out),
        "inputs": {
            "ns11_path": str(args.ns11_path),
            "ns14_path": str(args.ns14_path),
            "n_v5_rows_in_ns11": len(v5_rows),
            "n_ns11_evolved_rows": len(ns11_evolved),
            "n_ns14_rows": len(ns14_rows),
        },
        "build_meta": build_meta,
    }
    meta.update(summary)
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"variant            = {args.variant}")
    print(f"v5 rows            = {len(v5_rows)}")
    print(f"ns11 evolved rows  = {len(ns11_evolved)}")
    print(f"ns14 rows          = {len(ns14_rows)}")
    print(f"output             = {args.out} ({summary['total_pairs']} rows)")
    print(f"unique theorems    = {summary['unique_theorems']}")
    print(f"by domain          = {summary['by_domain']}")
    print(f"by variant         = {summary['by_variant']}")
    print(f"by role            = {summary['by_role']}")
    print(f"replay rows        = {summary['n_replay_rows']}")
    print(f"wrapper-Nat rows   = {summary['n_wrapper_nat_pattern_rows']}")
    print(f"meta               = {meta_path}")


if __name__ == "__main__":
    main()
