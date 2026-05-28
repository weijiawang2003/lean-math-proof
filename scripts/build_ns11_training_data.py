"""NS11 — scaled trace-to-training builder.

Extends `scripts/build_ns10_training_data.py` with:

  - Three filter variants: ``conservative`` (close only, strict
    held-out), ``medium`` (close + assist, standard held-out),
    ``coverage`` (close + assist, NO held-out — for upper-bound /
    memorization runs).
  - Default trace root is *all* of project/evolve/ — that picks up
    NS5 skeleton_runs and the older autonomous_runs that NS10 had
    not yet consumed (NS10 only walked NS6/NS7/NS9 dirs).
  - A ``--combine-with`` flag to merge with the original v5 dataset
    (``project/seq2seq_data_v5.jsonl``), so the small evolve
    corpus is trained jointly rather than alone.
  - Four prompt styles (``--prompt-style``): ``vanilla`` (baseline),
    ``origin`` (adds the origin tag), ``skeleton`` (adds the
    skeleton shape), ``premise`` (adds retrieved-premise context if
    available — required for retrieved_premise rows to be safe to
    train on).

Usage:

    # Conservative — close transitions only, hold-out enforced.
    python scripts/build_ns11_training_data.py --variant conservative \
        --out project/data/ns11_train_conservative.jsonl

    # Medium — adds K=3 assist transitions.
    python scripts/build_ns11_training_data.py --variant medium \
        --out project/data/ns11_train_medium.jsonl

    # Coverage — drops hold-out so the model can train on
    # Nat.div_pos / Nat.mul_eq_* etc. (use as upper bound, not honest
    # held-out evaluation).
    python scripts/build_ns11_training_data.py --variant coverage \
        --out project/data/ns11_train_coverage.jsonl

    # Combined — medium variant + v5 base data.
    python scripts/build_ns11_training_data.py --variant medium \
        --combine-with project/seq2seq_data_v5.jsonl \
        --out project/data/ns11_train_combined.jsonl

Output: a JSONL with both ``tactic`` and ``completion`` keys (the
trainer reads ``tactic``); plus a sibling ``*_meta.json`` with
counts and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# Default held-out theorems mirror NS10. The "coverage" variant
# overrides this with the empty set.
DEFAULT_HELD_OUT = {
    "Nat.AM_GM",
    "Nat.div_lt_iff_lt_mul'",
    "Nat.div_lt_one_iff",
    "Nat.div_pos",
    "Nat.div_pos_iff",
    "Nat.mul_eq_left",
    "Nat.mul_eq_right",
    "Nat.dvd_iff_div_mul_eq",
    "Nat.sqrt_lt",
    "Nat.pow_lt_pow_iff_left",
}

DEFAULT_ALLOWED_ORIGINS = {
    "fallback_tactic", "family_tactic", "generative_topk",
    "term_builder", "tactic_template",
}

MAX_TACTIC_LEN = 200
MAX_STATE_LEN = 2500


@dataclass(frozen=True)
class VariantConfig:
    name: str
    include_assist: bool
    k_assist_window: int
    held_out: frozenset[str]
    include_retrieved: bool

    @classmethod
    def get(cls, name: str) -> "VariantConfig":
        if name == "conservative":
            return cls(
                name="conservative",
                include_assist=False,
                k_assist_window=0,
                held_out=frozenset(DEFAULT_HELD_OUT),
                include_retrieved=False,
            )
        if name == "medium":
            return cls(
                name="medium",
                include_assist=True,
                k_assist_window=3,
                held_out=frozenset(DEFAULT_HELD_OUT),
                include_retrieved=False,
            )
        if name == "coverage":
            return cls(
                name="coverage",
                include_assist=True,
                k_assist_window=5,
                held_out=frozenset(),  # NO held-out
                include_retrieved=False,
            )
        raise ValueError(f"unknown variant: {name!r}")


# ----- trace walking ---------------------------------------------------------


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


def iter_trace_paths(roots: list[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        for p in root.rglob("traces.jsonl"):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield p


def load_episodes(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in paths:
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


def accepted_sequence(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], str]]:
    """Per accepted step, return (step, row, role)."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            s = int(r.get("step"))
        except (TypeError, ValueError):
            continue
        by_step[s].append(r)
    out = []
    for s in sorted(by_step.keys()):
        step_rows = by_step[s]
        cidx = next((i for i, r in enumerate(step_rows) if _is_close(r)), None)
        if cidx is not None:
            out.append((s, step_rows[cidx], "close"))
            continue
        aidx = next((i for i, r in enumerate(step_rows) if _is_advance(r)), None)
        if aidx is not None:
            out.append((s, step_rows[aidx], "advance"))
    return out


# ----- prompt builders -------------------------------------------------------


def build_prompt(r: dict[str, Any], style: str) -> str:
    full_name = r.get("full_name", "")
    state = r.get("state_pp", "")
    base = f"Theorem: {full_name}\n\nProof state:\n{state}\n"
    if style == "vanilla":
        return base
    if style == "origin":
        origin = r.get("tactic_origin") or "?"
        return f"Theorem: {full_name}\nOrigin: {origin}\n\nProof state:\n{state}\n"
    if style == "skeleton":
        shape = r.get("skeleton_shape") or "?"
        return f"Theorem: {full_name}\nShape: {shape}\n\nProof state:\n{state}\n"
    if style == "premise":
        prem = r.get("retrieved_premise") or r.get("tactic_template_source") or ""
        suffix = f"\nPremise: {prem}\n" if prem else ""
        return f"Theorem: {full_name}{suffix}\n\nProof state:\n{state}\n"
    raise ValueError(f"unknown prompt style: {style!r}")


# ----- admissibility ---------------------------------------------------------


def is_admissible(
    r: dict[str, Any],
    held_out: frozenset[str],
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
        return False
    origin = r.get("tactic_origin")
    if origin not in allowed:
        return False
    return True


# ----- main ------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["conservative", "medium", "coverage"],
                   default="medium")
    p.add_argument("--traces-dir", action="append", type=Path, default=None,
                   help="defaults to [project/evolve] which walks all subdirs")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--meta", default=None, type=Path,
                   help="defaults to <out stem>_meta.json")
    p.add_argument("--prompt-style", choices=["vanilla", "origin", "skeleton", "premise"],
                   default="vanilla")
    p.add_argument("--include-retrieved", action="store_true",
                   help="opt in to retrieved_premise rows (requires --prompt-style premise "
                        "to be reproducible at inference time)")
    p.add_argument("--combine-with", type=Path, default=None,
                   help="path to an existing JSONL (e.g. project/seq2seq_data_v5.jsonl) "
                        "whose rows are appended to the output unchanged")
    p.add_argument("--held-out", nargs="*", default=None,
                   help="override held-out set (use empty list to disable)")
    args = p.parse_args()

    variant = VariantConfig.get(args.variant)
    held_out = (
        frozenset(args.held_out) if args.held_out is not None
        else variant.held_out
    )
    allowed = set(DEFAULT_ALLOWED_ORIGINS)
    if args.include_retrieved or variant.include_retrieved:
        allowed.add("retrieved_premise")

    traces_dirs = args.traces_dir or [Path("project/evolve")]
    trace_paths = list(iter_trace_paths(traces_dirs))
    print(f"variant:     {variant.name}")
    print(f"prompt:      {args.prompt_style}")
    print(f"trace dirs:  {[str(t) for t in traces_dirs]}")
    print(f"trace files: {len(trace_paths)}")

    episodes = load_episodes(trace_paths)
    print(f"episodes:    {len(episodes)}")

    pairs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    by_origin: dict[str, int] = defaultdict(int)
    by_theorem: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    by_set: dict[str, int] = defaultdict(int)
    runs_seen: set[str] = set()
    held_out_seen: dict[str, int] = defaultdict(int)
    tactic_lens: list[int] = []

    for eid, ep_rows in episodes.items():
        acc = accepted_sequence(ep_rows)
        if not acc:
            continue
        for i, (step, r, role) in enumerate(acc):
            name = r.get("full_name") or ""
            if name in held_out:
                held_out_seen[name] += 1
                continue
            if not is_admissible(r, held_out, allowed):
                continue

            classify_role = role
            assist_distance: int | None = None
            if role == "advance":
                if not variant.include_assist:
                    continue
                close_in_window = None
                for j in range(i + 1, min(len(acc), i + 1 + variant.k_assist_window)):
                    if acc[j][2] == "close":
                        close_in_window = j - i
                        break
                if close_in_window is None:
                    continue
                classify_role = "advance_assist"
                assist_distance = close_in_window

            prompt = build_prompt(r, args.prompt_style)
            tactic = r["tactic"]
            key = (r["state_pp"], tactic)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            tactic_hash = hashlib.sha1(tactic.encode("utf-8")).hexdigest()[:12]
            pair = {
                "prompt": prompt,
                "tactic": tactic,
                "completion": tactic,
                "theorem": name,
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
                "_variant": variant.name,
                "_prompt_style": args.prompt_style,
            }
            pairs.append(pair)
            by_origin[pair["origin"] or "?"] += 1
            by_theorem[name or "?"] += 1
            by_role[classify_role] += 1
            by_set[pair["source_run"] or "?"] += 1
            runs_seen.add(pair["source_run"])
            tactic_lens.append(len(tactic))

    # Append combine-with rows as-is (already in trainer schema).
    combined_in_rows = 0
    if args.combine_with is not None:
        if not args.combine_with.exists():
            raise SystemExit(f"combine-with path not found: {args.combine_with}")
        with args.combine_with.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # The v5 base rows already have prompt+tactic. Tag
                # them for transparency, do not touch the strings.
                obj.setdefault("_variant", "v5_base")
                obj.setdefault("_prompt_style", "vanilla")
                pairs.append(obj)
                combined_in_rows += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    md5 = hashlib.md5(args.out.read_bytes()).hexdigest()
    meta = {
        "v": "ns11",
        "variant": variant.name,
        "prompt_style": args.prompt_style,
        "out_path": str(args.out),
        "md5": md5,
        "n_pairs_total": len(pairs),
        "n_pairs_from_traces": len(pairs) - combined_in_rows,
        "n_pairs_from_combine": combined_in_rows,
        "n_unique_theorems": len(by_theorem),
        "n_runs_seen": len(runs_seen),
        "held_out": sorted(held_out),
        "held_out_seen_in_traces": dict(held_out_seen),
        "by_origin": dict(by_origin),
        "by_role": dict(by_role),
        "top_theorems": dict(sorted(by_theorem.items(), key=lambda x: -x[1])[:15]),
        "top_runs": dict(sorted(by_set.items(), key=lambda x: -x[1])[:15]),
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
            "include_retrieved": args.include_retrieved or variant.include_retrieved,
            "include_assist": variant.include_assist,
            "k_assist_window": variant.k_assist_window,
            "rejects_self_reference": True,
            "rejects_held_out": bool(held_out),
            "dedup_by_state_tactic": True,
        },
        "combine_with": str(args.combine_with) if args.combine_with else None,
        "trace_roots": [str(t) for t in traces_dirs],
        "n_trace_files": len(trace_paths),
        "n_episodes_loaded": len(episodes),
    }
    meta_path = args.meta or args.out.with_suffix("").with_suffix(".meta.json")
    if not args.meta:
        # Keep meta path co-located with out.
        meta_path = args.out.parent / (args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"wrote {len(pairs)} pairs to {args.out}")
    print(f"  from traces:  {len(pairs) - combined_in_rows}")
    print(f"  from combine: {combined_in_rows}")
    print(f"  origins: {dict(by_origin)}")
    print(f"  roles: {dict(by_role)}")
    print(f"  unique theorems: {len(by_theorem)}")
    print(f"  held-out trace rows skipped: {sum(held_out_seen.values())}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
