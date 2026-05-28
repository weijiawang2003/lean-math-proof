"""NS8 — cache `gen_v5` model outputs per protected state.

Run the base generative policy once per protected state and persist
its top-K tactic predictions. Deterministic decoding (beam search +
fixed seed) ensures the cache is reusable across mutations.

Cache key = sha1((state_pp, full_name, model_path, decode_mode,
top_k, seed)). Two cache rows with the same key collapse to one.

Output schema (JSONL, one row per protected state):

    {
      "cache_key": "abc123...",
      "theorem": "Nat.div_lt_iff_lt_mul'",
      "state_hash": "deadbeef",
      "state_pp": "...",
      "full_name": "Nat.div_lt_iff_lt_mul'",
      "model_path": "project/models/gen_v5",
      "decode_mode": "beam",
      "top_k": 8,
      "seed": null,
      "model_outputs": ["aesop", "simp_all", ...]   # length == top_k
    }

Usage:
    python scripts/ns8_cache_model_outputs.py \\
        --protected-states project/evolve/archive/protected_states.jsonl \\
        --ckpt-dir project/models/gen_v5 \\
        --top-k 8 \\
        --out project/evolve/archive/model_outputs_cache.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _cache_key(
    state_pp: str,
    full_name: str,
    model_path: str,
    decode_mode: str,
    top_k: int,
    seed: int | None,
) -> str:
    canonical = "|".join((
        state_pp or "",
        full_name or "",
        model_path,
        decode_mode,
        str(top_k),
        str(seed),
    ))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protected-states", type=Path, required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--decode-mode", default="beam")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Load existing cache if present (resumable).
    existing: dict[str, dict[str, Any]] = {}
    if args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing[r["cache_key"]] = r
                except json.JSONDecodeError:
                    continue
        print(f"resuming: {len(existing)} cached entries already present")

    # Load protected states.
    states: list[dict[str, Any]] = []
    state_keys: set[tuple[str, str]] = set()
    with args.protected_states.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                k = (s.get("state_pp") or "", s.get("full_name") or "")
                if k in state_keys:
                    continue
                state_keys.add(k)
                states.append(s)
            except json.JSONDecodeError:
                continue
    print(f"protected states: {len(states)} unique (state_pp, full_name) pairs")

    # Load model (lazy — only if anything to cache).
    todo = []
    for s in states:
        ck = _cache_key(
            s.get("state_pp") or "", s.get("full_name") or "",
            args.ckpt_dir, args.decode_mode, args.top_k, args.seed,
        )
        if ck in existing:
            continue
        todo.append((ck, s))
    print(f"to cache: {len(todo)} states (rest already cached)")

    if todo:
        from generative_policy import GenerativePolicy
        print(f"loading {args.ckpt_dir} (decode_mode={args.decode_mode}, top_k={args.top_k})...")
        policy = GenerativePolicy(
            ckpt_dir=args.ckpt_dir,
            decode_mode=args.decode_mode,
            temperature=0.8,
            seed=args.seed,
        )
        print("model loaded")
        new_entries = []
        for i, (ck, s) in enumerate(todo, 1):
            try:
                outputs = policy.rank_tactics(
                    s.get("state_pp") or "",
                    s.get("full_name") or "",
                    k=args.top_k,
                )
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {s.get('theorem'):50s} ERROR: {exc}")
                outputs = []
            entry = {
                "cache_key": ck,
                "theorem": s.get("theorem"),
                "state_hash": s.get("state_hash"),
                "state_pp": s.get("state_pp"),
                "full_name": s.get("full_name"),
                "model_path": args.ckpt_dir,
                "decode_mode": args.decode_mode,
                "top_k": args.top_k,
                "seed": args.seed,
                "model_outputs": list(outputs),
            }
            new_entries.append(entry)
            if i % 5 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] cached")
        # Append to file.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as f:
            for e in new_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"appended {len(new_entries)} entries to {args.out}")
    print(f"total cached: {len(existing) + len(todo)}")


if __name__ == "__main__":
    main()
