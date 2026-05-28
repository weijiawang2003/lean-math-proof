"""NS16 Stage 4 — training dataset variants.

Three variants on top of NS15's nat_oversample corpus:

  - ``oversample_10x`` — NS15 corpus + 10× duplication of every
    NS16 wrapper-only Nat row. Trained from gen_v5 base.

  - ``oversample_20x`` — same but 20×.

  - ``curriculum_continue`` — just the NS16 wrapper-only rows
    (×20). Trained as a short continuation of the existing
    ``gen_v5_ns15_nat_oversample`` checkpoint.

The NS15 corpus is regenerated from
``scripts/build_ns15_training_data.py --variant nat_oversample``
so we don't have to keep a binary copy committed; the script
loads the existing JSONL if present, otherwise reconstructs.

Usage::

    python scripts/build_ns16_datasets.py --variant oversample_10x \\
        --out project/data/ns16_train_oversample_10x.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


NS15_PATH = Path("project/data/ns15_nat_oversample.jsonl")
NS16_WRAPPER_PATH = Path("project/data/ns16_nat_wrapper_only.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["oversample_10x", "oversample_20x",
                             "curriculum_continue"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not NS16_WRAPPER_PATH.exists():
        raise SystemExit(
            f"missing {NS16_WRAPPER_PATH} — run "
            "scripts/build_ns16_training_data.py first"
        )
    ns16 = load_jsonl(NS16_WRAPPER_PATH)

    if args.variant == "oversample_10x":
        if not NS15_PATH.exists():
            raise SystemExit(
                f"missing {NS15_PATH} — run "
                "scripts/build_ns15_training_data.py --variant nat_oversample first"
            )
        base = load_jsonl(NS15_PATH)
        out_rows = list(base)
        for _ in range(10):
            for r in ns16:
                out_rows.append(dict(r))
        meta_kind = "from_gen_v5"
        oversample_n = 10
    elif args.variant == "oversample_20x":
        if not NS15_PATH.exists():
            raise SystemExit(f"missing {NS15_PATH}")
        base = load_jsonl(NS15_PATH)
        out_rows = list(base)
        for _ in range(20):
            for r in ns16:
                out_rows.append(dict(r))
        meta_kind = "from_gen_v5"
        oversample_n = 20
    elif args.variant == "curriculum_continue":
        # Just the NS16 rows, oversampled. Trained from
        # gen_v5_ns15_nat_oversample (short fine-tune).
        out_rows = []
        for _ in range(20):
            for r in ns16:
                out_rows.append(dict(r))
        meta_kind = "from_ns15_nat_oversample"
        oversample_n = 20
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_origin: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    ns16_thms: set[str] = set()
    ns16_rows = 0
    for r in out_rows:
        by_origin[r.get("origin") or "unknown"] += 1
        by_role[r.get("role") or "unknown"] += 1
        if r.get("_variant") == "ns16":
            ns16_rows += 1
            ns16_thms.add(r["theorem"])

    meta = {
        "variant": args.variant,
        "out_path": str(args.out),
        "n_rows": len(out_rows),
        "n_ns16_rows": ns16_rows,
        "n_ns16_source_rows": len(ns16),
        "n_ns16_unique_theorems": len(ns16_thms),
        "oversample_factor": oversample_n,
        "init_from": meta_kind,
        "by_origin": dict(by_origin),
        "by_role": dict(by_role),
    }
    if args.variant != "curriculum_continue":
        meta["n_ns15_base_rows"] = len(base)
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"variant       = {args.variant}")
    print(f"init_from     = {meta_kind}")
    print(f"total rows    = {len(out_rows)}")
    print(f"NS16 copies   = {ns16_rows} ({oversample_n}× of {len(ns16)} unique)")
    print(f"NS16 thms     = {len(ns16_thms)}")
    print(f"out           = {args.out}")
    print(f"meta          = {meta_path}")


if __name__ == "__main__":
    main()
