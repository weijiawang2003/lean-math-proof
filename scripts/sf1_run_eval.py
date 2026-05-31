#!/usr/bin/env python3
"""SF1 live-eval unblocker — run eval_rollout_all on an SF1 batch FILE.

`eval_rollout_all.py` accepts only registered theorem-set NAMES
(`--theorem-set`, choices=list_theorem_sets()). Rather than edit it or tasks.py,
this thin wrapper registers an SF1 batch JSON as a theorem set *at runtime*, then
delegates to ``eval_rollout_all.main()`` with the registered name.

It does this by patching the two names ``eval_rollout_all`` imported
(`get_theorems`, `list_theorem_sets`) — so argparse's `choices` and the per-set
resolution both see the SF1 set. No file on disk is modified; production defaults,
the RC1 command, and protected configs are untouched.

Usage:
  python3 scripts/sf1_run_eval.py \
      --theorem-set-file project/evolve/experiments/sf1/theorem_sets/<set>.json \
      [--register-name <name>] \
      -- <all the normal eval_rollout_all flags except --theorem-set>

  # convenience: flags may also be passed without the `--` separator.

Theorem-set file schema (either form):
  {"<set_name>": [ {"file_path": "...", "full_name": "...", "namespace": "..."}, ... ]}
  or a flat list:  [ {"file_path": "...", "full_name": "..."}, ... ]

Only rows with a non-null ``file_path`` are registered (live eval needs it).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_rows(path):
    obj = json.load(open(path))
    rows = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                rows.extend(v)
    elif isinstance(obj, list):
        rows = obj
    return rows


def _build_theorem_configs(rows):
    from core_types import TheoremConfig
    fields = {f.name: f for f in dataclasses.fields(TheoremConfig)}
    required = [n for n, f in fields.items()
                if f.default is dataclasses.MISSING
                and getattr(f, "default_factory", dataclasses.MISSING) is dataclasses.MISSING]
    tcs, dropped = [], 0
    for r in rows:
        fp = r.get("file_path")
        fn = r.get("full_name") or r.get("name") or r.get("decl_name")
        if not fp or not fn:
            dropped += 1
            continue
        kwargs = {}
        for n in fields:
            if n in r and r[n] is not None:
                kwargs[n] = r[n]
        kwargs.setdefault("file_path", fp)
        kwargs.setdefault("full_name", fn)
        # fill any other required field we don't have with a benign default
        for n in required:
            kwargs.setdefault(n, kwargs.get(n))
        try:
            tcs.append(TheoremConfig(**{k: v for k, v in kwargs.items() if k in fields}))
        except TypeError as e:
            print(f"[sf1_run_eval] WARN: could not build TheoremConfig for {fn}: {e}",
                  file=sys.stderr)
            dropped += 1
    return tcs, dropped


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=True,
                                description="Run eval_rollout_all on an SF1 batch file.")
    p.add_argument("--theorem-set-file", required=True)
    p.add_argument("--register-name", default=None)
    # everything else is forwarded to eval_rollout_all
    known, rest = p.parse_known_args(argv)
    # strip a lone "--" separator if present
    rest = [a for a in rest if a != "--"]

    if not os.path.isfile(known.theorem_set_file):
        print(f"[sf1_run_eval] ERROR: theorem-set file not found: "
              f"{known.theorem_set_file}", file=sys.stderr)
        return 2

    rows = _load_rows(known.theorem_set_file)
    tcs, dropped = _build_theorem_configs(rows)
    if not tcs:
        print(f"[sf1_run_eval] ERROR: no rows with file_path in "
              f"{known.theorem_set_file} (dropped={dropped}); nothing to eval.",
              file=sys.stderr)
        return 3

    name = known.register_name or (
        "sf1_runtime_" + os.path.splitext(os.path.basename(known.theorem_set_file))[0])

    import eval_rollout_all as E
    _orig_get = E.get_theorems
    _orig_list = E.list_theorem_sets

    def patched_list():
        base = list(_orig_list())
        if name not in base:
            base.append(name)
        return base

    def patched_get(n):
        if n == name:
            return tcs
        return _orig_get(n)

    E.get_theorems = patched_get
    E.list_theorem_sets = patched_list

    fwd = ["eval_rollout_all.py", "--theorem-set", name] + rest
    print(f"[sf1_run_eval] registered '{name}' with {len(tcs)} theorems "
          f"(dropped {dropped} without file_path)")
    print(f"[sf1_run_eval] delegating: {' '.join(fwd)}")
    sys.argv = fwd
    try:
        rc = E.main()
    except SystemExit as e:  # eval_rollout_all may sys.exit
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
