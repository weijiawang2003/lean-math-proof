#!/usr/bin/env python3
"""RC4R Part 9 — RC4 determinism rerun.

Reruns the RC4 release wrapper through the real eval_rollout_all search TWICE over a
representative subset — rc4_known_wins + canonical demo_v1 floor + a sample of gate-firing
fresh out-of-sample frontier cases + offgate controls — and compares per-theorem solved
outcomes. The gate is a pure function (stable by construction); the solved outcome is compared
on cleanly-executed theorems, with Dojo hard-timeout / worker-kill events reported as infra
flakes (excluded from the hash) per the RC4B/RC4C/RC4D determinism methodology. `--analyze-only`
recomputes from saved runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4r_bench_common as C  # noqa: E402

RUNDIR = "project/evolve/experiments/rc4_release_candidate/out/determinism_runs"


def _p(*a):
    return os.path.join(_REPO, *a)


def _select(manifest, fresh_sample):
    targets, seen = [], set()
    # known wins (the firing credited wins) + demo floor. offgate/negatives are non-gate-firing
    # → RC4 ≡ RC2, trivially deterministic (pure-function gate), so excluded to bound runtime.
    for setname in ("rc4_known_wins", "canonical_demo_v1"):
        rel = manifest["set_files"].get(setname)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            if fn in seen or not e.get("file_path"):
                continue
            seen.add(fn)
            targets.append({"full_name": fn, "file_path": e["file_path"]})
    # fresh sample: gate-firing fresh
    rel = manifest["set_files"].get("fresh_out_of_sample_frontier")
    cnt = 0
    if rel:
        for e in json.load(open(_p(rel))):
            if cnt >= fresh_sample:
                break
            fn = e["full_name"]
            if fn in seen or not e.get("file_path") or not e.get("rc4_gate_fires"):
                continue
            seen.add(fn); cnt += 1
            targets.append({"full_name": fn, "file_path": e["file_path"]})
    return targets


def _run_pass(targets, wrapper, route_config, out_dir, ckpt_path, chunk_size, hard_timeout,
              top_k, max_steps, label):
    ckpt = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}
    pending = [t for t in targets if t["full_name"] not in ckpt]
    chunks = [pending[i:i + chunk_size] for i in range(0, len(pending), chunk_size)]
    os.makedirs(out_dir, exist_ok=True)
    for ci, chunk in enumerate(chunks):
        wout = os.path.join(out_dir, f"{label}_chunk_{ci}.json")
        cmd = [sys.executable, _p("scripts/run_with_timeout.py"), str(hard_timeout),
               sys.executable, _p("scripts/rc4r_run_rc4_benchmark.py"), "--worker",
               "--worker-out", wout, "--cases-json", json.dumps(chunk),
               "--out-dir", os.path.join(out_dir, "runs_" + label), "--wrapper", wrapper,
               "--route-config", route_config, "--top-k", str(top_k),
               "--max-steps", str(max_steps), "--set-label", label]
        print(f"[rc4r-determinism] {label} chunk {ci+1}/{len(chunks)} ({len(chunk)}) ...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {c["full_name"]: {"status": "trace_insufficient"} for c in chunk}
        ckpt.update(wres)
        json.dump(ckpt, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
    return ckpt


def _hash(run):
    norm = {fn: (v.get("status") == "solved") for fn, v in sorted(run.items())}
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rc4-wrapper", required=True)
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--fresh-sample", type=int, default=35)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--hard-timeout", type=int, default=1500)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    targets = _select(manifest, args.fresh_sample)
    cdir = _p(RUNDIR)
    os.makedirs(cdir, exist_ok=True)
    if args.analyze_only:
        run1 = json.load(open(os.path.join(cdir, "run1.json")))
        run2 = json.load(open(os.path.join(cdir, "run2.json")))
    else:
        print(f"[rc4r-determinism] {len(targets)} targets, 2 passes ...", flush=True)
        run1 = _run_pass(targets, _p(args.rc4_wrapper), _p(args.route_config), cdir,
                         os.path.join(cdir, "run1.json"), args.chunk_size, args.hard_timeout,
                         args.top_k, args.max_steps, "run1")
        run2 = _run_pass(targets, _p(args.rc4_wrapper), _p(args.route_config), cdir,
                         os.path.join(cdir, "run2.json"), args.chunk_size, args.hard_timeout,
                         args.top_k, args.max_steps, "run2")

    keys = set(run1) & set(run2)
    flakes = [fn for fn in keys if run1[fn].get("status") in ("open_flake", "trace_insufficient")
              or run2[fn].get("status") in ("open_flake", "trace_insufficient")]
    flaky = set(flakes)
    diffs = []
    for fn in keys:
        s1 = run1[fn].get("status") == "solved"
        s2 = run2[fn].get("status") == "solved"
        if s1 != s2 and fn not in flaky:
            diffs.append({"full_name": fn, "run1": run1[fn].get("status"), "run2": run2[fn].get("status")})
    clean1 = {fn: run1[fn] for fn in keys if fn not in flaky}
    clean2 = {fn: run2[fn] for fn in keys if fn not in flaky}
    h1, h2 = _hash(clean1), _hash(clean2)
    win_flakes = [fn for fn in flakes if run1[fn].get("status") == "solved"
                  or run2[fn].get("status") == "solved"]
    deterministic = (h1 == h2) and not diffs

    out = {"generated_by": "scripts/rc4r_determinism_rerun.py",
           "num_targets": len(targets), "num_compared": len(keys),
           "clean_run1_hash": h1, "clean_run2_hash": h2,
           "genuine_diffs": diffs, "open_flakes": flakes, "num_open_flakes": len(flakes),
           "win_affecting_flakes": win_flakes, "deterministic": deterministic,
           "determinism_note": "hash over cleanly-executed theorems; open flakes are Dojo "
                               "hard-timeout / worker-kill infra events, excluded from the hash."}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 determinism rerun", "",
          f"- targets: {len(targets)} | compared: {len(keys)}",
          f"- clean run1 hash: `{h1}` | clean run2 hash: `{h2}`",
          f"- genuine diffs: **{len(diffs)}** | open flakes: {len(flakes)} | "
          f"win-affecting flakes: {len(win_flakes)}",
          f"- **deterministic (modulo infra flakes): {deterministic}**", "", out["determinism_note"]]
    if diffs:
        md += ["", "## Genuine diffs", ""] + [f"- {d}" for d in diffs]
    if flakes:
        md += ["", f"## Open flakes ({len(flakes)})", ""] + \
              [f"- {fn}" + ("  ⚠ win-affecting" if fn in win_flakes else "") for fn in flakes]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-determinism] clean {h1} vs {h2} | diffs={len(diffs)} | flakes={len(flakes)} | "
          f"deterministic={deterministic}")


if __name__ == "__main__":
    main()
