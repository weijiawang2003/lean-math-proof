#!/usr/bin/env python3
"""RC4D Part 9 — determinism check.

Re-runs the gated RC4D probe twice over known wins (RC4A/RC4B/RC4C_residue) + fresh holdouts
+ overlap controls + negative controls, comparing per-theorem (gate-fired, solved,
winning_component) outcomes. The gate is a pure function, so gate/component decisions must be
100% stable; the solved outcome is compared on cleanly-executed theorems only (heavy `<;>
aesop` probes can hit the Dojo hard-timeout and get worker-killed → infrastructure flakes,
excluded from the hash, reported separately, per the RC4B/RC4C determinism lesson). Reports
run1/run2 clean hashes, genuine diffs, flake diffs, component-stability, deterministic
true/false. `--analyze-only` recomputes from saved runs.
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
import rc4d_gate as G  # noqa: E402

_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
# Determinism scope = every credited-win-bearing set (covers all 23 wins) + the de-dup
# overlap controls + negatives. composition_fresh_holdout is intentionally EXCLUDED: it is
# dominated by emitted-and-failed heavy `<;> aesop` probes that add hours of wall-clock and
# no win-stability signal (the gate is a pure function checked separately on every set).
SETS = ("rc4a_known_wins", "rc4b_known_wins", "rc4c_residue_known_wins",
        "component_overlap_controls",
        "negative_controls", "namespace_negative_controls")
RUNDIR = "project/evolve/experiments/rc4_candidates/composition_rc4d/out/determinism_runs"


def _p(*a):
    return os.path.join(_REPO, *a)


def worker(args):
    case = json.loads(args.case_json)
    res = G.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                             open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _one_run(targets, args, ckpt_path):
    out = json.load(open(ckpt_path)) if os.path.exists(ckpt_path) else {}
    for t in targets:
        fn = t["full_name"]
        if fn in out:
            continue
        if not t["tactics"]:
            out[fn] = {"gate_fired": False, "solved": False, "winning_component": None,
                       "outcome": "no_emission"}
            json.dump(out, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
            continue
        with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
               sys.executable, os.path.abspath(__file__), "--worker", "--worker-out", wout,
               "--case-json", json.dumps({"full_name": fn, "file_path": t["file_path"],
                                          "tactics": t["tactics"]}),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            wres = json.load(open(wout))
        except (ValueError, OSError):
            wres = {"ran": [], "setup_error": "worker_output_unreadable"}
        finally:
            try:
                os.unlink(wout)
            except OSError:
                pass
        ran = {x["tactic"]: x for x in wres.get("ran", [])}
        win_comp = None
        for em in t["emissions"]:
            if ran.get(em["tactic"], {}).get("solved"):
                win_comp = em["component"]
                break
        out[fn] = {"gate_fired": True, "solved": win_comp is not None,
                   "winning_component": win_comp, "setup_error": wres.get("setup_error")}
        json.dump(out, open(ckpt_path, "w"), ensure_ascii=False, indent=2)
    return out


def _hash(run):
    norm = {fn: {"g": v["gate_fired"], "s": v["solved"], "c": v.get("winning_component")}
            for fn, v in sorted(run.items())}
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--manifest", "--validation-manifest", dest="manifest")
    ap.add_argument("--policy")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=300)
    args = ap.parse_args()
    if args.worker:
        sys.exit(worker(args))

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    targets, seen = [], set()
    for setname in SETS:
        rel = manifest["set_files"].get(setname)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            fn = e["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            fires, em = G.gate_fires(policy, e.get("namespace"), e.get("goal_text"), fn)
            targets.append({"full_name": fn, "file_path": e.get("file_path"),
                            "tactics": G.tactics_of(em) if fires else None,
                            "emissions": em if fires else []})

    nfire = sum(1 for t in targets if t["tactics"])
    cdir = _p(RUNDIR)
    os.makedirs(cdir, exist_ok=True)
    if args.analyze_only:
        run1 = json.load(open(os.path.join(cdir, "run1.json")))
        run2 = json.load(open(os.path.join(cdir, "run2.json")))
    else:
        print(f"[rc4d-determinism] run 1/2 over {len(targets)} ({nfire} gate-firing) ...", flush=True)
        run1 = _one_run(targets, args, os.path.join(cdir, "run1.json"))
        print("[rc4d-determinism] run 2/2 ...", flush=True)
        run2 = _one_run(targets, args, os.path.join(cdir, "run2.json"))

    flakes = [fn for fn in run1 if run1[fn].get("setup_error") or run2[fn].get("setup_error")]
    flaky = set(flakes)
    genuine_diffs, flake_diffs = [], []
    for fn in run1:
        a, b = run1[fn], run2[fn]
        if (a["gate_fired"] != b["gate_fired"] or a["solved"] != b["solved"]
                or a.get("winning_component") != b.get("winning_component")):
            (flake_diffs if fn in flaky else genuine_diffs).append({"full_name": fn, "run1": a, "run2": b})
    clean1 = {fn: v for fn, v in run1.items() if fn not in flaky}
    clean2 = {fn: v for fn, v in run2.items() if fn not in flaky}
    h1, h2 = _hash(clean1), _hash(clean2)
    gate_stable = all(run1[fn]["gate_fired"] == run2[fn]["gate_fired"] for fn in run1)
    comp_stable = all(run1[fn].get("winning_component") == run2[fn].get("winning_component")
                      for fn in clean1)
    deterministic = (h1 == h2) and not genuine_diffs and gate_stable
    win_flakes = [fn for fn in flakes if run1[fn].get("solved") or run2[fn].get("solved")]

    out = {"generated_by": "scripts/rc4d_determinism_check.py",
           "num_targets": len(targets), "num_gate_firing": nfire,
           "clean_run1_hash": h1, "clean_run2_hash": h2,
           "gate_decisions_stable": gate_stable, "component_decisions_stable": comp_stable,
           "genuine_diffs": genuine_diffs, "flake_diffs": flake_diffs,
           "open_flakes": flakes, "num_open_flakes": len(flakes),
           "win_affecting_flakes": win_flakes, "deterministic": deterministic,
           "determinism_note": ("hash over cleanly-executed theorems; open flakes are Dojo "
                                "hard-timeout / worker-kill on heavy `<;> aesop` goals, excluded."),
           "run1": run1, "run2": run2}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4D determinism check", "",
          f"- targets: {len(targets)} ({nfire} gate-firing)",
          f"- clean run1 hash: `{h1}` | clean run2 hash: `{h2}`",
          f"- gate decisions stable: **{gate_stable}** | component decisions stable: **{comp_stable}**",
          f"- genuine diffs: **{len(genuine_diffs)}** | flake diffs: {len(flake_diffs)} | "
          f"open flakes: {len(flakes)}",
          f"- **deterministic (modulo infra flakes): {deterministic}**", "", out["determinism_note"]]
    if genuine_diffs:
        md += ["", "## Genuine diffs", ""] + [f"- {d}" for d in genuine_diffs]
    if flakes:
        md += ["", f"## Open flakes ({len(flakes)})", ""] + \
              [f"- {fn}" + ("  ⚠ win-affecting" if fn in win_flakes else "") for fn in flakes]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4d-determinism] clean {h1} vs {h2} | genuine_diffs={len(genuine_diffs)} | "
          f"flakes={len(flakes)} | gate_stable={gate_stable} | comp_stable={comp_stable} | "
          f"deterministic={deterministic}")


if __name__ == "__main__":
    main()
