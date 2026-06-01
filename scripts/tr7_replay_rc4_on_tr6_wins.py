#!/usr/bin/env python3
"""TR7 Part 5 — replay RC4 static actions on the TR6 fresh wins (small live task).

For each TR6 fresh win: literal RC2 status (all are CONFIRMED_RC2_FAILURE — reused), the RC4
static wrapper outcome (reused from the RC4R benchmark for the 11 RC4R-known wins; run live via
eval_rollout for the rest), and the exact TR6 winning program run single-shot (run_transition)
to confirm it still reproduces. Classify per win:
  RC4_REPRODUCES_TR6_WIN / RC4_MISSES_GATE / RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS /
  TR6_PROGRAM_NO_LONGER_REPRODUCES / INFRA_FLAKE / NEEDS_REVIEW.
Does NOT alter the RC4 wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as PG  # noqa: E402  (run_tactics_live)
import rc4r_bench_common as C  # noqa: E402

_TIMEOUT = os.path.join(_REPO, "scripts", "run_with_timeout.py")
RC4R = "project/evolve/experiments/rc4_release_candidate"


def _p(*a):
    return os.path.join(_REPO, *a)


def worker_probe(args):
    case = json.loads(args.case_json)
    res = PG.run_tactics_live(case["file_path"], case["full_name"], case["tactics"],
                              open_timeout=args.open_timeout, per_tactic=args.timeout_per_tactic)
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def worker_wrapper(args):
    C.run_worker(args.worker_out, args.cases_json, args.out_dir, args.wrapper,
                 args.route_config, "hybrid_evolved", args.top_k, args.max_steps, "tr7replay")
    return 0


def _probe(fn, fp, tactics, args):
    with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        wout = tf.name
    cmd = [sys.executable, _TIMEOUT, str(args.hard_timeout), sys.executable,
           os.path.abspath(__file__), "--worker-probe", "--worker-out", wout,
           "--case-json", json.dumps({"full_name": fn, "file_path": fp, "tactics": tactics}),
           "--open-timeout", str(args.open_timeout), "--timeout-per-tactic", str(args.timeout_per_tactic)]
    subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.load(open(wout))
    except Exception:
        return {"ran": [], "setup_error": "unreadable"}
    finally:
        try:
            os.unlink(wout)
        except OSError:
            pass


def _wrapper_run(cases, args):
    """Run the RC4 wrapper via eval_rollout over `cases`; return {fn: status}."""
    if not cases:
        return {}
    with __import__("tempfile").NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        wout = tf.name
    cmd = [sys.executable, _TIMEOUT, "1500", sys.executable, os.path.abspath(__file__),
           "--worker-wrapper", "--worker-out", wout, "--cases-json", json.dumps(cases),
           "--out-dir", _p(RC4R, "../../experiments/tr7/out/replay_runs"),
           "--wrapper", _p(args.rc4_wrapper), "--route-config", _p(args.route_config),
           "--top-k", "8", "--max-steps", "8"]
    print(f"[tr7-replay] RC4 wrapper live on {len(cases)} ...", flush=True)
    subprocess.run(cmd, capture_output=True, text=True)
    try:
        return {fn: v.get("status") for fn, v in json.load(open(wout)).items()}
    except Exception:
        return {c["full_name"]: "trace_insufficient" for c in cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-probe", action="store_true")
    ap.add_argument("--worker-wrapper", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--cases-json")
    ap.add_argument("--out-dir")
    ap.add_argument("--wrapper")
    ap.add_argument("--corpus")
    ap.add_argument("--rc2-wrapper")
    ap.add_argument("--rc4-wrapper")
    ap.add_argument("--route-config", default="project/evolve/routing/ns24_router.json")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--open-timeout", type=int, default=90)
    ap.add_argument("--timeout-per-tactic", type=int, default=15)
    ap.add_argument("--hard-timeout", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    if args.worker_probe:
        sys.exit(worker_probe(args))
    if args.worker_wrapper:
        sys.exit(worker_wrapper(args))

    rows = [json.loads(l) for l in open(_p(args.corpus))]
    wins = [r for r in rows if r["is_tr6_fresh_win"]]
    rc4_bench = {r["full_name"]: r for r in json.load(open(_p(RC4R, "out/rc4_benchmark_results.json")))["results"]}

    # RC4 wrapper outcome: reuse benchmark for known; live for the rest
    need_live = [{"full_name": r["full_name"], "file_path": r["file_path"]}
                 for r in wins if r["full_name"] not in rc4_bench and r["file_path"]]
    live_wrap = _wrapper_run(need_live, args)

    records = []
    for r in wins:
        fn = r["full_name"]
        fp = r["file_path"]
        rc2_status = "failed"  # all TR6 fresh wins are CONFIRMED_RC2_FAILURE
        if fn in rc4_bench:
            rc4_wrap = rc4_bench[fn]["status"]
        else:
            rc4_wrap = live_wrap.get(fn, "not_run")
        gate = r["rc4_static_gate_fired"]
        # confirm exact TR6 program single-shot
        tr6_prog = r.get("tr6_winning_program")
        tr6_works = None
        rc4_action_solves = None
        if tr6_prog and fp:
            tactics = [tr6_prog]
            if gate and r.get("rc4_static_tactics"):
                tactics = tactics + r["rc4_static_tactics"]
            wres = _probe(fn, fp, tactics, args)
            ran = {x["tactic"]: x.get("solved") for x in wres.get("ran", [])}
            tr6_works = bool(ran.get(tr6_prog))
            if gate and r.get("rc4_static_tactics"):
                rc4_action_solves = any(ran.get(t) for t in r["rc4_static_tactics"])

        rc4_solved = rc4_wrap == "solved"
        if rc4_wrap in ("open_flake", "trace_insufficient", "not_run"):
            cls = "INFRA_FLAKE" if rc4_wrap != "not_run" else "NEEDS_REVIEW"
        elif rc4_solved:
            cls = "RC4_REPRODUCES_TR6_WIN"
        elif not gate:
            cls = "RC4_MISSES_GATE"
        elif tr6_works is False:
            cls = "TR6_PROGRAM_NO_LONGER_REPRODUCES"
        elif gate and not rc4_solved and (tr6_works or rc4_action_solves):
            cls = "RC4_ACTION_FAILS_BUT_TR6_PROGRAM_WORKS"
        else:
            cls = "NEEDS_REVIEW"
        records.append({
            "full_name": fn, "namespace": r["namespace"],
            "tr6_winning_program": tr6_prog, "tr6_winning_lemma": r.get("tr6_winning_lemma"),
            "rc2_status": rc2_status, "rc4_wrapper_status": rc4_wrap,
            "rc4_gate_fired": gate, "rc4_static_tactics": r.get("rc4_static_tactics"),
            "tr6_program_reproduces": tr6_works, "rc4_action_single_shot_solves": rc4_action_solves,
            "classification": cls,
        })

    from collections import Counter
    hist = Counter(r["classification"] for r in records)
    out = {"generated_by": "scripts/tr7_replay_rc4_on_tr6_wins.py",
           "num_wins": len(wins), "classification_histogram": dict(hist),
           "rc4_reproduces": sum(1 for r in records if r["classification"] == "RC4_REPRODUCES_TR6_WIN"),
           "records": records}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 RC4 replay on TR6 fresh wins", "",
          f"- wins replayed: {len(wins)} | classification: {dict(hist)}",
          f"- RC4 reproduces: **{out['rc4_reproduces']}/{len(wins)}**", "",
          "| theorem | ns | rc4_wrapper | gate | tr6_prog_works | rc4_action_solves | class |",
          "|---|---|---|---|---|---|---|"]
    for r in records:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['rc4_wrapper_status']} | "
                  f"{r['rc4_gate_fired']} | {r['tr6_program_reproduces']} | "
                  f"{r['rc4_action_single_shot_solves']} | {r['classification']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-replay] {dict(hist)}")
    print(f"[tr7-replay] rc4_reproduces={out['rc4_reproduces']}/{len(wins)}")


if __name__ == "__main__":
    main()
