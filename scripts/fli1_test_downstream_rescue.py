#!/usr/bin/env python3
"""FLI1 Part 9 — faithful downstream-rescue test, at the theorem's LeanDojo position.

These seeds are real Mathlib lemmas the restricted RC5 battery couldn't rediscover at their file
position (where the lemma + downstream are out of scope). So rescue is tested IN LeanDojo at that
position — never against a fresh full import. For each candidate with a usable lemma/proof we run:
  (1) CONTROLS (no candidate): simp / aesop / constructor<;>simp / ext — if any closes → DIRECT_SOLVE_DUPLICATE.
  (2) RESCUE: `simp [L]`-style deployment of an existing close lemma, or the prefix+proof path of
      a proved new candidate (chained). If a rescue closes and no control did → DOWNSTREAM_RESCUE.
Robustness = the winning rescue is re-run twice. Driver/worker subprocess + hard timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
PLAN = "project/evolve/experiments/fli1/cases/fli1_live_rerun_plan.json"
CONTROLS = ["simp", "aesop", "constructor <;> simp", "constructor <;> aesop", "ext x <;> simp"]


def _p(*a):
    return os.path.join(_REPO, *a)


# ----------------------------- worker -----------------------------
def worker(case, open_timeout, per_tactic):
    import signal
    out = {"theorem": case["theorem"], "controls": [], "rescues": [],
           "initial_num_goals": None, "setup_error": None}

    class _PT(Exception):
        pass

    def _alarm(sig, frm):
        raise _PT()

    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"], full_name=case["theorem"]))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(open_timeout)
        try:
            cm = _Dojo(thm)
            dojo, state0 = cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        out["initial_num_goals"] = getattr(state0, "num_goals", None)

        def run_seq(seq):
            cur = state0
            for tac in seq:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(per_tactic)
                try:
                    o = _env.run_transition(dojo, thm, cur, tac)
                except _PT:
                    return {"seq": seq, "solved": False, "outcome": "timeout"}
                except Exception as e:  # noqa: BLE001
                    return {"seq": seq, "solved": False, "outcome": f"exc:{type(e).__name__}"}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                if getattr(o, "session_dead", False):
                    return {"seq": seq, "solved": False, "outcome": "dead", "dead": True}
                if getattr(o, "is_finished", False):
                    return {"seq": seq, "solved": True, "outcome": "finished"}
                ns = getattr(o, "next_state", None)
                if ns is None or getattr(o, "is_error", False):
                    return {"seq": seq, "solved": False, "outcome": "failed",
                            "num_goals": getattr(cur, "num_goals", None)}
                cur = ns
            return {"seq": seq, "solved": False, "outcome": "progressed",
                    "num_goals": getattr(cur, "num_goals", None)}

        for ctrl in case["controls"]:
            r = run_seq([ctrl])
            out["controls"].append(r)
            if r.get("dead"):
                break
        for seq in case["rescue_sequences"]:
            r = run_seq(seq)
            out["rescues"].append(r)
            if r.get("dead"):
                break
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
    except _PT:
        out["setup_error"] = f"dojo open exceeded {open_timeout}s"
    except Exception as e:  # noqa: BLE001
        out["setup_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


# ----------------------------- driver -----------------------------
def _rescue_sequences(c):
    seqs = []
    L = c.get("closest_existing_lemma")
    if c.get("retrieval_gap") and L:
        seqs += [[f"simp [{L}]"], [f"constructor <;> simp [{L}]"],
                 [f"simp [{L}] <;> aesop"], [f"ext x <;> simp [{L}]"], [f"rw [{L}]"]]
    if c.get("prove") == "PROVED":
        prefix = c.get("prefix_to_reach") or []
        proof = c.get("proof_tactic")
        if proof:
            seqs.append([proof])                       # proof tactic from initial state
            if prefix:
                seqs.append(prefix + [proof])          # reach residual, then prove it
    # dedup
    seen, out = set(), []
    for s in seqs:
        k = tuple(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def driver(args):
    plan = {s["seed_id"]: s for s in json.load(open(_p(PLAN)))["seeds"]}
    cands = [json.loads(l) for l in open(_p(args.proofs)) if l.strip()]
    trace_dir = _p("project/evolve/experiments/fli1/live_traces/rescue")
    os.makedirs(trace_dir, exist_ok=True)
    results = []
    for c in cands:
        sid = c["source_seed_ids"][0]
        thm = c["downstream_targets"][0]
        fp = plan.get(sid, {}).get("file_path")
        rescue_seqs = _rescue_sequences(c)
        base = {"candidate_id": c["candidate_id"], "seed_id": sid, "theorem": thm,
                "namespace": c["namespace"], "closest_existing_lemma": c.get("closest_existing_lemma"),
                "retrieval_gap": c.get("retrieval_gap"), "prove": c.get("prove")}
        if not rescue_seqs or not fp:
            base.update({"classification": "NO_RESCUE" if not rescue_seqs else "NEEDS_REVIEW",
                         "reason": "no usable lemma/proof" if not rescue_seqs else "no file_path",
                         "control_solved": None, "rescue_solved_by": None, "robust": None})
            results.append(base)
            continue
        ckpt = os.path.join(trace_dir, f"{c['candidate_id']}.json")
        if os.path.exists(ckpt):
            wres = json.load(open(ckpt))
        else:
            wout = ckpt + ".tmp"
            case = {"theorem": thm, "file_path": fp, "controls": CONTROLS,
                    "rescue_sequences": rescue_seqs}
            cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout), sys.executable,
                   os.path.abspath(__file__), "--worker", "--worker-out", wout,
                   "--case-json", json.dumps(case), "--open-timeout", str(args.open_timeout),
                   "--timeout-per-tactic", str(args.timeout_per_tactic)]
            print(f"[fli1-rescue] {c['candidate_id']} {thm} ...", flush=True)
            subprocess.run(cmd, capture_output=True, text=True)
            wres = json.load(open(wout)) if os.path.exists(wout) else {"setup_error": "no worker output"}
            if os.path.exists(wout):
                os.replace(wout, ckpt)
            else:
                json.dump(wres, open(ckpt, "w"))
        ctrl_solved = [r["seq"][0] for r in wres.get("controls", []) if r.get("solved")]
        rescue_win = next((r for r in wres.get("rescues", []) if r.get("solved")), None)
        progressed = any(r.get("outcome") == "progressed"
                         and (r.get("num_goals") or 99) < (wres.get("initial_num_goals") or 99)
                         for r in wres.get("rescues", []))
        if wres.get("setup_error"):
            cls = "NEEDS_REVIEW"
        elif ctrl_solved:
            cls = "DIRECT_SOLVE_DUPLICATE"
        elif rescue_win:
            cls = "DOWNSTREAM_RESCUE"
        elif progressed:
            cls = "PARTIAL_PROGRESS"
        else:
            cls = "NO_RESCUE"
        # robustness: re-run the winning rescue once more (fresh worker)
        robust = None
        if cls == "DOWNSTREAM_RESCUE":
            wout2 = ckpt + ".rerun"
            case2 = {"theorem": thm, "file_path": fp, "controls": [],
                     "rescue_sequences": [rescue_win["seq"]]}
            cmd2 = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout), sys.executable,
                    os.path.abspath(__file__), "--worker", "--worker-out", wout2,
                    "--case-json", json.dumps(case2), "--open-timeout", str(args.open_timeout),
                    "--timeout-per-tactic", str(args.timeout_per_tactic)]
            subprocess.run(cmd2, capture_output=True, text=True)
            if os.path.exists(wout2):
                w2 = json.load(open(wout2))
                robust = any(r.get("solved") for r in w2.get("rescues", []))
                os.unlink(wout2)
        base.update({"classification": cls, "control_solved": ctrl_solved,
                     "rescue_solved_by": rescue_win["seq"] if rescue_win else None,
                     "rescue_tactic": (" ; ".join(rescue_win["seq"]) if rescue_win else None),
                     "robust": robust, "initial_num_goals": wres.get("initial_num_goals"),
                     "setup_error": wres.get("setup_error")})
        results.append(base)
        print(f"  -> {cls}" + (f" via `{base.get('rescue_tactic')}` robust={robust}"
                               if rescue_win else ""), flush=True)

    with open(_p(args.out_jsonl), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["classification"] for r in results)
    rescues = [r for r in results if r["classification"] == "DOWNSTREAM_RESCUE"]
    summary = {"generated_by": "scripts/fli1_test_downstream_rescue.py",
               "num_tested": len(results), "classification_histogram": dict(hist),
               "downstream_rescues": len(rescues),
               "robust_rescues": sum(1 for r in rescues if r.get("robust")),
               "partial_progress": hist.get("PARTIAL_PROGRESS", 0),
               "direct_solve_duplicates": hist.get("DIRECT_SOLVE_DUPLICATE", 0),
               "rescue_targets": [{"theorem": r["theorem"], "lemma": r["closest_existing_lemma"],
                                   "tactic": r["rescue_tactic"], "robust": r["robust"]}
                                  for r in rescues]}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 downstream rescue summary", "",
          f"- tested: {summary['num_tested']} | **DOWNSTREAM_RESCUE: {summary['downstream_rescues']}** "
          f"(robust {summary['robust_rescues']}) | partial: {summary['partial_progress']} | "
          f"direct-solve-dup: {summary['direct_solve_duplicates']}",
          f"- histogram: {summary['classification_histogram']}", "",
          "| candidate | theorem | class | rescue tactic | robust |", "|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['candidate_id']} | `{r['theorem']}` | {r['classification']} | "
                  f"`{r.get('rescue_tactic') or ''}` | {r.get('robust')} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-rescue] DONE rescues={summary['downstream_rescues']} "
          f"hist={summary['classification_histogram']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out")
    ap.add_argument("--case-json")
    ap.add_argument("--open-timeout", type=int, default=120)
    ap.add_argument("--timeout-per-tactic", type=int, default=25)
    ap.add_argument("--proofs")
    ap.add_argument("--residual-goals")
    ap.add_argument("--seeds")
    ap.add_argument("--out-jsonl")
    ap.add_argument("--out-summary-json")
    ap.add_argument("--out-summary-md")
    ap.add_argument("--hard-timeout", type=int, default=240)
    args = ap.parse_args()
    if args.worker:
        r = worker(json.loads(args.case_json), args.open_timeout, args.timeout_per_tactic)
        json.dump(r, open(args.worker_out, "w"), ensure_ascii=False)
        return
    driver(args)


if __name__ == "__main__":
    main()
