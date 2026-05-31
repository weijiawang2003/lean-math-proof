#!/usr/bin/env python3
"""TR2 Part 6 — run live probes (reuse-first, live-increment).

Driver/worker model (mirrors scripts/sf4_run_candidate_probes.py): one worker
subprocess per theorem under an OS hard timeout (scripts/run_with_timeout.py); the
worker opens ONE Dojo (SIGALRM-bounded) and runs the *uncovered* probes from the
initial state with a per-tactic SIGALRM.

Reuse layer: SF4 already executed the four bare controls (and depth-1 sub-controls
and cluster probes) live on every confirmed RC2 failure. Those verified outcomes are
REUSED (provenance "sf4_reused"); only tactics SF4 never ran on a theorem
(chiefly `exact?`, the depth-gap battery extras, and controls on RC2-solved cases)
are executed live (provenance "tr2_live"). The merged per-theorem record matches the
SF4 schema so SX4 attribution (Part 7) consumes it unchanged.

No win is final here — SX4 attribution is required. Production configs untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")
CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop"]


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def _classify_outcome(err, solved):
    if solved:
        return "success"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if "unexpected token" in e or ("expected" in e and "tactic" in e) or "unexpected identifier" in e:
        return "parse_error"
    if "maximum recursion" in e or "maxrecdepth" in e:
        return "max_recursion"
    return "proof_failed"


def _depth1_parts(tac):
    if "<;>" not in tac:
        return []
    parts = [p.strip() for p in tac.split("<;>") if p.strip()]
    out = []
    if parts:
        out.append(parts[0])
        if parts[-1] != parts[0]:
            out.append(parts[-1])
    return out


# ------------------------------- worker -----------------------------------
def worker(args):
    case = json.loads(args.case_json)
    tactics = json.loads(args.tactics_json)  # list of {tactic, kind, family}
    full_name = case["full_name"]
    res = {"full_name": full_name, "file_path": case.get("file_path"), "live": False,
           "ran": [], "setup_error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"], full_name=full_name))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(args.open_timeout)
        try:
            dojo_cm = _Dojo(thm)
            dojo, state0 = dojo_cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        try:
            res["live"] = True

            def run_one(tac):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(args.timeout_per_tactic)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                    rec = getattr(out, "record", None)
                    fin = bool(getattr(out, "is_finished", False))
                    err = getattr(rec, "error_message", None) if rec else None
                    dead = bool(getattr(out, "session_dead", False))
                except _ProbeTimeout:
                    return {"solved": False, "outcome": "timeout", "dead": False}
                except Exception as e:
                    return {"solved": False, "outcome": "exception",
                            "error": f"{type(e).__name__}: {str(e)[:140]}", "dead": False}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                r = {"solved": bool(fin), "outcome": _classify_outcome(err, fin), "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            for t in tactics:
                r = run_one(t["tactic"])
                r.update({"tactic": t["tactic"], "kind": t["kind"], "family": t.get("family")})
                res["ran"].append(r)
                if r["dead"]:
                    break
        finally:
            try:
                dojo_cm.__exit__(None, None, None)
            except Exception:
                pass
    except _ProbeTimeout:
        res["setup_error"] = f"dojo open exceeded {args.open_timeout}s"
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + traceback.format_exc()[-300:]
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def _sf4_reuse_map(path):
    """full_name -> {tactic: {solved, outcome, error}} from SF4 verified outcomes."""
    m = {}
    if not os.path.exists(path):
        return m
    for r in json.load(open(path)).get("results", []):
        d = {}
        for c in r.get("controls", []) + r.get("depth1_subcontrols", []) + r.get("probes_tried", []):
            d[c["tactic"]] = {"solved": c.get("solved"), "outcome": c.get("outcome"),
                              "error": c.get("error")}
        m[r["full_name"]] = {"live": r.get("live"), "tactics": d}
    return m


def driver(args):
    plan = json.load(open(args.probe_plan))
    reuse = _sf4_reuse_map(args.sf4_probe_results)

    results = []
    n_live_theorems = n_live_tactics = n_reused_tactics = 0
    for t in plan["theorems"]:
        fn = t["full_name"]
        fp = t.get("file_path")
        rmap = reuse.get(fn, {}).get("tactics", {})
        # full tactic universe for this theorem: controls + prediction probes
        wanted = []
        for c in CONTROLS:
            wanted.append({"tactic": c, "kind": "control", "family": "control"})
        for pr in t.get("probes", []):
            wanted.append({"tactic": pr["tactic_or_sequence"], "kind": "probe", "family": pr["family"]})
        # dedup by tactic, keeping first kind/family (control precedence)
        seen, uni = set(), []
        for w in wanted:
            if w["tactic"] not in seen:
                seen.add(w["tactic"]); uni.append(w)

        reused_recs, to_run = [], []
        for w in uni:
            if w["tactic"] in rmap:
                rr = rmap[w["tactic"]]
                reused_recs.append({"tactic": w["tactic"], "kind": w["kind"], "family": w["family"],
                                    "solved": rr["solved"], "outcome": rr["outcome"],
                                    "error": rr.get("error"), "provenance": "sf4_reused"})
            else:
                to_run.append(w)

        live_recs, setup_error, live_ok = [], None, None
        if to_run and fp:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                wout = tf.name
            cmd = [sys.executable, _TIMEOUT_HELPER, str(args.hard_timeout),
                   sys.executable, os.path.abspath(__file__), "--worker",
                   "--worker-out", wout, "--case-json", json.dumps({"full_name": fn, "file_path": fp}),
                   "--tactics-json", json.dumps(to_run),
                   "--open-timeout", str(args.open_timeout),
                   "--timeout-per-tactic", str(args.timeout_per_tactic)]
            print(f"[tr2-probe] {fn}: reuse={len(reused_recs)} live={len(to_run)} ...", flush=True)
            rc = subprocess.run(cmd, capture_output=True, text=True).returncode
            try:
                wres = json.load(open(wout))
            except Exception:
                wres = {"live": False, "setup_error": f"worker no output (rc={rc})", "ran": []}
            finally:
                try: os.unlink(wout)
                except OSError: pass
            live_ok = wres.get("live")
            setup_error = wres.get("setup_error")
            for rr in wres.get("ran", []):
                rr["provenance"] = "tr2_live"
                live_recs.append(rr)
            if live_ok:
                n_live_theorems += 1
            n_live_tactics += len(live_recs)
        elif not to_run:
            live_ok = True  # everything reused
        n_reused_tactics += len(reused_recs)

        # assemble in SF4 schema for SX4 attribution
        allrecs = reused_recs + live_recs
        by_tac = {r["tactic"]: r for r in allrecs}
        controls = [{"tactic": c, **{k: by_tac[c][k] for k in ("solved", "outcome", "provenance")},
                     **({"error": by_tac[c].get("error")} if by_tac[c].get("error") else {})}
                    for c in CONTROLS if c in by_tac]
        probes_tried = [r for r in allrecs if r["kind"] == "probe"]
        # depth-1 sub-controls for any sequence probe (from reuse or live)
        sub_tacs = []
        for pr in probes_tried:
            for sc in _depth1_parts(pr["tactic"]):
                if sc not in sub_tacs:
                    sub_tacs.append(sc)
        depth1 = [{"tactic": sc, **{k: by_tac[sc][k] for k in ("solved", "outcome", "provenance")}}
                  for sc in sub_tacs if sc in by_tac]

        live_flag = bool(live_ok) and (live_ok if to_run else True)
        results.append({
            "full_name": fn, "file_path": fp, "namespace": t.get("namespace"),
            "rc2_classification": t.get("rc2_classification"),
            "predicted_label": t.get("predicted_label"), "probe_family": t.get("probe_family"),
            "for_sf5_retrieval": t.get("for_sf5_retrieval"),
            "live": live_flag, "setup_error": setup_error,
            "num_reused": len(reused_recs), "num_live": len(live_recs),
            "controls": controls, "depth1_subcontrols": depth1,
            "probes_tried": probes_tried,
            "winning_probes": [p["tactic"] for p in probes_tried if p.get("solved")],
        })
        print(f"[tr2-probe]   {fn} -> wins={results[-1]['winning_probes']} "
              f"reused={len(reused_recs)} live={len(live_recs)}", flush=True)

    import collections
    win_hist = collections.Counter("win" if r["winning_probes"] else "no_win" for r in results)
    out = {"probe_plan_input": args.probe_plan, "sf4_probe_results_input": args.sf4_probe_results,
           "num_theorems": len(results),
           "num_live_theorems": n_live_theorems, "num_live_tactics": n_live_tactics,
           "num_reused_tactics": n_reused_tactics,
           "win_histogram": dict(win_hist),
           "any_probe_win": sorted(r["full_name"] for r in results if r["winning_probes"]),
           "results": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# TR2 live probe results (pre-SX4)", "",
         f"- theorems: **{len(results)}**  ·  live theorems: {n_live_theorems}  ·  "
         f"live tactics: {n_live_tactics}  ·  reused tactics: {n_reused_tactics}",
         f"- probe wins (pre-SX4): **{len(out['any_probe_win'])}** {out['any_probe_win']}", "",
         "| theorem | family | live | reused | live# | controls solved | probe wins |",
         "|---|---|---|---|---|---|---|"]
    for r in results:
        cs = [c["tactic"] for c in r["controls"] if c.get("solved")]
        L.append(f"| `{r['full_name']}` | {r['probe_family']} | {r['live']} | {r['num_reused']} | "
                 f"{r['num_live']} | {cs} | {r['winning_probes']} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr2-probe] theorems={len(results)} live_theorems={n_live_theorems} "
          f"live_tactics={n_live_tactics} reused={n_reused_tactics} wins={len(out['any_probe_win'])}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-out")
    p.add_argument("--case-json")
    p.add_argument("--tactics-json")
    p.add_argument("--probe-plan")
    p.add_argument("--sf4-probe-results", default="project/evolve/experiments/sf4/out/sf4_probe_results.json")
    p.add_argument("--out-json")
    p.add_argument("--out-md")
    p.add_argument("--hard-timeout", type=int, default=300)
    p.add_argument("--open-timeout", type=int, default=90)
    p.add_argument("--timeout-per-tactic", type=int, default=30)
    args = p.parse_args(argv)
    if args.worker:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
