#!/usr/bin/env python3
"""SF4 Part 6 — live candidate-probe runner over CONFIRMED RC2 failures.

Driver/worker model (mirrors scripts/sx3_run_depth2_sequences.py and
scripts/rc3_minimal_relabel_set_ite_aesop.py): the driver spawns ONE worker
subprocess per theorem under an OS hard timeout (scripts/run_with_timeout.py); the
worker opens ONE Dojo, bounds the open with SIGALRM, runs the always-on controls
and every GATED probe (plus depth-1 sub-controls for sequences) from the initial
state with a per-tactic SIGALRM.

No win is final here — SX4 attribution (Part 7) is required. Production configs are
never read or modified.
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
BASELINE_CONTROLS = set(CONTROLS)


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
    if ("unexpected token" in e or "expected" in e and "tactic" in e or "unexpected identifier" in e):
        return "parse_error"
    if "maximum recursion" in e or "maxrecdepth" in e:
        return "max_recursion"
    return "proof_failed"


def _gate_fires(probe, full_name):
    gate = probe.get("gate", {})
    ns = full_name.split(".")[0] if "." in full_name else ""
    nss = gate.get("namespaces") or []
    if nss and ns not in nss and not any(full_name.startswith(n + ".") for n in nss):
        return False
    feats = gate.get("name_features") or []
    if feats and not any(f.lower() in full_name.lower() for f in feats):
        return False
    return True


def _depth1_parts(tac):
    if "<;>" not in tac:
        return []
    parts = [p.strip() for p in tac.split("<;>") if p.strip()]
    # first and last sub-tactics as standalone depth-1 controls
    out = []
    if parts:
        out.append(parts[0])
        if parts[-1] != parts[0]:
            out.append(parts[-1])
    return out


# ------------------------------- worker -----------------------------------
def worker(args):
    case = json.loads(args.case_json)
    probes = json.loads(args.probes_json)
    full_name = case["full_name"]
    gated = [p for p in probes if _gate_fires(p, full_name)]
    # depth-1 sub-controls for the gated sequences
    sub_controls = []
    for p in gated:
        for sc in _depth1_parts(p["tactic_or_sequence"]):
            if sc not in sub_controls:
                sub_controls.append(sc)

    res = {"full_name": full_name, "file_path": case.get("file_path"), "live": False,
           "gated_probes": [p["tactic_or_sequence"] for p in gated],
           "controls": [], "depth1_subcontrols": [], "probes_tried": [],
           "setup_error": None}
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
                    return {"tactic": tac, "solved": False, "outcome": "timeout", "dead": False}
                except Exception as e:
                    return {"tactic": tac, "solved": False, "outcome": "exception",
                            "error": f"{type(e).__name__}: {str(e)[:140]}", "dead": False}
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                r = {"tactic": tac, "solved": bool(fin),
                     "outcome": _classify_outcome(err, fin), "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            dead = False
            for c in CONTROLS:
                r = run_one(c); res["controls"].append(r)
                if r["dead"]:
                    dead = True; break
            if not dead:
                for sc in sub_controls:
                    r = run_one(sc); res["depth1_subcontrols"].append(r)
                    if r["dead"]:
                        dead = True; break
            if not dead:
                for p in gated:
                    r = run_one(p["tactic_or_sequence"])
                    r["family"] = p["family"]; r["risk"] = p.get("risk")
                    res["probes_tried"].append(r)
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


def _classify_pre_sx4(wres):
    if not wres.get("live"):
        return "open_flake" if wres.get("setup_error") else "needs_review"
    ctl_solved = {c["tactic"] for c in wres.get("controls", []) if c.get("solved")}
    sub_solved = {c["tactic"] for c in wres.get("depth1_subcontrols", []) if c.get("solved")}
    probe_wins = [p["tactic"] for p in wres.get("probes_tried", []) if p.get("solved")]
    if ctl_solved:
        return "baseline_duplicate"
    if probe_wins and not sub_solved:
        return "candidate_win"
    if probe_wins and sub_solved:
        return "depth1_duplicate"
    return "no_win"


def driver(args):
    conf = json.load(open(args.confirmed_failures))
    probes = json.load(open(args.probes))["probes"]
    failures = [r for r in conf.get("results", []) if r.get("classification") == "CONFIRMED_RC2_FAILURE"]

    results = []
    for r in failures:
        case = {"full_name": r["full_name"], "file_path": r.get("file_path")}
        gated = [p for p in probes if _gate_fires(p, r["full_name"])]
        if not case["file_path"] or not gated:
            results.append({"full_name": r["full_name"], "rc2_confirmed_failure": True,
                            "probes_tried": [], "winning_probes": [], "controls": [],
                            "classification_pre_sx4": "no_win" if case["file_path"] else "open_flake",
                            "note": "no gated probe" if case["file_path"] else "no file_path"})
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            wout = tf.name
        cmd = ["/opt/anaconda3/bin/python3", _TIMEOUT_HELPER, str(args.hard_timeout),
               "/opt/anaconda3/bin/python3", os.path.abspath(__file__), "--worker",
               "--worker-out", wout, "--case-json", json.dumps(case),
               "--probes-json", json.dumps(gated),
               "--open-timeout", str(args.open_timeout),
               "--timeout-per-tactic", str(args.timeout_per_tactic)]
        print(f"[sf4-probe] {r['full_name']} ({len(gated)} gated probes) ...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        try:
            wres = json.load(open(wout))
        except Exception:
            wres = {"live": False, "setup_error": f"worker no output (rc={rc})"}
        finally:
            try: os.unlink(wout)
            except OSError: pass
        cls = _classify_pre_sx4(wres)
        wins = [p["tactic"] for p in wres.get("probes_tried", []) if p.get("solved")]
        results.append({"full_name": r["full_name"], "rc2_confirmed_failure": True,
                        "live": wres.get("live"), "setup_error": wres.get("setup_error"),
                        "probes_tried": wres.get("probes_tried", []),
                        "winning_probes": wins,
                        "controls": wres.get("controls", []),
                        "depth1_subcontrols": wres.get("depth1_subcontrols", []),
                        "classification_pre_sx4": cls})
        print(f"[sf4-probe]   -> {cls} wins={wins}", flush=True)

    hist = {}
    for r in results:
        hist[r["classification_pre_sx4"]] = hist.get(r["classification_pre_sx4"], 0) + 1
    out = {"confirmed_failures_input": args.confirmed_failures, "probes_input": args.probes,
           "num_theorems": len(results), "classification_histogram": hist,
           "candidate_wins_pre_sx4": sorted(r["full_name"] for r in results
                                            if r["classification_pre_sx4"] == "candidate_win"),
           "results": results}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# SF4 candidate-probe results (pre-SX4)", "",
         f"- confirmed failures probed: **{len(results)}**",
         f"- histogram: {hist}",
         f"- candidate_win (pre-SX4): **{len(out['candidate_wins_pre_sx4'])}** "
         f"{out['candidate_wins_pre_sx4']}", "",
         "| theorem | pre-SX4 class | winning probes | controls solved |", "|---|---|---|---|"]
    for r in results:
        cs = [c["tactic"] for c in r.get("controls", []) if c.get("solved")]
        L.append(f"| `{r['full_name']}` | {r['classification_pre_sx4']} | "
                 f"{r.get('winning_probes')} | {cs} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf4-probe] theorems={len(results)} hist={hist} "
          f"candidate_wins={len(out['candidate_wins_pre_sx4'])}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-out")
    p.add_argument("--case-json")
    p.add_argument("--probes-json")
    p.add_argument("--confirmed-failures")
    p.add_argument("--probes")
    p.add_argument("--out-json")
    p.add_argument("--out-md")
    p.add_argument("--hard-timeout", type=int, default=240)
    p.add_argument("--open-timeout", type=int, default=90)
    p.add_argument("--timeout-per-tactic", type=int, default=45)
    args = p.parse_args(argv)
    if args.worker:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
