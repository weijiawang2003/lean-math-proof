#!/usr/bin/env python3
"""SX3 — live LeanDojo depth-2 symbolic sequence runner.

Mirrors the proven sf2_run_set_probe_ladders driver/worker pattern:
  * DRIVER iterates cases; for each it spawns ONE worker subprocess under an
    OS-level hard timeout (scripts/run_with_timeout.py) so one hang can't stall
    the run.
  * WORKER opens ONE Dojo session for its theorem, runs the always-on controls
    (simp / simp_all / aesop / classical <;> aesop / simp [Set.ite]) and every
    gated depth-2 sequence from the initial state with a per-tactic SIGALRM.

Each gated sequence and each control is applied from state0 only (depth-2
sequences are single grouped `<;>` tactic strings — never bare semicolons or
multi-line bullet blocks, which env.run_transition rejects).

Attribution is computed honestly per theorem:
  classification:
    new_depth2_win      — a gated sequence solves it AND every control fails
                          (so it is genuinely beyond the RC2 single-shot battery)
    single_step_duplicate — a control's single-shot `simp [Set.ite]` solves it
                          (belongs to RC2, NOT SX3)
    baseline_duplicate  — a bare control (simp/simp_all/aesop/classical) solves it
    rc2_already_solved  — rc2_baseline_finished==true for this theorem
    no_sequence_win     — nothing (control or sequence) closed it
    parse_limited       — sequences only failed via parse/recursion limits
    unknown             — setup/resolution error

No solve is a confirmed win; every win carries requires_minimal_relabel=true.
RC1/RC2 production configs are never read or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIMEOUT_HELPER = os.path.join(_REPO, "scripts", "run_with_timeout.py")

DEFAULT_CONTROLS = ["simp", "simp_all", "aesop", "classical <;> aesop", "simp [Set.ite]"]
BASELINE_CONTROLS = {"simp", "simp_all", "aesop", "classical <;> aesop"}
SINGLE_STEP_CONTROL = "simp [Set.ite]"


class _ProbeTimeout(Exception):
    pass


def _alarm(_s, _f):
    raise _ProbeTimeout()


def classify_outcome(err, solved):
    if solved:
        return "solved"
    e = (err or "").lower()
    if not e:
        return "proof_failed"
    if ("expected end of input" in e or "expected '{' or tactic" in e
            or "unexpected token" in e or "unexpected identifier" in e
            or "expected term" in e):
        return "parse_error"
    if "maximum recursion depth" in e or "maxrecdepth" in e:
        return "max_recursion"
    if "applyexttheorem only applies" in e:
        return "ext_not_applicable"
    if "no goals" in e:
        return "no_goals"
    if "unknown" in e and ("identifier" in e or "constant" in e):
        return "unknown_ident"
    return "proof_failed"


# ----------------------------- gating -------------------------------------
def _namespace_of(case):
    ns = case.get("namespace")
    if ns:
        return ns
    fn = case.get("full_name", "")
    return fn.split(".")[0] if "." in fn else ""


def _shape_cat(case):
    s = (case.get("shape") or case.get("primary_goal_shape") or "").lower()
    if "subset_iff" in s:
        return "set_subset_iff"
    if "membership_iff" in s or ("member" in s and "iff" in s):
        return "set_membership_iff"
    if s == "iff" or "iff" in s:
        return "set_iff"
    if "equal" in s or s == "eq":
        return "set_equality"
    if "subset" in s:
        return "set_subset"
    if "member" in s:
        return "set_membership"
    if "union" in s:
        return "set_union"
    if "diff" in s:
        return "set_diff"
    return "unknown"


_IFF_SHAPES = {"set_iff", "set_membership_iff", "set_subset_iff"}


def gate_matches(fam, case):
    """Return (emitted: bool, reason: str). Pure, deterministic."""
    gate = fam.get("gate", {})
    fn = case.get("full_name", "")
    ns = _namespace_of(case)
    # forbid namespaces
    for bad in gate.get("forbid_namespaces", []):
        if ns == bad or fn.startswith(bad + "."):
            return False, f"forbidden_ns:{bad}"
    # name/namespace must contain one of the required tokens
    needs = gate.get("namespace_or_name_contains", [])
    if needs and not any(tok in fn or tok.rstrip(".") == ns for tok in needs):
        return False, "name_token_miss"
    # signal-gated (ite families): dominated by the ite/dite signal
    if "signal" in gate:
        low = fn.lower()
        if "ite" in low:  # catches ite and dite
            return True, "signal_ite"
        return False, "no_ite_signal"
    # shape-gated families
    target = gate.get("target_shape")
    if target:
        cat = _shape_cat(case)
        accept = set()
        for t in target:
            if t == "set_iff":
                accept |= _IFF_SHAPES
            else:
                accept.add(t)
        if cat in accept:
            return True, f"shape:{cat}"
        return False, f"shape_miss:{cat}"
    return True, "no_shape_gate"


def load_cases(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    for k in ("cases", "selected", "theorems"):
        if k in d and isinstance(d[k], list):
            return d[k]
    raise ValueError(f"cannot find case list in {path}")


def select_families(registry, families_arg):
    fams = registry["families"]
    if not families_arg:
        keys = list(fams.keys())
    else:
        keys = [k.strip() for k in families_arg.split(",") if k.strip()]
    out = []
    for k in keys:
        if k not in fams:
            raise ValueError(f"unknown family {k}")
        if fams[k].get("status") == "diagnostic_only" and "BYCASES" in k:
            # by_cases requires ?p inference we do not implement; skip live emission
            continue
        out.append(fams[k])
    return out


# ----------------------------- worker -------------------------------------
def worker(args):
    cases = load_cases(args.cases)
    registry = json.load(open(args.registry))
    fams = select_families(registry, args.families)
    case = cases[args.worker_theorem]
    controls = [] if args.no_controls else list(DEFAULT_CONTROLS)

    gated = []
    for fam in fams:
        ok, reason = gate_matches(fam, case)
        gated.append({"family": fam["family"], "sequence": fam["sequence"],
                      "emitted": ok, "gate_reason": reason})
    emitted_seqs = [g for g in gated if g["emitted"]][:args.max_sequences_per_theorem]

    res = {"full_name": case["full_name"], "file_path": case.get("file_path"),
           "namespace": _namespace_of(case), "role": case.get("role"),
           "shape": _shape_cat(case),
           "rc2_baseline_finished": case.get("known_rc2_status_finished",
                                             case.get("rc2_baseline_finished")),
           "live": False, "initial_goal": None,
           "gate_decisions": gated,
           "gated_sequences_tried": [], "controls": [], "wins": [],
           "best_win": None, "classification": None, "notes": [], "setup_error": None}
    try:
        sys.path.insert(0, _REPO)
        import env as _env
        from core_types import TheoremConfig as _TC
        from lean_dojo import Dojo as _Dojo
        repo = _env.make_repo()
        thm = _env.make_theorem(repo, _TC(file_path=case["file_path"],
                                          full_name=case["full_name"]))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
        # Bound Dojo session open: lean_dojo's open occasionally sleeps/retries
        # indefinitely on certain theorems; SIGALRM interrupts its time.sleep so a
        # bad open aborts cleanly instead of burning the whole OS hard-timeout budget.
        if hasattr(signal, "SIGALRM"):
            signal.alarm(args.open_timeout)
        try:
            dojo_cm = _Dojo(thm)
            dojo, state0 = dojo_cm.__enter__()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        try:
            res["live"] = True
            res["initial_goal"] = getattr(state0, "pp", None) or getattr(state0, "state", None)

            def apply(tac):
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(args.timeout_per_sequence)
                try:
                    out = _env.run_transition(dojo, thm, state0, tac)
                finally:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                rec = getattr(out, "record", None)
                return (bool(getattr(out, "is_finished", False)),
                        bool(getattr(out, "session_dead", False)),
                        getattr(rec, "error_message", None) if rec else None)

            def run_one(tac):
                try:
                    fin, dead, err = apply(tac)
                except _ProbeTimeout:
                    return {"tactic": tac, "solved": False, "outcome": "timeout_inner",
                            "error": f"exceeded {args.timeout_per_sequence}s", "dead": False}
                except Exception as e:
                    return {"tactic": tac, "solved": False, "outcome": "exception",
                            "error": f"{type(e).__name__}: {str(e)[:160]}", "dead": False}
                oc = classify_outcome(err, fin)
                r = {"tactic": tac, "solved": bool(fin), "outcome": oc, "dead": bool(dead)}
                if err and not fin:
                    r["error"] = err[:200]
                return r

            dead_seen = False
            # controls first (cheap-ish, needed for attribution)
            for c in controls:
                r = run_one(c)
                res["controls"].append(r)
                if r["solved"]:
                    res["wins"].append({"kind": "control", "tactic": c})
                if r["dead"]:
                    dead_seen = True
                    res["notes"].append(f"session_dead after control {c}")
                    break
            # gated depth-2 sequences
            if not dead_seen:
                for g in emitted_seqs:
                    r = run_one(g["sequence"])
                    r["family"] = g["family"]
                    res["gated_sequences_tried"].append(r)
                    if r["solved"]:
                        res["wins"].append({"kind": "sequence", "family": g["family"],
                                            "tactic": g["sequence"]})
                    if r["dead"]:
                        res["notes"].append(f"session_dead after sequence {g['sequence']}")
                        break
        finally:
            try:
                dojo_cm.__exit__(None, None, None)
            except Exception:
                pass
        res["classification"] = classify(res)
        res["best_win"] = pick_best_win(res)
    except _ProbeTimeout:
        res["setup_error"] = f"dojo open exceeded {args.open_timeout}s"
        res["classification"] = "dojo_open_timeout"
    except Exception as e:
        res["setup_error"] = f"{type(e).__name__}: {str(e)[:200]}\n" + traceback.format_exc()[-300:]
        res["classification"] = "unknown"
    json.dump(res, open(args.worker_out, "w"), ensure_ascii=False, indent=2)
    return 0


def classify(res):
    if res.get("rc2_baseline_finished") is True:
        return "rc2_already_solved"
    seq_wins = [w for w in res["wins"] if w["kind"] == "sequence"]
    ctl_solved = {c["tactic"] for c in res["controls"] if c["solved"]}
    if SINGLE_STEP_CONTROL in ctl_solved:
        return "single_step_duplicate"
    if ctl_solved & BASELINE_CONTROLS:
        return "baseline_duplicate"
    if seq_wins:
        return "new_depth2_win"
    # nothing solved; was it only parse/recursion limited?
    seq_outcomes = {s["outcome"] for s in res["gated_sequences_tried"]}
    if seq_outcomes and seq_outcomes <= {"parse_error", "max_recursion", "timeout_inner",
                                         "ext_not_applicable", "no_goals", "exception"}:
        return "parse_limited"
    return "no_sequence_win"


def pick_best_win(res):
    """Prefer the cheapest depth-2 sequence among solving sequences (registry order)."""
    seq_wins = [w for w in res["wins"] if w["kind"] == "sequence"]
    if seq_wins:
        return seq_wins[0]["tactic"]
    ctl_wins = [w for w in res["wins"] if w["kind"] == "control"]
    if ctl_wins:
        return ctl_wins[0]["tactic"]
    return None


# ----------------------------- driver -------------------------------------
def _hist(results, key):
    h = {}
    for r in results:
        v = r.get(key)
        h[v] = h.get(v, 0) + 1
    return h


def result_hash(results):
    canon = []
    for r in sorted(results, key=lambda x: x.get("full_name", "")):
        canon.append({"full_name": r.get("full_name"),
                      "classification": r.get("classification"),
                      "best_win": r.get("best_win"),
                      "wins": sorted(json.dumps(w, sort_keys=True) for w in r.get("wins", []))})
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def driver(args):
    cases = load_cases(args.cases)
    registry = json.load(open(args.registry))
    fams = select_families(registry, args.families)
    n_ctl = 0 if args.no_controls else len(DEFAULT_CONTROLS)
    results = []
    # unique tmp prefix per run (out-json hash) so concurrent runs never collide
    run_tag = hashlib.sha256(os.path.abspath(args.out_json).encode()).hexdigest()[:10]

    def checkpoint():
        agg = {"cases_file": args.cases, "registry": args.registry,
               "families": [f["family"] for f in fams],
               "num_theorems": len(cases),
               "num_live": sum(1 for r in results if r.get("live")),
               "classification_histogram": _hist(results, "classification"),
               "new_depth2_wins": sorted(r["full_name"] for r in results
                                         if r.get("classification") == "new_depth2_win"),
               "results": results,
               "result_hash": result_hash(results),
               "note": "Live LeanDojo depth-2 sequences. No solve is a confirmed win; "
                       "minimal-sufficient relabel + RC2-baseline comparison required before "
                       "any promotion. RC1/RC2 production configs untouched."}
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(agg, open(args.out_json, "w"), ensure_ascii=False, indent=2)
        return agg

    for idx in range(len(cases)):
        nseq = 0
        for fam in fams:
            ok, _ = gate_matches(fam, cases[idx])
            if ok:
                nseq += 1
        nseq = min(nseq, args.max_sequences_per_theorem)
        hard = args.timeout_per_sequence * (nseq + n_ctl + 1) + args.open_timeout + 45
        wout = f"/tmp/sx3_seq_{run_tag}_t{idx}.json"
        if os.path.exists(wout):
            os.remove(wout)
        cmd = [sys.executable, _TIMEOUT_HELPER, str(hard), sys.executable,
               os.path.abspath(__file__), "--worker-theorem", str(idx),
               "--worker-out", wout, "--cases", args.cases, "--registry", args.registry,
               "--families", args.families or "",
               "--max-sequences-per-theorem", str(args.max_sequences_per_theorem),
               "--timeout-per-sequence", str(args.timeout_per_sequence),
               "--open-timeout", str(args.open_timeout)]
        if args.no_controls:
            cmd.append("--no-controls")
        print(f"[sx3:seq] ({idx+1}/{len(cases)}) {cases[idx]['full_name']} "
              f"seqs={nseq} ctl={n_ctl} hard={hard}s", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if os.path.exists(wout):
            try:
                rec = json.load(open(wout))
            except Exception as e:
                rec = {"full_name": cases[idx]["full_name"], "live": False,
                       "classification": "unknown",
                       "setup_error": f"unreadable worker out: {e}", "wins": []}
        else:
            rec = {"full_name": cases[idx]["full_name"], "live": False,
                   "classification": "unknown", "wins": [],
                   "setup_error": f"no worker output (rc={rc}); OS-killed at {hard}s"}
        results.append(rec)
        print(f"          -> live={rec.get('live')} class={rec.get('classification')} "
              f"best_win={rec.get('best_win')}", flush=True)
        checkpoint()

    final = checkpoint()
    write_md(final, args.out_md)
    print(f"[sx3:seq] DONE theorems={final['num_theorems']} live={final['num_live']} "
          f"hash={final['result_hash']} hist={final['classification_histogram']}")
    print(f"[sx3:seq] new_depth2_wins={final['new_depth2_wins']}")
    return 0


def write_md(agg, path):
    L = ["# SX3 Depth-2 Sequence Search — Live Results", ""]
    L.append(f"- cases: `{agg['cases_file']}`")
    L.append(f"- families: {', '.join(agg['families'])}")
    L.append(f"- theorems: {agg['num_theorems']} | live: {agg['num_live']} | "
             f"result_hash: `{agg['result_hash']}`")
    L.append(f"- classification histogram: `{agg['classification_histogram']}`")
    L.append(f"- new_depth2_wins ({len(agg['new_depth2_wins'])}): "
             f"{', '.join('`'+w+'`' for w in agg['new_depth2_wins']) or '(none)'}")
    L.append("")
    L.append("| theorem | live | shape | class | best win |")
    L.append("|---|---|---|---|---|")
    for r in sorted(agg["results"], key=lambda x: x.get("full_name", "")):
        L.append(f"| `{r['full_name']}` | {r.get('live')} | {r.get('shape','')} | "
                 f"**{r.get('classification')}** | `{(r.get('best_win') or '')[:38]}` |")
    L.append("")
    for r in sorted(agg["results"], key=lambda x: x.get("full_name", "")):
        L.append(f"## `{r['full_name']}`")
        if r.get("setup_error"):
            L.append(f"- setup_error: {r['setup_error'][:200]}")
        if r.get("initial_goal"):
            L.append(f"- initial goal: `{(r['initial_goal'] or '')[:200]}`")
        L.append(f"- classification: **{r.get('classification')}** | best_win=`{r.get('best_win')}`")
        if r.get("controls"):
            L.append("- controls:")
            for c in r["controls"]:
                L.append(f"    - `{c['tactic']}` -> {c['outcome']} (solved={c['solved']})")
        if r.get("gated_sequences_tried"):
            L.append("- gated sequences:")
            for s in r["gated_sequences_tried"]:
                L.append(f"    - [{s.get('family')}] `{s['tactic']}` -> {s['outcome']} "
                         f"(solved={s['solved']})")
        L.append("")
    L.append("> " + agg["note"])
    open(path, "w").write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--registry",
                   default="project/evolve/experiments/sx3/sx3_sequence_registry.json")
    p.add_argument("--families", default="")
    p.add_argument("--out-json", default="project/evolve/experiments/sx3/out/sx3_results.json")
    p.add_argument("--out-md", default="project/evolve/experiments/sx3/out/sx3_results.md")
    p.add_argument("--max-sequences-per-theorem", type=int, default=20)
    p.add_argument("--timeout-per-sequence", type=int, default=30)
    p.add_argument("--open-timeout", type=int, default=75,
                   help="max seconds for a single Dojo session open before aborting it")
    p.add_argument("--mode", default="live")
    p.add_argument("--rc2-baseline-results", default=None)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--worker-theorem", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    args = p.parse_args(argv)
    if args.worker_theorem is not None:
        return worker(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
