"""AX2 Stage 5 — minimal-symbolic relabeling of symbolic-only wins.

NS23/WX discipline: a wrapper "win" attributed to a symbolic CASES_SIMP may
actually be closable by a plain name-free tactic (simp / aesop). Such an
example is over-attributed and must NOT enter the symbolic-action training
set. For each symbolic-only-beyond-NS9 win (Stage 3), open a LeanDojo
session and try a battery from the initial state, simplest first:

  1 assumption  2 rfl  3 decide  4 simp  5 simp_all  6 aesop
  7 symbolic CASES_SIMP[ns,simp|simp_all]      (vars from the live state)
  8 symbolic INDUCTION_SIMP[ns,simp|simp_all]  (List only)
  9 raw instantiated tactic fallback           (the wrapper's winning tactic)

The first tactic that closes is the minimal tactic. `symbolic_action_needed`
is True only if a symbolic action (7/8) is the minimal closer — i.e. no
name-free tactic worked. `final_training_label` is the minimal symbolic
action id when needed, else a non-symbolic marker (dropped from training).

Outputs:
  project/data/ax2_minimal_symbolic_labels.json
  project/data/ax2_symbolic_family_pools_meta.json
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from project.evolve.symbolic_actions import (  # noqa: E402
    SymbolicAction, instantiate_symbolic_action)

PROBE = ROOT / "project/data/ax2_symbolic_mining_probe_meta.json"
OUT_LABELS = ROOT / "project/data/ax2_minimal_symbolic_labels.json"
OUT_POOLS = ROOT / "project/data/ax2_symbolic_family_pools_meta.json"

PER_TACTIC_TIMEOUT_S = 30
PER_THEOREM_TIMEOUT_S = 400
GATE_UNIQUE_REQUIRED = 5

NAME_FREE = [
    ("assumption", "assumption"), ("rfl", "fallback_rfl"),
    ("decide", "fallback_decide"), ("simp", "simp_other"),
    ("simp_all", "simp_all"), ("aesop", "aesop"),
]
_NS_VARTYPE = {"List": "List", "Option": "Option", "Bool": "Bool"}


class _Timeout(Exception):
    pass


def _handler(_s, _f):
    raise _Timeout()


def _try(dojo, state, tactic, t_s):
    from lean_dojo import ProofFinished
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(t_s)
    try:
        return isinstance(dojo.run_tac(state, tactic), ProofFinished)
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)


def _symbolic_battery(state_pp, namespace):
    """[(tactic, family, action_id, var)] in minimal-first order."""
    out = []
    var_type = _NS_VARTYPE.get(namespace)
    if var_type is None:
        return out
    # CASES first (cheaper), simp before simp_all; INDUCTION last (List only).
    specs = [("CASES_SIMP", "simp"), ("CASES_SIMP", "simp_all")]
    if var_type == "List":
        specs += [("INDUCTION_SIMP", "simp"), ("INDUCTION_SIMP", "simp_all")]
    for action_type, mode in specs:
        act = SymbolicAction(action_type=action_type, var_type=var_type,
                             simp_mode=mode, namespace_gate=var_type,
                             max_vars=2, priority=40)
        for tac, fam, action_id in instantiate_symbolic_action(act, state_pp):
            var = tac.split()[1] if len(tac.split()) > 1 else None
            out.append((tac, fam, action_id, var))
    return out


def relabel_one(cfg):
    from lean_dojo import Dojo, Theorem
    from env import make_repo
    repo = make_repo()
    thm = Theorem(repo=repo, file_path=cfg["file_path"],
                  full_name=cfg["full_name"])
    wrapper_origin = cfg.get("winning_origin")
    res = {
        "full_name": cfg["full_name"], "file_path": cfg["file_path"],
        "namespace": cfg["namespace"],
        "wrapper_winning_tactic": cfg.get("winning_tactic"),
        "wrapper_action_id": cfg.get("symbolic_action_id"),
        "wrapper_origin": wrapper_origin,
        "battery_results": [],
        "minimal_tactic": None, "minimal_family": None, "minimal_var": None,
        "minimal_is_short_token": None,
        "minimal_symbolic_action_id": None,
        "symbolic_action_needed": None,
        "single_shot_closable": None,
        "symbolic_in_winning_path": wrapper_origin == "wrapper_symbolic_action",
        "final_training_label": None,
        "dojo_status": "ok",
    }
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            state_pp = getattr(init, "pp", "") or ""
            # name-free tactics (short tokens)
            for tac, fam in NAME_FREE:
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "kind": "name_free",
                     "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_is_short_token=True,
                               symbolic_action_needed=False,
                               final_training_label=f"NON_SYMBOLIC[{fam}]")
            # symbolic actions (only matters if no name-free closed)
            for tac, fam, action_id, var in _symbolic_battery(
                    state_pp, cfg["namespace"]):
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "action_id": action_id,
                     "kind": "symbolic", "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_var=var,
                               minimal_is_short_token=False,
                               minimal_symbolic_action_id=action_id,
                               symbolic_action_needed=True,
                               final_training_label=action_id)
            # raw wrapper fallback
            if res["minimal_tactic"] is None:
                wt = (cfg.get("winning_tactic") or "").strip()
                ok = bool(wt) and _try(dojo, init, wt, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": wt, "family": "wrapper_original",
                     "kind": "raw_fallback", "finished": ok})
                if ok:
                    res.update(minimal_tactic=wt,
                               minimal_family="wrapper_original",
                               minimal_is_short_token=False,
                               symbolic_action_needed=True,
                               final_training_label=(cfg.get("symbolic_action_id")
                                                     or "RAW_ONLY"))
            # classify single-shot vs multi-step
            res["single_shot_closable"] = res["minimal_tactic"] is not None
            if res["minimal_tactic"] is None:
                # No single tactic closes from init: a multi-step search proof.
                # Not a clean single-action training example. Record whether the
                # symbolic action was in the winning path (probe attribution).
                if res["symbolic_in_winning_path"]:
                    res["final_training_label"] = (
                        f"MULTISTEP_SYMBOLIC[{cfg.get('symbolic_action_id')}]")
                else:
                    res["final_training_label"] = "MULTISTEP_NONSYMBOLIC"
    except _Timeout:
        res["dojo_status"] = "theorem_timeout"
    except Exception as exc:  # noqa: BLE001
        res["dojo_status"] = f"dojo_error: {str(exc)[:120]}"
    finally:
        signal.alarm(0)
    return res


def _osf(u):
    return 20 if u <= 1 else 15 if u <= 3 else 10 if u <= 6 else 5 if u <= 12 else 2


def write_pools(results):
    pools = defaultdict(lambda: {"thms": {}})
    for r in results:
        if not r.get("symbolic_action_needed"):
            continue
        label = r.get("final_training_label")
        if not label or label.startswith("NON_SYMBOLIC"):
            continue
        pools[label]["thms"][r["full_name"]] = {
            "minimal_tactic": r["minimal_tactic"],
            "minimal_var": r.get("minimal_var")}
    fam_out = {}
    for label, info in pools.items():
        u = len(info["thms"])
        fam_out[label] = {
            "symbolic_action_id": label, "unique_count": u,
            "count_gate_met": u >= GATE_UNIQUE_REQUIRED,
            "recommended_oversample_factor": _osf(u),
            "theorems": info["thms"]}
    fam_out = dict(sorted(fam_out.items(), key=lambda kv: -kv[1]["unique_count"]))
    needed = sum(1 for r in results if r.get("symbolic_action_needed"))
    dropped = sum(1 for r in results
                  if r.get("symbolic_action_needed") is False)
    single_shot = sum(1 for r in results if r.get("single_shot_closable"))
    multistep_sym = sum(1 for r in results
                        if r.get("single_shot_closable") is False
                        and r.get("symbolic_in_winning_path"))
    multistep_nonsym = sum(1 for r in results
                           if r.get("single_shot_closable") is False
                           and not r.get("symbolic_in_winning_path"))
    meta = {
        "gate_unique_required": GATE_UNIQUE_REQUIRED,
        "total_relabeled": len(results),
        "single_shot_closable": single_shot,
        "clean_single_shot_symbolic_needed": needed,
        "dropped_simpler_tactic_closes_single_shot": dropped,
        "multistep_symbolic_assisted": multistep_sym,
        "multistep_nonsymbolic": multistep_nonsym,
        "note": ("clean_single_shot_symbolic_needed counts examples where a "
                 "single symbolic CASES/INDUCTION action closes from init and "
                 "no name-free tactic does (the AX3-trainable label class). "
                 "multistep_symbolic_assisted wins used the symbolic action "
                 "inside a multi-step search but no single tactic closes them "
                 "-> weak labels, excluded from clean pools."),
        "unresolved": [r["full_name"] for r in results
                       if not r.get("minimal_tactic")],
        "any_label_gate_met": any(v["count_gate_met"] for v in fam_out.values()),
        "symbolic_label_pools": fam_out,
    }
    OUT_POOLS.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nwrote {OUT_POOLS.relative_to(ROOT)}")
    for k, v in fam_out.items():
        gate = "GATE" if v["count_gate_met"] else " -- "
        print(f"  [{gate}] {k}: {v['unique_count']}")
    print(f"clean_single_shot_symbolic={needed} "
          f"multistep_symbolic_assisted={multistep_sym} "
          f"multistep_nonsymbolic={multistep_nonsym} "
          f"any_gate_met={meta['any_label_gate_met']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROBE))
    ap.add_argument("--output", default=str(OUT_LABELS))
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    src = json.load(open(args.input))
    thms = src["symbolic_only_beyond_ns9_theorems"]
    out_path = Path(args.output)
    results, done = [], set()
    if out_path.exists():
        try:
            results = json.load(open(out_path)).get("relabel_results", [])
            done = {r["full_name"] for r in results}
            print(f"resuming from {len(done)} done")
        except Exception:
            pass
    for i, t in enumerate(thms):
        if t["full_name"] in done:
            continue
        print(f"[{i+1}/{len(thms)}] {t['full_name']}", flush=True)
        r = relabel_one(t)
        print(f"  -> minimal={r['minimal_tactic']} "
              f"needed={r['symbolic_action_needed']} "
              f"label={r['final_training_label']}", flush=True)
        results.append(r)
        if (i + 1) % args.checkpoint_every == 0:
            json.dump({"relabel_results": results}, open(out_path, "w"),
                      indent=2, ensure_ascii=False)
    json.dump({"relabel_results": results}, open(out_path, "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} ({len(results)} results)")
    write_pools(results)


if __name__ == "__main__":
    main()
