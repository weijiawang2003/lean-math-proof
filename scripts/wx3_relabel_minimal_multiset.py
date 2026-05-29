"""WX3 Stage 8 — minimal-tactic/action relabeling of WX3-only Multiset wins.

NS23/WX discipline: a wrapper "win" may actually be closable by a plain
name-free tactic, or it may need a full multi-step search and not be a clean
single-action label. For every WX3-only-beyond-NS9 win, open a LeanDojo
session and try a battery from the INITIAL state, simplest first:

  1 assumption  2 rfl  3 decide  4 simp  5 simp_all  6 aesop
  7 ext x <;> simp            8 ext x <;> simp_all
  9 cases {m} <;> simp        10 cases {m} <;> simp_all
  11 induction {m} using Multiset.induction_on <;> simp
  12 induction {m} using Multiset.induction_on <;> simp_all
  13 wrapper winning tactic fallback

({m} ranges over Multiset variables read from the live state.)

The first tactic that closes is the minimal closer. A win is:
  - single-shot symbolic (wrapper-ready + SFT-ready) iff a Multiset symbolic
    action (7-12) is the minimal closer AND no name-free tactic (1-6) closed.
  - over-attributed (dropped) iff a name-free tactic closes it single-shot.
  - multi-step-assisted iff nothing single-shot closes but the wrapper found a
    proof (symbolic action used inside a multi-step search) — weak label.

Outputs:
  project/data/wx3_minimal_multiset_labels.json
  project/data/wx3_multiset_family_pools_meta.json
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
from project.evolve.state_vars import vars_of_type  # noqa: E402

PROBE = ROOT / "project/data/wx3_multiset_probe_meta.json"
OUT_LABELS = ROOT / "project/data/wx3_minimal_multiset_labels.json"
OUT_POOLS = ROOT / "project/data/wx3_multiset_family_pools_meta.json"

PER_TACTIC_TIMEOUT_S = 30
PER_THEOREM_TIMEOUT_S = 500
GATE_UNIQUE_REQUIRED = 5
GATE_FAMILY_REQUIRED = 20  # one-family symbolic-learning gate

NAME_FREE = [
    ("assumption", "assumption"), ("rfl", "fallback_rfl"),
    ("decide", "fallback_decide"), ("simp", "simp_other"),
    ("simp_all", "simp_all"), ("aesop", "aesop"),
]


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


def _symbolic_battery(state_pp):
    """[(tactic, family, action_id, var)] minimal-first for Multiset.

    ext (var-independent, gated on a Multiset var being present) first, then
    cases, then quotient induction; simp before simp_all within each.
    """
    out = []
    ms_vars = vars_of_type(state_pp, "Multiset", max_vars=2)
    if not ms_vars:
        return out
    # ext: emitted once per mode, gated on a Multiset var present.
    for mode in ("simp", "simp_all"):
        out.append((f"ext x <;> {mode}", f"multiset_ext_{mode}",
                    f"EXT_SIMP[Multiset,{mode}]", None))
    # cases
    for mode in ("simp", "simp_all"):
        for v in ms_vars:
            out.append((f"cases {v} <;> {mode}", f"multiset_cases_{mode}",
                        f"CASES_SIMP[Multiset,{mode}]", v))
    # quotient induction
    for mode in ("simp", "simp_all"):
        for v in ms_vars:
            out.append((
                f"induction {v} using Multiset.induction_on <;> {mode}",
                f"multiset_induction_{mode}",
                f"MULTISET_INDUCTION_SIMP[Multiset,{mode}]", v))
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
        "namespace": cfg["full_name"].split(".")[0],
        "set": cfg.get("set"),
        "raw_solved": cfg.get("raw_solved"),
        "ns9_solved": cfg.get("ns9_solved"),
        "wx3_solved": True,
        "wrapper_winning_tactic": cfg.get("winning_tactic"),
        "wrapper_origin": wrapper_origin,
        "wrapper_family": cfg.get("winning_family"),
        "battery_results": [],
        "minimal_tactic": None, "minimal_family": None, "minimal_var": None,
        "minimal_action_id": None,
        "minimal_is_name_free": None,
        "symbolic_action_needed": None,
        "single_shot_symbolic": None,
        "multi_step_assisted": None,
        "wrapper_ready": None,
        "sft_ready": None,
        "symbolic_in_winning_path": wrapper_origin == "wrapper_symbolic_action",
        "final_label": None,
        "dojo_status": "ok",
    }
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            state_pp = getattr(init, "pp", "") or ""
            for tac, fam in NAME_FREE:
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "kind": "name_free",
                     "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_is_name_free=True,
                               symbolic_action_needed=False,
                               final_label=f"NON_SYMBOLIC[{fam}]")
            for tac, fam, action_id, var in _symbolic_battery(state_pp):
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "action_id": action_id,
                     "kind": "symbolic", "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_var=var, minimal_action_id=action_id,
                               minimal_is_name_free=False,
                               symbolic_action_needed=True,
                               final_label=action_id)
            if res["minimal_tactic"] is None:
                wt = (cfg.get("winning_tactic") or "").strip()
                ok = bool(wt) and _try(dojo, init, wt, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": wt, "family": "wrapper_original",
                     "kind": "raw_fallback", "finished": ok})
                if ok:
                    res.update(minimal_tactic=wt,
                               minimal_family="wrapper_original",
                               minimal_is_name_free=False,
                               symbolic_action_needed=True,
                               final_label="RAW_ONLY")

            single_shot = res["minimal_tactic"] is not None
            sym_needed = bool(res["symbolic_action_needed"])
            res["single_shot_symbolic"] = single_shot and sym_needed
            res["multi_step_assisted"] = (not single_shot) and \
                res["symbolic_in_winning_path"]
            # wrapper-ready: the symbolic action closes (single-shot) and is
            # the minimal closer; SFT-ready: same + a stable action id label.
            res["wrapper_ready"] = res["single_shot_symbolic"]
            res["sft_ready"] = bool(res["single_shot_symbolic"]
                                    and res["minimal_action_id"])
            if not single_shot:
                res["final_label"] = (
                    f"MULTISTEP_SYMBOLIC[{res['wrapper_family']}]"
                    if res["symbolic_in_winning_path"]
                    else "MULTISTEP_NONSYMBOLIC")
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
        if not r.get("sft_ready"):
            continue
        label = r.get("minimal_action_id")
        if not label:
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
    fam_out = dict(sorted(fam_out.items(),
                          key=lambda kv: -kv[1]["unique_count"]))

    single_shot_sym = sum(1 for r in results if r.get("single_shot_symbolic"))
    dropped = sum(1 for r in results
                  if r.get("symbolic_action_needed") is False)
    multistep_sym = sum(1 for r in results if r.get("multi_step_assisted"))
    multistep_nonsym = sum(
        1 for r in results
        if r.get("single_shot_symbolic") is False
        and not r.get("symbolic_in_winning_path")
        and r.get("symbolic_action_needed") is not False)
    biggest = max((v["unique_count"] for v in fam_out.values()), default=0)
    meta = {
        "gate_unique_required": GATE_UNIQUE_REQUIRED,
        "gate_one_family_required": GATE_FAMILY_REQUIRED,
        "total_relabeled": len(results),
        "clean_single_shot_symbolic": single_shot_sym,
        "sft_ready_count": sum(1 for r in results if r.get("sft_ready")),
        "wrapper_ready_count": sum(1 for r in results
                                   if r.get("wrapper_ready")),
        "dropped_simpler_tactic_closes_single_shot": dropped,
        "multistep_symbolic_assisted": multistep_sym,
        "multistep_nonsymbolic": multistep_nonsym,
        "biggest_single_family": biggest,
        "any_unique_gate_met": any(v["count_gate_met"]
                                   for v in fam_out.values()),
        "one_family_gate_met": biggest >= GATE_FAMILY_REQUIRED,
        "forty_label_gate_met": single_shot_sym >= 40,
        "note": ("clean_single_shot_symbolic = a single Multiset symbolic "
                 "action (ext/cases/induction_on) closes from init and no "
                 "name-free tactic does (the AX3-trainable class). "
                 "multistep_symbolic_assisted = symbolic action used in a "
                 "multi-step proof but no single tactic closes -> weak "
                 "label, excluded from clean pools."),
        "unresolved_no_single_shot": [r["full_name"] for r in results
                                      if not r.get("minimal_tactic")],
        "symbolic_label_pools": fam_out,
    }
    OUT_POOLS.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nwrote {OUT_POOLS.relative_to(ROOT)}")
    for k, v in fam_out.items():
        gate = "GATE" if v["count_gate_met"] else " -- "
        print(f"  [{gate}] {k}: {v['unique_count']}")
    print(f"clean_single_shot_symbolic={single_shot_sym} "
          f"dropped={dropped} multistep_symbolic={multistep_sym} "
          f"multistep_nonsym={multistep_nonsym} "
          f"biggest_family={biggest}")


def _run_wins(tag, s):
    import glob
    ms = sorted(glob.glob(
        f"project/evolve/eval_runs/wx3_{tag}_{s}/eval-*/metrics.json"))
    if not ms:
        return set()
    d = json.load(open(ms[0]))
    return {t["full_name"] for t in d.get("per_theorem", [])
            if t.get("finished")}


def load_targets(input_path, best=None):
    src = json.load(open(input_path))
    best = best or src.get("best_config", "comb")
    wins = src.get("wx3_only_wins", {}).get(best, [])
    # annotate each target with raw_solved / ns9_solved (ns9_solved is False
    # by construction for WX3-only-beyond-NS9, confirmed here).
    raw_cache, ns9_cache = {}, {}
    for w in wins:
        s = w.get("set")
        if s not in raw_cache:
            raw_cache[s] = _run_wins("raw", s)
            ns9_cache[s] = _run_wins("ns9", s)
        w["raw_solved"] = w["full_name"] in raw_cache[s]
        w["ns9_solved"] = w["full_name"] in ns9_cache[s]
    return best, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROBE))
    ap.add_argument("--output", default=str(OUT_LABELS))
    ap.add_argument("--config", default=None,
                    help="which WX3 config's only-wins to relabel "
                         "(default: probe meta best_config)")
    ap.add_argument("--checkpoint-every", type=int, default=3)
    args = ap.parse_args()

    best, thms = load_targets(args.input, args.config)
    print(f"relabeling WX3-only wins for config={best}: {len(thms)} theorems")
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
              f"single_shot_symbolic={r['single_shot_symbolic']} "
              f"label={r['final_label']}", flush=True)
        results.append(r)
        if (i + 1) % args.checkpoint_every == 0:
            json.dump({"config": best, "relabel_results": results},
                      open(out_path, "w"), indent=2, ensure_ascii=False)
    json.dump({"config": best, "relabel_results": results},
              open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} ({len(results)} results)")
    write_pools(results)


if __name__ == "__main__":
    main()
