"""AX3 Stage 4 — minimal-symbolic relabeling of AX3 WX3-only Multiset wins.

Same NS23/WX discipline and battery as WX3 Stage 8, applied to the AX3
mining/held-out WX3-only-beyond-NS9 wins. Additionally captures the initial
`state_pp` so Stage 5 can build classifier features without re-opening Dojo.

Battery (minimal-first): assumption, rfl, decide, simp, simp_all, aesop,
ext×2, cases×2, induction_on×2, wrapper-fallback.

A win is a clean single-shot symbolic label iff a Multiset symbolic action is
the minimal closer AND no name-free tactic closes it from init.

Outputs:
  project/data/ax3_minimal_multiset_symbolic_labels.json
  project/data/ax3_multiset_symbolic_family_pools_meta.json
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

PROBE = ROOT / "project/data/ax3_multiset_mining_probe_meta.json"
OUT_LABELS = ROOT / "project/data/ax3_minimal_multiset_symbolic_labels.json"
OUT_POOLS = ROOT / "project/data/ax3_multiset_symbolic_family_pools_meta.json"

PER_TACTIC_TIMEOUT_S = 30
PER_THEOREM_TIMEOUT_S = 500
GATE_UNIQUE_REQUIRED = 5

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
    out = []
    ms_vars = vars_of_type(state_pp, "Multiset", max_vars=2)
    if not ms_vars:
        return out
    for mode in ("simp", "simp_all"):
        out.append((f"ext x <;> {mode}", f"multiset_ext_{mode}",
                    f"EXT_SIMP[Multiset,{mode}]", None))
    for mode in ("simp", "simp_all"):
        for v in ms_vars:
            out.append((f"cases {v} <;> {mode}", f"multiset_cases_{mode}",
                        f"CASES_SIMP[Multiset,{mode}]", v))
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
        "set": cfg.get("set"), "state_pp": None,
        "raw_solved": cfg.get("raw_solved"),
        "ns9_solved": cfg.get("ns9_solved"), "wx3_solved": True,
        "wrapper_winning_tactic": cfg.get("winning_tactic"),
        "wrapper_origin": wrapper_origin,
        "wrapper_family": cfg.get("winning_family"),
        "battery_results": [],
        "minimal_tactic": None, "minimal_family": None, "minimal_var": None,
        "minimal_action_id": None, "minimal_is_name_free": None,
        "symbolic_action_needed": None, "single_shot_symbolic": None,
        "multi_step_assisted": None, "over_attributed": None,
        "wrapper_ready": None, "sft_ready": None,
        "symbolic_in_winning_path": wrapper_origin == "wrapper_symbolic_action",
        "final_label": None, "dojo_status": "ok",
    }
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            state_pp = getattr(init, "pp", "") or ""
            res["state_pp"] = state_pp
            for tac, fam in NAME_FREE:
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "kind": "name_free",
                     "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_is_name_free=True,
                               symbolic_action_needed=False,
                               over_attributed=True,
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
                               over_attributed=False,
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
                               over_attributed=False, final_label="RAW_ONLY")
            single_shot = res["minimal_tactic"] is not None
            sym_needed = bool(res["symbolic_action_needed"])
            res["single_shot_symbolic"] = single_shot and sym_needed
            res["multi_step_assisted"] = (not single_shot) and \
                res["symbolic_in_winning_path"]
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
        if label:
            pools[label]["thms"][r["full_name"]] = {
                "minimal_tactic": r["minimal_tactic"],
                "minimal_var": r.get("minimal_var")}
    fam_out = {}
    for label, info in pools.items():
        u = len(info["thms"])
        fam_out[label] = {"symbolic_action_id": label, "unique_count": u,
                          "count_gate_met": u >= GATE_UNIQUE_REQUIRED,
                          "recommended_oversample_factor": _osf(u),
                          "theorems": info["thms"]}
    fam_out = dict(sorted(fam_out.items(),
                          key=lambda kv: -kv[1]["unique_count"]))
    clean = sum(1 for r in results if r.get("single_shot_symbolic"))
    dropped = sum(1 for r in results if r.get("over_attributed"))
    multistep_sym = sum(1 for r in results if r.get("multi_step_assisted"))
    biggest = max((v["unique_count"] for v in fam_out.values()), default=0)
    meta = {
        "gate_unique_required": GATE_UNIQUE_REQUIRED,
        "total_relabeled": len(results),
        "ax3_new_clean_single_shot_symbolic": clean,
        "dropped_over_attributed": dropped,
        "multistep_symbolic_assisted": multistep_sym,
        "biggest_single_family": biggest,
        "symbolic_label_pools": fam_out,
        "clean_theorems": [
            {"full_name": r["full_name"], "set": r["set"],
             "minimal_action_id": r["minimal_action_id"],
             "minimal_tactic": r["minimal_tactic"],
             "minimal_var": r.get("minimal_var")}
            for r in results if r.get("single_shot_symbolic")],
    }
    OUT_POOLS.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nwrote {OUT_POOLS.relative_to(ROOT)}")
    for k, v in fam_out.items():
        g = "GATE" if v["count_gate_met"] else " -- "
        print(f"  [{g}] {k}: {v['unique_count']}")
    print(f"ax3_new_clean={clean} dropped={dropped} "
          f"multistep_symbolic={multistep_sym} biggest={biggest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROBE))
    ap.add_argument("--output", default=str(OUT_LABELS))
    ap.add_argument("--checkpoint-every", type=int, default=3)
    args = ap.parse_args()
    src = json.load(open(args.input))
    thms = src.get("wx3_only_wins", [])
    print(f"relabeling {len(thms)} AX3 WX3-only wins")
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
            json.dump({"relabel_results": results}, open(out_path, "w"),
                      indent=2, ensure_ascii=False)
    json.dump({"relabel_results": results}, open(out_path, "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} ({len(results)} results)")
    write_pools(results)


if __name__ == "__main__":
    main()
