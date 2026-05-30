"""WX2 Stage 7 — relabel WX2-generalized-only wins by minimal tactic.

For each theorem the WX2 generalized wrapper solves that NS9 did not
(from wx2_generalized_cases_probe_meta.json), open a LeanDojo session and
try a battery from the initial state, simplest first, including
state-aware `cases/induction <var> <;> …` forms whose variable is read
from the state (type matched to the theorem's namespace).

Battery:
  1 assumption 2 rfl 3 decide 4 simp 5 simp_all 6 aesop
  7 cases <v> <;> decide   8 cases <v> <;> simp   9 cases <v> <;> simp_all
  10 induction <v> <;> simp 11 induction <v> <;> simp_all
  12 wrapper winning tactic (fallback)

Classifies each minimal family as short-token (SFT-ready) vs state-aware
compound (wrapper-ready only).

Outputs:
  project/data/wx2_minimal_tactic_labels.json
  project/data/wx2_minimal_family_pools_meta.json
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evolve.strategy_wrapper import _extract_cases_vars

PER_TACTIC_TIMEOUT_S = 30
PER_THEOREM_TIMEOUT_S = 400

NAME_FREE = [
    ("assumption", "assumption"), ("rfl", "fallback_rfl"),
    ("decide", "fallback_decide"), ("simp", "simp_other"),
    ("simp_all", "simp_all"), ("aesop", "aesop"),
]
SHORT_TOKEN_FAMILIES = {
    "assumption", "fallback_rfl", "fallback_decide",
    "simp_other", "simp_all", "aesop",
}
# namespace -> type matcher tokens for _extract_cases_vars
NS_TYPE = {
    "List": ["List"], "Option": ["Option"], "Bool": ["Bool"],
    "Prod": ["×"], "Sum": ["⊕"],
}


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


def _battery(state_pp, namespace):
    """[(tactic, family, short_token, var)]."""
    bat = [(t, f, True, None) for (t, f) in NAME_FREE]
    ns_low = namespace.lower()
    vars_ = _extract_cases_vars(state_pp, NS_TYPE.get(namespace, [namespace]), 2)
    for v in vars_:
        bat.append((f"cases {v} <;> decide", f"{ns_low}_cases_decide", False, v))
        bat.append((f"cases {v} <;> simp", f"{ns_low}_cases_simp", False, v))
        bat.append((f"cases {v} <;> simp_all", f"{ns_low}_cases_simp_all", False, v))
        bat.append((f"induction {v} <;> simp", f"{ns_low}_induction_simp", False, v))
        bat.append((f"induction {v} <;> simp_all", f"{ns_low}_induction_simp_all", False, v))
    return bat


def relabel_one(cfg):
    from lean_dojo import Dojo, Theorem
    from env import make_repo
    repo = make_repo()
    thm = Theorem(repo=repo, file_path=cfg["file_path"], full_name=cfg["full_name"])
    res = {
        "full_name": cfg["full_name"], "file_path": cfg["file_path"],
        "namespace": cfg["namespace"],
        "wx2_winning_tactic": cfg.get("winning_tactic"),
        "battery_results": [], "minimal_tactic": None, "minimal_family": None,
        "minimal_var": None, "minimal_is_short_token": None,
        "state_aware_stable": None, "dojo_status": "ok",
    }
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            for tac, fam, short_tok, var in _battery(
                    getattr(init, "pp", "") or "", cfg["namespace"]):
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res.update(minimal_tactic=tac, minimal_family=fam,
                               minimal_var=var, minimal_is_short_token=short_tok,
                               state_aware_stable=(var is not None))
            if res["minimal_tactic"] is None:
                wt = (cfg.get("winning_tactic") or "").strip()
                if wt and _try(dojo, init, wt, PER_TACTIC_TIMEOUT_S):
                    res.update(minimal_tactic=wt, minimal_family="wrapper_original",
                               minimal_is_short_token=False,
                               state_aware_stable="{var}" not in wt)
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
        if not r.get("minimal_family"):
            continue
        key = (r["minimal_family"], r["namespace"])
        pools[key]["thms"][r["full_name"]] = {
            "minimal_tactic": r["minimal_tactic"], "minimal_var": r.get("minimal_var")}
    fam_out = {}
    for (fam, ns), info in pools.items():
        u = len(info["thms"])
        short = fam in SHORT_TOKEN_FAMILIES
        fam_out[f"{fam}|{ns}"] = {
            "minimal_family": fam, "namespace": ns, "unique_count": u,
            "is_short_token_family": short, "count_gate_met": u >= 5,
            "sft_ready": bool(short and u >= 5),
            "wrapper_ready": bool((not short) and u >= 5),
            "recommended_oversample_factor": _osf(u), "theorems": info["thms"]}
    fam_out = dict(sorted(fam_out.items(), key=lambda kv: -kv[1]["unique_count"]))
    sft = {k: v for k, v in fam_out.items() if v["sft_ready"]}
    wrap = {k: v for k, v in fam_out.items() if v["wrapper_ready"]}
    meta = {
        "training_gate_unique_required": 5,
        "total_relabeled": len(results),
        "resolved": sum(1 for r in results if r.get("minimal_family")),
        "unresolved": [r["full_name"] for r in results if not r.get("minimal_family")],
        "sft_ready_families": list(sft), "wrapper_ready_families": list(wrap),
        "any_sft_gate_met": bool(sft), "any_wrapper_gate_met": bool(wrap),
        "families": fam_out}
    Path("project/data/wx2_minimal_family_pools_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote project/data/wx2_minimal_family_pools_meta.json")
    for k, v in fam_out.items():
        tag = "SFT" if v["sft_ready"] else "WRAP" if v["wrapper_ready"] else " -- "
        print(f"  [{tag}] {k}: {v['unique_count']} (short_token={v['is_short_token_family']})")
    print(f"any SFT gate: {meta['any_sft_gate_met']}  any wrapper gate: {meta['any_wrapper_gate_met']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="project/data/wx2_generalized_cases_probe_meta.json")
    ap.add_argument("--output", default="project/data/wx2_minimal_tactic_labels.json")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()
    src = json.load(open(args.input))
    thms = src["generalized_only_theorems"]
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
        print(f"  -> minimal={r['minimal_tactic']} family={r['minimal_family']} "
              f"short={r['minimal_is_short_token']}", flush=True)
        results.append(r)
        if (i + 1) % args.checkpoint_every == 0:
            json.dump({"relabel_results": results}, open(out_path, "w"), indent=2, ensure_ascii=False)
    json.dump({"relabel_results": results}, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path} ({len(results)} results)")
    write_pools(results)


if __name__ == "__main__":
    main()
