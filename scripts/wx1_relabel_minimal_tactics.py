"""WX1 Stage 6 — relabel WX1-only wins by minimal sufficient tactic.

For each theorem WX1 solves that NS9 did not (from
wx1_option_cases_probe_meta.json), open a LeanDojo session and try a
battery from the initial state, simplest first, including state-aware
`cases <var> <;> ...` forms whose variable is read from the initial
state (the same `_extract_cases_vars` the WX1 wrapper uses).

Battery (simple → complex):
  1. assumption   2. rfl   3. decide   4. simp   5. simp_all   6. aesop
  7. cases <opt_var> <;> simp        8. cases <opt_var> <;> simp_all
  9. cases <bool_var> <;> decide    10. cases <bool_var> <;> simp_all
 11. wrapper winning tactic (fallback)

The point: distinguish a SHORT-TOKEN minimal (simp/simp_all/aesop —
SFT-ready) from a STATE-AWARE compound minimal (`cases {var} <;> ...` —
wrapper-ready but not short-token SFT, per the CX3/NS22 finding).

Outputs:
  project/data/wx1_minimal_tactic_labels.json
  project/data/wx1_minimal_family_pools_meta.json
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

# Name-free portion (tactic, family, short_token?).
NAME_FREE = [
    ("assumption", "assumption", True),
    ("rfl", "fallback_rfl", True),
    ("decide", "fallback_decide", True),
    ("simp", "simp_other", True),
    ("simp_all", "simp_all", True),
    ("aesop", "aesop", True),
]
SHORT_TOKEN_FAMILIES = {
    "assumption", "fallback_rfl", "fallback_decide",
    "simp_other", "simp_all", "aesop", "norm_num",
}


class _Timeout(Exception):
    pass


def _handler(_s, _f):
    raise _Timeout()


def _try(dojo: Any, state: Any, tactic: str, t_s: int) -> bool:
    from lean_dojo import ProofFinished
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(t_s)
    try:
        result = dojo.run_tac(state, tactic)
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        signal.alarm(0)
    return isinstance(result, ProofFinished)


def _build_battery(state_pp: str) -> list[tuple[str, str, bool, str | None]]:
    """Return [(tactic, family, short_token, var_used)] in order."""
    bat: list[tuple[str, str, bool, str | None]] = [
        (t, f, st, None) for (t, f, st) in NAME_FREE
    ]
    opt_vars = _extract_cases_vars(state_pp, ["Option"], 2)
    bool_vars = _extract_cases_vars(state_pp, ["Bool"], 2)
    for v in opt_vars:
        bat.append((f"cases {v} <;> simp", "option_cases_simp", False, v))
        bat.append((f"cases {v} <;> simp_all", "option_cases_simp_all", False, v))
    for v in bool_vars:
        bat.append((f"cases {v} <;> decide", "bool_cases_decide", False, v))
        bat.append((f"cases {v} <;> simp_all", "bool_cases_simp_all", False, v))
    return bat


def relabel_one(cfg: dict) -> dict:
    from lean_dojo import Dojo, Theorem
    from env import make_repo

    repo = make_repo()
    thm = Theorem(repo=repo, file_path=cfg["file_path"],
                  full_name=cfg["full_name"])
    res = {
        "full_name": cfg["full_name"],
        "file_path": cfg["file_path"],
        "namespace": cfg["namespace"],
        "wx1_winning_tactic": cfg.get("wx1_winning_tactic"),
        "wx1_winning_origin": cfg.get("wx1_winning_origin"),
        "battery_results": [],
        "minimal_tactic": None,
        "minimal_family": None,
        "minimal_var": None,
        "minimal_is_short_token": None,
        "state_aware_stable": None,
        "dojo_status": "ok",
    }
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            battery = _build_battery(getattr(init, "pp", "") or "")
            for tac, fam, short_tok, var in battery:
                ok = _try(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append(
                    {"tactic": tac, "family": fam, "finished": ok})
                if ok and res["minimal_tactic"] is None:
                    res["minimal_tactic"] = tac
                    res["minimal_family"] = fam
                    res["minimal_var"] = var
                    res["minimal_is_short_token"] = short_tok
                    # state-aware stable = a cases form with a recovered var
                    res["state_aware_stable"] = (var is not None)
            if res["minimal_tactic"] is None:
                wt = (cfg.get("wx1_winning_tactic") or "").strip()
                if wt and _try(dojo, init, wt, PER_TACTIC_TIMEOUT_S):
                    res["minimal_tactic"] = wt
                    res["minimal_family"] = "wrapper_original"
                    res["minimal_is_short_token"] = False
                    res["state_aware_stable"] = "{var}" not in wt
    except _Timeout:
        res["dojo_status"] = "theorem_timeout"
    except Exception as exc:  # noqa: BLE001
        res["dojo_status"] = f"dojo_error: {str(exc)[:120]}"
    finally:
        signal.alarm(0)
    return res


def _osf(u: int) -> int:
    return 20 if u <= 1 else 15 if u <= 3 else 10 if u <= 6 else 5 if u <= 12 else 2


def write_pools(results: list[dict]) -> None:
    pools: dict[tuple[str, str], dict] = defaultdict(lambda: {"thms": {}})
    for r in results:
        if not r.get("minimal_family"):
            continue
        key = (r["minimal_family"], r["namespace"])
        pools[key]["thms"][r["full_name"]] = {
            "minimal_tactic": r["minimal_tactic"],
            "minimal_var": r.get("minimal_var"),
            "short_token": r.get("minimal_is_short_token"),
        }
    fam_out = {}
    for (fam, ns), info in pools.items():
        u = len(info["thms"])
        short = fam in SHORT_TOKEN_FAMILIES
        fam_out[f"{fam}|{ns}"] = {
            "minimal_family": fam,
            "namespace": ns,
            "unique_count": u,
            "is_short_token_family": short,
            "count_gate_met": u >= 5,
            "sft_ready": bool(short and u >= 5),
            "wrapper_ready": bool((not short) and u >= 5),
            "recommended_oversample_factor": _osf(u),
            "theorems": info["thms"],
        }
    fam_out = dict(sorted(fam_out.items(),
                          key=lambda kv: -kv[1]["unique_count"]))
    sft = {k: v for k, v in fam_out.items() if v["sft_ready"]}
    wrap = {k: v for k, v in fam_out.items() if v["wrapper_ready"]}
    meta = {
        "training_gate_unique_required": 5,
        "total_relabeled": len(results),
        "resolved": sum(1 for r in results if r.get("minimal_family")),
        "unresolved": [r["full_name"] for r in results
                       if not r.get("minimal_family")],
        "sft_ready_families": list(sft),
        "wrapper_ready_families": list(wrap),
        "any_sft_gate_met": bool(sft),
        "any_wrapper_gate_met": bool(wrap),
        "families": fam_out,
    }
    Path("project/data/wx1_minimal_family_pools_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote project/data/wx1_minimal_family_pools_meta.json")
    for k, v in fam_out.items():
        tag = ("SFT" if v["sft_ready"] else
               "WRAP" if v["wrapper_ready"] else " -- ")
        print(f"  [{tag}] {k}: {v['unique_count']} unique "
              f"(short_token={v['is_short_token_family']})")
    print(f"any SFT gate: {meta['any_sft_gate_met']}  "
          f"any wrapper gate: {meta['any_wrapper_gate_met']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", default="project/data/wx1_option_cases_probe_meta.json")
    ap.add_argument(
        "--output", default="project/data/wx1_minimal_tactic_labels.json")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    src = json.load(open(args.input))
    thms = src["wx1_only_theorems"]
    out_path = Path(args.output)
    results: list[dict] = []
    done: set[str] = set()
    if out_path.exists():
        try:
            prev = json.load(open(out_path))
            results = prev.get("relabel_results", [])
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
              f"short_token={r['minimal_is_short_token']}", flush=True)
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
