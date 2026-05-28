"""CX3 Stage 5 — relabel Bool/Option wrapper-only wins by minimal tactic.

Mandatory post-NS23/NS24 step: the wrapper's `winning_tactic` is only a
hint. For every wrapper-only win found in CX3 (Stage 4), open a LeanDojo
session and try a battery of tactics from the initial state, simplest
first, and record the *minimal sufficient* tactic + family. This is what
a NS25 training pool would actually be labeled with.

Battery (simple → complex; stops recording the first success but runs
all to show "decide and simp both work" vs "only aesop"):

  1. assumption
  2. rfl
  3. decide                  — decidable (Bool props over finite domain)
  4. simp
  5. simp_all
  6. norm_num
  7. tauto
  8. aesop                   — general; subsumes Option none/some case split
  9. constructor <;> simp
 10. constructor <;> decide
 11. <parsed cases> <;> simp  — best-effort: intro+cases on a detected
                               Option/Bool binder (name-free fallback skips)
 12. wrapper original tactic  — always last

Input : project/data/cx3_bool_option_probe_meta.json  (wrapper_only_theorems)
Output: project/data/cx3_minimal_tactic_labels.json
        project/data/cx3_minimal_family_pools_meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# (tactic, family_label, kind) — name-free portion of the battery.
BATTERY: list[tuple[str, str, str]] = [
    ("assumption", "assumption", "trivial"),
    ("rfl", "fallback_rfl", "trivial"),
    ("decide", "fallback_decide", "decide"),
    ("simp", "simp_other", "simp"),
    ("simp_all", "simp_all", "simp"),
    ("norm_num", "norm_num", "arith"),
    ("tauto", "tauto", "logic"),
    ("aesop", "aesop", "general"),
    ("constructor <;> simp", "constructor_simp", "iff_simp"),
    ("constructor <;> decide", "constructor_decide", "iff_decide"),
]

PER_TACTIC_TIMEOUT_S = 30   # a "minimal cheap" tactic should close fast
PER_THEOREM_TIMEOUT_S = 400

_BINDER = re.compile(r"\(?\s*([A-Za-z_][A-Za-z0-9_']*)\s*:\s*"
                     r"(Option\b|Bool\b)")


class _Timeout(Exception):
    pass


def _handler(_s, _f):
    raise _Timeout()


def _parsed_cases_tactic(state_pp: str) -> str | None:
    """Best-effort: build `rintro ... ; cases <v> <;> simp` from a
    detected Option/Bool binder in the goal. Returns None if no safe
    candidate is found."""
    if not state_pp:
        return None
    m = _BINDER.search(state_pp)
    if not m:
        return None
    var = m.group(1)
    # intro everything, then case-split the detected variable.
    return f"intros <;> cases {var} <;> simp_all"


def _try_tactic(dojo: Any, state: Any, tactic: str, timeout_s: int) -> dict:
    from lean_dojo import ProofFinished, TacticState
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        result = dojo.run_tac(state, tactic)
    except _Timeout:
        return {"finished": False, "kind": "Timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"finished": False, "kind": "PythonError",
                "error": str(exc)[:140]}
    finally:
        signal.alarm(0)
    finished = isinstance(result, ProofFinished)
    return {"finished": finished, "kind": type(result).__name__}


def relabel_one(cfg: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lean_dojo import Dojo, Theorem
    from env import make_repo

    repo = make_repo()
    thm = Theorem(repo=repo, file_path=cfg["file_path"],
                  full_name=cfg["full_name"])

    res = {
        "full_name": cfg["full_name"],
        "file_path": cfg["file_path"],
        "namespace": cfg["namespace"],
        "original_family": cfg["original_family"],
        "wrapper_tactic": cfg.get("wrapper_tactic", ""),
        "currently_solved_raw": cfg.get("currently_solved_raw"),
        "currently_solved_wrap": cfg.get("currently_solved_wrap"),
        "expected_bucket": cfg.get("expected_bucket"),
        "battery_results": [],
        "minimal_tactic": None,
        "minimal_family": None,
        "minimal_kind": None,
        "wrapper_tactic_succeeds": None,
        "changed_label": None,
        "proof_success": False,
        "dojo_status": "ok",
        "error": None,
    }

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, init):
            battery = list(BATTERY)
            cases_t = _parsed_cases_tactic(getattr(init, "pp", "") or "")
            if cases_t:
                battery.append((cases_t, "cases_simp", "cases"))
            for tac, family, kind in battery:
                out = _try_tactic(dojo, init, tac, PER_TACTIC_TIMEOUT_S)
                res["battery_results"].append({
                    "tactic": tac, "family": family, "kind": kind,
                    "finished": out["finished"],
                    "result_kind": out["kind"],
                })
                if out["finished"] and res["minimal_tactic"] is None:
                    res["minimal_tactic"] = tac
                    res["minimal_family"] = family
                    res["minimal_kind"] = kind
                    res["proof_success"] = True
            # control: wrapper original tactic
            wt = (cfg.get("wrapper_tactic") or "").strip()
            if wt and wt not in {b[0] for b in battery}:
                out = _try_tactic(dojo, init, wt, PER_TACTIC_TIMEOUT_S)
                res["wrapper_tactic_succeeds"] = out["finished"]
                if res["minimal_tactic"] is None and out["finished"]:
                    res["minimal_tactic"] = wt
                    res["minimal_family"] = "wrapper_original"
                    res["minimal_kind"] = "wrapper_original"
                    res["proof_success"] = True
    except _Timeout:
        res["dojo_status"] = "theorem_timeout"
    except Exception as exc:  # noqa: BLE001
        res["dojo_status"] = "dojo_error"
        res["error"] = str(exc)[:200]
    finally:
        signal.alarm(0)

    if res["minimal_family"] is not None:
        res["changed_label"] = (res["minimal_family"] != res["original_family"])
    return res


def _osf(unique: int) -> int:
    if unique <= 1:
        return 20
    if unique <= 3:
        return 15
    if unique <= 6:
        return 10
    if unique <= 12:
        return 5
    return 2


def _fam_pools(results: list[dict], predicate) -> dict:
    pools: dict[tuple[str, str], dict] = defaultdict(lambda: {"thms": {}})
    for r in results:
        if not (r.get("proof_success") and r.get("minimal_family")):
            continue
        if not predicate(r):
            continue
        key = (r["minimal_family"], r["namespace"])
        pools[key]["thms"][r["full_name"]] = {
            "minimal_tactic": r["minimal_tactic"],
            "currently_solved_raw": r.get("currently_solved_raw"),
            "wrapper_tactic": r["wrapper_tactic"],
        }
    out = {}
    for (fam, ns), info in pools.items():
        unique = len(info["thms"])
        out[f"{fam}|{ns}"] = {
            "minimal_family": fam,
            "namespace": ns,
            "unique_count": unique,
            "trainable": unique >= 5,
            "recommended_oversample_factor": _osf(unique),
            "theorems": info["thms"],
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["unique_count"]))


def write_pools(results: list[dict]) -> None:
    # All resolved theorems, regardless of current solve status.
    all_pools = _fam_pools(results, lambda r: True)
    # HEADROOM = theorems the routed model does NOT currently solve but a
    # short battery tactic closes from the initial state. These are the
    # only theorems a NS25 specialist could *newly* capture.
    headroom_pools = _fam_pools(
        results, lambda r: not r.get("currently_solved_raw"))
    # Already-solved (characterization only; not a training opportunity).
    solved_pools = _fam_pools(
        results, lambda r: bool(r.get("currently_solved_raw")))

    unresolved = [r["full_name"] for r in results
                  if not r.get("proof_success")]
    headroom_total = sum(p["unique_count"] for p in headroom_pools.values())
    gate_families = {k: p for k, p in headroom_pools.items()
                     if p["trainable"]}

    meta = {
        "training_gate_unique_required": 5,
        "gate_basis": (
            "wrapper-only wins == 0 on Bool/Option (raw routed model "
            "already solves everything the NS9 wrapper does), so the "
            "wrapper-only gate is empty. Headroom = currently-unsolved "
            "theorems closed by a short tactic from the initial state — "
            "the only set a NS25 specialist could newly capture."
        ),
        "total_relabeled": len(results),
        "resolved_by_battery": sum(1 for r in results
                                   if r.get("proof_success")),
        "unresolved_by_battery": unresolved,
        "headroom_total": headroom_total,
        "headroom_gate_met": bool(gate_families),
        "headroom_gate_families": list(gate_families),
        "headroom_pools": headroom_pools,
        "already_solved_pools": solved_pools,
        "all_resolved_pools": all_pools,
    }
    Path("project/data/cx3_minimal_family_pools_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("\nwrote project/data/cx3_minimal_family_pools_meta.json")
    print(f"resolved by battery: {meta['resolved_by_battery']}/"
          f"{len(results)}  headroom (unsolved+closeable): "
          f"{headroom_total}")
    print("HEADROOM pools (potential NS25 capture):")
    for k, info in headroom_pools.items():
        gate = "GATE" if info["trainable"] else " -- "
        print(f"  [{gate}] {k}: {info['unique_count']} unique "
              f"(osf {info['recommended_oversample_factor']}x)")
    print(f"headroom gate met: {meta['headroom_gate_met']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", default="project/data/cx3_bool_option_probe_meta.json")
    ap.add_argument(
        "--output", default="project/data/cx3_minimal_tactic_labels.json")
    ap.add_argument(
        "--key", default="wrapper_only_theorems",
        help="probe-meta key to relabel (wrapper_only_theorems or "
             "relabel_candidates)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    src = json.load(open(args.input))
    thms = src[args.key]
    if args.limit:
        thms = thms[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"[{i+1}/{len(thms)}] {t['full_name']} "
              f"(orig={t['original_family']})", flush=True)
        r = relabel_one(t)
        print(f"  -> minimal={r['minimal_tactic']} "
              f"family={r['minimal_family']} "
              f"changed={r['changed_label']} status={r['dojo_status']}",
              flush=True)
        results.append(r)
        if (i + 1) % args.checkpoint_every == 0:
            json.dump({"relabel_results": results}, open(out_path, "w"),
                      indent=2)

    json.dump({"relabel_results": results}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path} ({len(results)} results)")
    write_pools(results)


if __name__ == "__main__":
    main()
