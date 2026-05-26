"""NS23 Stage 3 — relabel wrapper-only wins by minimal sufficient tactic.

For each theorem in `ns23_wrapper_only_wins_raw_meta.json`, opens a
LeanDojo session and tries a battery of tactics in order from simple
to complex against the initial state. Stops at the first success.
Records the minimal_tactic and a derived minimal_family.

Battery (simple → complex). Stops at first proof_finished:

  1. assumption                                — hypothesis match
  2. rfl                                        — reflexivity
  3. decide                                     — decidable
  4. omega                                      — linear arith
  5. norm_num                                   — numerical
  6. simp                                       — basic simp
  7. simp_all                                   — full simp
  8. aesop                                      — general
  9. constructor <;> omega                      — iff with omega
 10. constructor <;> simp_all                   — iff with simp
 11. split_ifs <;> omega                        — conditional + arith
 12. exact ⟨fun h => by omega, fun h => by omega⟩ — the iff-pair (control)
 13. wrapper original tactic                    — fallback (always last)

For each theorem we record which tactics succeeded (not just the
first) so the report can show "omega and aesop both work" vs
"only aesop works" patterns.

Outputs:
  project/data/ns23_minimal_tactic_labels.json
  project/data/ns23_minimal_family_pools_meta.json
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any


# Battery ordered simple → complex.
BATTERY: list[tuple[str, str, str]] = [
    # (tactic, family_label, kind)
    ("assumption",  "assumption",       "trivial"),
    ("rfl",         "fallback_rfl",     "trivial"),
    ("decide",      "fallback_decide",  "trivial"),
    ("omega",       "fallback_omega",   "arith"),
    ("norm_num",    "norm_num",         "arith"),
    ("simp",        "simp_other",       "simp"),
    ("simp_all",    "simp_all",         "simp"),
    ("aesop",       "aesop",            "general"),
    ("constructor <;> omega",
     "constructor_omega", "iff_arith"),
    ("constructor <;> simp_all",
     "constructor_simp_all", "iff_simp"),
    ("split_ifs <;> omega",
     "split_ifs_omega",  "conditional_arith"),
    ("exact ⟨fun h => by omega, fun h => by omega⟩",
     "iff_omega_pair",   "iff_pair_omega"),
]


PER_TACTIC_TIMEOUT_S = 60   # individual tactic
PER_THEOREM_TIMEOUT_S = 600  # whole battery + dojo open


class _Timeout(Exception):
    pass


def _handler(_signum, _frame):
    raise _Timeout()


def _try_tactic(dojo: Any, initial_state: Any, tactic: str,
                timeout_s: int) -> dict:
    """Run a single tactic on the initial state. Returns a dict with:
       finished (bool), error (str|None), kind (str).
    """
    from lean_dojo import ProofFinished, TacticState
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_s)
    try:
        result = dojo.run_tac(initial_state, tactic)
    except _Timeout:
        return {"finished": False, "error": "timeout", "kind": "Timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"finished": False, "error": str(exc)[:140],
                "kind": "PythonError"}
    finally:
        signal.alarm(0)
    finished = isinstance(result, ProofFinished)
    is_state = isinstance(result, TacticState)
    err = None
    if not finished and not is_state:
        err = (getattr(result, "message", None)
               or getattr(result, "error", None)
               or str(result))
        if err:
            err = err[:140]
    return {
        "finished": finished,
        "error": err,
        "kind": type(result).__name__,
    }


def relabel_one(theorem_cfg: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lean_dojo import Dojo, Theorem
    from env import make_repo

    repo = make_repo()
    thm = Theorem(
        repo=repo,
        file_path=theorem_cfg["file_path"],
        full_name=theorem_cfg["full_name"],
    )

    result = {
        "full_name": theorem_cfg["full_name"],
        "file_path": theorem_cfg["file_path"],
        "namespace": theorem_cfg["namespace"],
        "original_family": theorem_cfg["original_family"],
        "wrapper_tactic": theorem_cfg.get("wrapper_tactic", ""),
        "battery_results": [],
        "minimal_tactic": None,
        "minimal_family": None,
        "minimal_kind": None,
        "wrapper_tactic_succeeds": None,
        "dojo_status": "ok",
        "error": None,
    }

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(PER_THEOREM_TIMEOUT_S)
    try:
        with Dojo(thm) as (dojo, initial_state):
            for tac, family, kind in BATTERY:
                outcome = _try_tactic(dojo, initial_state, tac,
                                       PER_TACTIC_TIMEOUT_S)
                result["battery_results"].append({
                    "tactic": tac, "family": family, "kind": kind,
                    "finished": outcome["finished"],
                    "error_kind": outcome["kind"],
                })
                if outcome["finished"]:
                    if result["minimal_tactic"] is None:
                        result["minimal_tactic"] = tac
                        result["minimal_family"] = family
                        result["minimal_kind"] = kind
            # Also test the original wrapper tactic (control).
            wrap_t = theorem_cfg.get("wrapper_tactic", "").strip()
            if wrap_t and wrap_t not in {b[0] for b in BATTERY}:
                outcome = _try_tactic(dojo, initial_state, wrap_t,
                                       PER_TACTIC_TIMEOUT_S)
                result["wrapper_tactic_succeeds"] = outcome["finished"]
                if (result["minimal_tactic"] is None
                        and outcome["finished"]):
                    result["minimal_tactic"] = wrap_t
                    result["minimal_family"] = "wrapper_original"
                    result["minimal_kind"] = "wrapper_original"
    except _Timeout:
        result["dojo_status"] = "theorem_timeout"
        result["error"] = (f">{PER_THEOREM_TIMEOUT_S}s on whole-theorem "
                            "battery")
    except Exception as exc:  # noqa: BLE001
        result["dojo_status"] = "dojo_error"
        result["error"] = str(exc)[:200]
    finally:
        signal.alarm(0)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="project/data/ns23_wrapper_only_wins_raw_meta.json",
    )
    ap.add_argument(
        "--output",
        default="project/data/ns23_minimal_tactic_labels.json",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Write partial output every N theorems.",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N theorems (0 = all). Useful for smoke.",
    )
    args = ap.parse_args()

    src = json.load(open(args.input))
    thms = src["theorems"]
    if args.limit:
        thms = thms[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing partial output if any.
    results: list[dict] = []
    done_names: set[str] = set()
    if out_path.exists():
        try:
            prev = json.load(open(out_path))
            results = prev.get("relabel_results", [])
            done_names = {r["full_name"] for r in results}
            print(f"resuming from {len(done_names)} done")
        except Exception:
            pass

    for i, t in enumerate(thms):
        if t["full_name"] in done_names:
            continue
        print(f"[{i+1}/{len(thms)}] {t['full_name']} "
              f"(orig={t['original_family']})", flush=True)
        r = relabel_one(t)
        print(f"  → minimal_tactic={r['minimal_tactic']} "
              f"family={r['minimal_family']} "
              f"status={r['dojo_status']}",
              flush=True)
        results.append(r)
        if (i + 1) % args.checkpoint_every == 0:
            json.dump({"relabel_results": results}, open(out_path, "w"),
                      indent=2)
            print(f"  [checkpoint] wrote {len(results)} results", flush=True)

    json.dump({"relabel_results": results}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path} ({len(results)} results)")


if __name__ == "__main__":
    main()
