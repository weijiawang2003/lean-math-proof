"""Rollout the learned policy on every theorem in a set and report per-theorem results.

Supports both greedy (top-1) and top-k fallback rollout strategies.
In top-k mode, if the top-ranked tactic errors, the next-best tactic is tried
before advancing to the next step.

Supports two policy types:
  --policy-type classifier  (default) — fixed action space, policy.Policy
  --policy-type generative  — seq2seq tactic generation, generative_policy.GenerativePolicy

Usage:
  python eval_rollout_all.py --theorem-set demo_v1 --ckpt-dir clf_ckpt --top-k 5 --max-steps 8
  python eval_rollout_all.py --theorem-set demo_v1 --ckpt-dir gen_ckpt --policy-type generative
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from lean_dojo import Dojo

from env import make_repo, make_theorem, run_transition
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems, list_theorem_sets
from trace_io import append_jsonl


def _load_policy(policy_type: str, ckpt_dir: str, action_space: str = "search_v4"):
    """Load the appropriate policy based on type string."""
    if policy_type == "classifier":
        from policy import Policy
        return Policy(ckpt_dir=ckpt_dir)
    elif policy_type == "generative":
        from generative_policy import GenerativePolicy
        return GenerativePolicy(ckpt_dir=ckpt_dir)
    elif policy_type == "hybrid":
        from hybrid_policy import HybridPolicy
        return HybridPolicy(gen_ckpt_dir=ckpt_dir, action_space=action_space)
    elif policy_type == "strategic":
        from strategic_policy import StrategicPolicy
        return StrategicPolicy(
            base_policy="hybrid",
            gen_ckpt_dir=ckpt_dir,
            action_space=action_space,
        )
    elif policy_type == "premise_augmented":
        from generative_policy import PremiseAugmentedPolicy
        return PremiseAugmentedPolicy(ckpt_dir=ckpt_dir)
    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Use 'classifier', 'generative', 'hybrid', 'strategic', or 'premise_augmented'."
        )


def rollout_one_theorem(
    pol,
    theorem_cfg,
    repo,
    max_steps: int,
    top_k: int,
    run_id: str,
    traces_path: str,
    domain: str,
) -> dict:
    """Run rollout on a single theorem with top-k fallback, return result dict."""
    theorem = make_theorem(repo, theorem_cfg)
    episode_id = f"{theorem_cfg.full_name}:{run_id[-8:]}"

    result = {
        "full_name": theorem_cfg.full_name,
        "file_path": theorem_cfg.file_path,
        "available": False,
        "finished": False,
        "has_error": False,
        "num_steps": 0,
        "tactics_used": [],
        "error_message": None,
        "skip_reason": None,
        "winning_tactic": None,
        "fallbacks_used": 0,
    }

    try:
        with Dojo(theorem) as (dojo, state):
            result["available"] = True

            for step in range(1, max_steps + 1):
                ranked = pol.rank_tactics(state.pp, theorem.full_name, k=top_k)

                # Try tactics in ranked order until one doesn't error
                step_succeeded = False
                for rank, tac in enumerate(ranked):
                    outcome = run_transition(
                        dojo, theorem, state, tac,
                        step=step,
                        domain=domain,
                        run_id=run_id,
                        episode_id=episode_id,
                        method="policy_rollout_topk",
                    )
                    append_jsonl(traces_path, outcome.record)

                    # REPL crashed — Dojo is dead, abort theorem
                    if outcome.session_dead:
                        result["has_error"] = True
                        result["num_steps"] = step
                        result["error_message"] = f"REPL crashed on `{tac}` at step {step}"
                        break

                    if outcome.is_finished:
                        result["finished"] = True
                        result["winning_tactic"] = tac
                        result["num_steps"] = step
                        result["tactics_used"].append(tac)
                        if rank > 0:
                            result["fallbacks_used"] += 1
                        step_succeeded = True
                        break

                    if not outcome.is_error:
                        # Good transition — advance state
                        state = outcome.next_state
                        result["num_steps"] = step
                        result["tactics_used"].append(tac)
                        if rank > 0:
                            result["fallbacks_used"] += 1
                        step_succeeded = True
                        break

                    # This tactic errored — try next in ranking
                    continue

                if result["finished"] or outcome.session_dead:
                    break

                if not step_succeeded:
                    # All k tactics errored
                    result["has_error"] = True
                    result["num_steps"] = step
                    result["error_message"] = f"All top-{top_k} tactics errored at step {step}"
                    break

    except Exception as exc:
        result["skip_reason"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser(description="Rollout policy on all theorems in a set.")
    parser.add_argument("--theorem-set", default="nat_single", choices=list_theorem_sets())
    parser.add_argument("--ckpt-dir", default="clf_ckpt")
    parser.add_argument("--policy-type", default="classifier",
                        choices=["classifier", "generative", "hybrid", "strategic", "premise_augmented"],
                        help="Policy type: 'classifier', 'generative', 'hybrid', 'strategic', or 'premise_augmented'.")
    parser.add_argument("--action-space", default="search_v4",
                        help="Action space for hybrid policy fallback.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Try up to k tactics per step before declaring failure.")
    parser.add_argument("--domain", default="mathlib4")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    theorems = get_theorems(args.theorem_set)
    print(f"\n{'='*64}")
    print(f"  LEAN TACTIC POLICY EVALUATION")
    print(f"{'='*64}")
    print(f"  Theorem set : {args.theorem_set} ({len(theorems)} theorems)")
    print(f"  Checkpoint  : {args.ckpt_dir}")
    print(f"  Policy type : {args.policy_type}")
    print(f"  Strategy    : top-{args.top_k} fallback, max {args.max_steps} steps")
    print(f"{'='*64}\n")

    pol = _load_policy(args.policy_type, args.ckpt_dir,
                       action_space=getattr(args, "action_space", "search_v4"))
    repo = make_repo()

    run_id = f"eval-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="policy_rollout_topk",
        run_id=run_id,
        config={
            "method": "policy_rollout_topk",
            "policy_type": args.policy_type,
            "theorem_set": args.theorem_set,
            "ckpt_dir": args.ckpt_dir,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "num_theorems": len(theorems),
        },
    )

    results = []
    for i, cfg in enumerate(theorems):
        print(f"[{i+1}/{len(theorems)}] {cfg.full_name}")

        r = rollout_one_theorem(
            pol, cfg, repo,
            max_steps=args.max_steps,
            top_k=args.top_k,
            run_id=run_id,
            traces_path=artifacts["traces_path"],
            domain=args.domain,
        )
        results.append(r)

        if not r["available"]:
            reason = (r["skip_reason"] or "unknown")[:60]
            print(f"       SKIP  (unavailable: {reason})\n")
        elif r["finished"]:
            tactics_str = " -> ".join(r["tactics_used"])
            fb = f" (used {r['fallbacks_used']} fallback(s))" if r["fallbacks_used"] else ""
            print(f"       PROVED in {r['num_steps']} step(s){fb}")
            print(f"       Proof: {tactics_str}\n")
        elif r["has_error"]:
            print(f"       FAILED at step {r['num_steps']} ({r['error_message']})\n")
        else:
            print(f"       EXHAUSTED after {r['num_steps']} steps (no proof found)\n")

    # ---- Aggregate metrics ------------------------------------------------
    available = [r for r in results if r["available"]]
    proved = [r for r in available if r["finished"]]
    errored = [r for r in available if r["has_error"]]
    exhausted = [r for r in available if not r["finished"] and not r["has_error"]]
    skipped = [r for r in results if not r["available"]]

    n = len(results)
    n_avail = len(available)
    n_proved = len(proved)

    print(f"\n{'='*64}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*64}")

    # Per-theorem table
    print(f"\n  {'Theorem':<40s} {'Status':<12s} {'Steps':<6s} {'Tactic'}")
    print(f"  {'─'*40} {'─'*12} {'─'*6} {'─'*30}")
    for r in results:
        name = r["full_name"][:39]
        if not r["available"]:
            status = "SKIP"
            steps = "—"
            tac = ""
        elif r["finished"]:
            status = "PROVED"
            steps = str(r["num_steps"])
            tac = r["winning_tactic"][:30] if r["winning_tactic"] else ""
        elif r["has_error"]:
            status = "ERROR"
            steps = str(r["num_steps"])
            tac = ""
        else:
            status = "EXHAUSTED"
            steps = str(r["num_steps"])
            tac = ""
        print(f"  {name:<40s} {status:<12s} {steps:<6s} {tac}")

    print(f"\n  {'─'*64}")
    print(f"  Total theorems:    {n}")
    print(f"  Available:         {n_avail}/{n}")
    if n_avail:
        print(f"  Proved:            {n_proved}/{n_avail}  ({n_proved/n_avail:.0%})")
        print(f"  Errored:           {len(errored)}/{n_avail}")
        print(f"  Exhausted:         {len(exhausted)}/{n_avail}")
    print(f"  Skipped:           {len(skipped)}/{n}")
    if proved:
        avg_steps = sum(r["num_steps"] for r in proved) / len(proved)
        total_fb = sum(r["fallbacks_used"] for r in proved)
        print(f"  Avg steps (proved): {avg_steps:.1f}")
        print(f"  Fallbacks used:     {total_fb}")
    print(f"{'='*64}\n")

    metrics = {
        "run_id": run_id,
        "method": "policy_rollout_topk",
        "theorem_set": args.theorem_set,
        "ckpt_dir": args.ckpt_dir,
        "max_steps": args.max_steps,
        "top_k": args.top_k,
        "total_theorems": n,
        "available": n_avail,
        "proved": n_proved,
        "errored": len(errored),
        "exhausted": len(exhausted),
        "skipped": len(skipped),
        "success_rate": (n_proved / n_avail) if n_avail else 0.0,
        "per_theorem": results,
    }
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"Run artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
