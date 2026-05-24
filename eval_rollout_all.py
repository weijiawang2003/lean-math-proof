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
import hashlib
import json
import uuid
from pathlib import Path

from lean_dojo import Dojo

from env import make_repo, make_theorem, run_transition


def _agg_by_key(rows: list[dict], key: str) -> dict[str, int]:
    """NS4.1 helper: count how many rows carry each value at row[key].

    Used to aggregate `winning_tactic_skeleton_*` fields over the
    proved-theorem set. None values are dropped. Integer specificities
    are stringified so the resulting dict is JSON-friendly.
    """
    out: dict[str, int] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _state_hash(state_pp: str | None) -> str | None:
    """Short, deterministic, cross-run-stable hash of a normalized Lean
    pretty-printed state. None when state_pp is empty/None.

    Normalization: strip leading/trailing whitespace per line and overall,
    so insignificant formatting differences don't break equality.
    """
    if not state_pp:
        return None
    normalized = "\n".join(line.rstrip() for line in state_pp.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems, list_theorem_sets
from trace_io import append_jsonl


def _load_policy(
    policy_type: str,
    ckpt_dir: str,
    action_space: str = "search_v4",
    decode_mode: str = "beam",
    temperature: float = 0.8,
    seed: int | None = None,
    strategy_config: str | None = None,
):
    """Load the appropriate policy based on type string.

    decode_mode/temperature/seed are forwarded to generative-family policies.
    Classifier and other non-generative policies ignore them.

    strategy_config is consumed only by 'hybrid_evolved'; it points at a JSON
    file with `fallback_tactics` and `tactic_templates` arrays.
    """
    if policy_type == "classifier":
        from policy import Policy
        return Policy(ckpt_dir=ckpt_dir)
    elif policy_type == "generative":
        from generative_policy import GenerativePolicy
        return GenerativePolicy(
            ckpt_dir=ckpt_dir,
            decode_mode=decode_mode,
            temperature=temperature,
            seed=seed,
        )
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
        return PremiseAugmentedPolicy(
            ckpt_dir=ckpt_dir,
            decode_mode=decode_mode,
            temperature=temperature,
            seed=seed,
        )
    elif policy_type == "translation_guided":
        from translation_graph import TranslationGraph, TranslationGuidedPolicy
        graph = TranslationGraph()
        graph.learn_from_proofs("project/project_state.json")
        # Wrap a generative base policy so the graph guides first, model fills gaps
        from generative_policy import GenerativePolicy
        base = GenerativePolicy(ckpt_dir=ckpt_dir)
        return TranslationGuidedPolicy(graph=graph, base_policy=base)
    elif policy_type == "hybrid_evolved":
        # v3 / v3.2 / v3.4 strategy-wrapper: GenerativePolicy + candidate-
        # provided fallback_tactics / tactic_templates + per-state extras
        # budget + theorem-name-aware family tactics, deduped, deterministic.
        from generative_policy import GenerativePolicy
        from evolve.strategy_wrapper import (
            StrategyWrapperPolicy, load_strategy_config,
        )
        base = GenerativePolicy(
            ckpt_dir=ckpt_dir,
            decode_mode=decode_mode,
            temperature=temperature,
            seed=seed,
        )
        if strategy_config:
            (fb, tmpl, cap, fam, fam_budgets, deny,
             retrieval_enabled, retrieval_top_k, retrieval_forms,
             retrieval_filter_self, retrieval_filter_unavailable,
             retrieval_skip_bloating_apply, retrieval_shape_filter,
             term_builder_templates, term_builder_budget,
             priority_templates, priority_template_budget,
             use_skeleton_bag,
             retrieval_requires_family, retrieval_family_gates) = (
                load_strategy_config(strategy_config)
            )
        else:
            (fb, tmpl, cap, fam, fam_budgets, deny,
             retrieval_enabled, retrieval_top_k, retrieval_forms,
             retrieval_filter_self, retrieval_filter_unavailable,
             retrieval_skip_bloating_apply, retrieval_shape_filter,
             term_builder_templates, term_builder_budget,
             priority_templates, priority_template_budget,
             use_skeleton_bag,
             retrieval_requires_family, retrieval_family_gates) = (
                [], [], None, {}, {}, {}, False, 0, [], True, True, True, True,
                {}, 0, {}, 0, False,
                True, [],
            )
        wrapper = StrategyWrapperPolicy(
            base_policy=base, fallback_tactics=fb, tactic_templates=tmpl,
            max_extra_tactics_per_state=cap,
            theorem_family_tactics=fam,
            family_budgets=fam_budgets,
            theorem_tactic_denylist=deny,
            retrieval_enabled=retrieval_enabled,
            retrieval_top_k=retrieval_top_k,
            retrieval_tactic_forms=retrieval_forms,
            retrieval_filter_self=retrieval_filter_self,
            retrieval_filter_unavailable=retrieval_filter_unavailable,
            retrieval_shape_filter=retrieval_shape_filter,
            retrieval_requires_family=retrieval_requires_family,
            retrieval_family_gates=retrieval_family_gates,
            term_builder_templates=term_builder_templates,
            term_builder_budget=term_builder_budget,
            priority_templates=priority_templates,
            priority_template_budget=priority_template_budget,
            use_skeleton_bag=use_skeleton_bag,
        )
        # v4.3 bloat-filter flag is consumed by rollout_one_theorem, not
        # the wrapper itself. Stash it on the wrapper so the eval loop
        # can read it via a single getattr without restructuring the
        # _load_policy / rollout_one_theorem signatures.
        wrapper.retrieval_skip_bloating_apply = bool(
            retrieval_skip_bloating_apply
        )
        return wrapper
    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Use 'classifier', 'generative', 'hybrid', 'strategic', "
            f"'premise_augmented', 'translation_guided', or 'hybrid_evolved'."
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
    enable_loop_avoidance: bool = False,
) -> dict:
    """Run rollout on a single theorem with top-k fallback, return result dict.

    When enable_loop_avoidance is True (v3.3): the rollout tracks
    per-theorem seen state-pp hashes and (state_hash, tactic) pairs that
    have already errored. Tactics that produce already-seen states are
    deferred to last-resort; tactics known to error on the current state
    are skipped. The first non-erroring transition to an *unseen* state
    wins. If all candidates either error or produce seen states, the
    first seen-state advance (if any) is taken to preserve v3.2 behavior.
    """
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
        "tactics_used_origins": [],
        "error_message": None,
        "skip_reason": None,
        "winning_tactic": None,
        "winning_tactic_origin": None,
        "winning_tactic_template_source": None,
        "winning_tactic_family_source": None,
        "winning_tactic_retrieved_premise": None,
        "fallbacks_used": 0,
        # v3.3 per-theorem counters
        "loop_transition_count": 0,
        "skipped_repeated_tactic_count": 0,
        "unseen_progress_count": 0,
        # v3.4: family keys whose tactics activated for this theorem.
        # Computed once at the first rank_tactics call (it depends only on
        # full_name, not on state). Stays empty when no family matched.
        "activated_families": [],
        # v3.6: count of tactics filtered out by the per-theorem deny-list
        # across all rank_tactics calls in this rollout.
        "denied_tactic_count": 0,
        # v4.1 per-theorem retrieval counters. retrieval_activated is True
        # if at least one rank_tactics call on this theorem returned a non-
        # empty retrieved_lemma_set; attempt counts the retrieved_premise
        # tactics actually run; advanced counts those that produced a
        # non-error transition (closing or advancing state).
        "retrieval_activated": False,
        "retrieved_premise_attempt_count": 0,
        "retrieved_premise_advanced_count": 0,
        # v4.2 retrieval form / filter tracking. Forms are the short
        # form-family labels ("rw" / "simp" / "apply" / "exact") attached
        # to each retrieved-tactic candidate; attempt_by_form counts how
        # many of those forms were actually run (Lean roundtripped) on
        # this theorem; advanced_by_form counts those that produced a
        # non-error transition. Filter counts sum across rank_tactics
        # calls — how many catalog entries the wrapper dropped via the
        # self-filter and the static unavailable-lemma denylist.
        "retrieved_premise_attempt_by_form": {},
        "retrieved_premise_advanced_by_form": {},
        "retrieved_premise_filtered_self_count": 0,
        "retrieved_premise_filtered_unavailable_count": 0,
        "winning_tactic_retrieved_form": None,
        # NS4.1 skeleton attribution. Populated only when the winning
        # tactic was emitted by a Skeleton (via use_skeleton_bag=True);
        # None on legacy-path proofs and on generative_topk wins.
        "winning_tactic_skeleton_name": None,
        "winning_tactic_skeleton_shape": None,
        "winning_tactic_skeleton_family": None,
        "winning_tactic_skeleton_specificity": None,
        "winning_tactic_skeleton_priority": None,
        # NS4.1 per-theorem skeleton-level counters. attempt counts every
        # skeleton-sourced candidate run on this theorem; advanced counts
        # those that produced a non-error transition; proved is 1 if the
        # winning tactic was skeleton-sourced. skeletons_seen accumulates
        # the set of skeleton names encountered (closed by step).
        "skeleton_attempt_count": 0,
        "skeleton_advanced_count": 0,
        "skeletons_seen": [],
        # v4.3 goal-shape filter counters. retrieved_apply_goal_*_count
        # bucket every retrieved-apply transition by whether it grew /
        # shrunk / held the open-goal count. skipped_bloating_apply
        # counts the pre-filter hits (apply LEMMA suppressed because
        # the same lemma's apply already bloated earlier on this
        # theorem). bloating_apply_lemmas lists the lemma names that
        # bloated at least once.
        "retrieved_apply_goal_increase_count": 0,
        "retrieved_apply_goal_decrease_count": 0,
        "retrieved_apply_no_goal_change_count": 0,
        "skipped_bloating_apply_count": 0,
        "bloating_apply_lemmas": [],
        # v4.4 shape-filter counters. retrieved_shape_counts buckets each
        # retrieved-tactic *attempt* by its lemma shape; *_success_counts
        # buckets the non-error transitions; goal_shape captures the
        # current target's shape at step 1 (the iff/lt/le/dvd label that
        # drove the shape-aware form selection). shape_mismatch_filtered_
        # count sums the (configured forms − allowed forms) drop across
        # rank_tactics calls — i.e. how many candidate forms shape-aware
        # emission suppressed.
        "goal_shape": "unknown",
        "retrieved_shape_counts": {},
        "retrieved_shape_success_counts": {},
        "shape_mismatch_filtered_count": 0,
        # v5 term-mode (term_builder origin) tracking. attempt counts
        # term_builder candidates actually run (Lean roundtripped) on
        # this theorem; advanced counts those that produced a non-error
        # transition (closing or advancing state); proved is 1 if the
        # winning tactic is a term_builder entry. shape_keys lists the
        # term_builder_templates shape keys that fired at any step.
        "term_builder_attempt_count": 0,
        "term_builder_advanced_count": 0,
        "term_builder_proved_count": 0,
        "term_builder_shape_keys": [],
    }

    # v4.3: per-theorem set of (lemma) names whose `apply LEMMA` already
    # produced a strictly-bloating transition on this theorem. Once a
    # lemma appears here, subsequent `apply LEMMA` retrieved tactics are
    # pre-filtered for this theorem only. The lemma is NOT globally
    # banned — `rw [LEMMA]`/`simp [LEMMA]` for the same lemma still flow.
    bloating_apply_lemmas: set[str] = set()
    skip_bloating_apply = bool(getattr(pol, "retrieval_skip_bloating_apply", True))

    # Per-theorem state tracking (only consulted when enable_loop_avoidance).
    seen_state_hashes: set[str] = set()
    errored_pairs: dict[str, set[str]] = {}  # state_hash -> {tactics that errored}

    try:
        with Dojo(theorem) as (dojo, state):
            result["available"] = True
            initial_h = _state_hash(state.pp)
            if initial_h is not None:
                seen_state_hashes.add(initial_h)

            for step in range(1, max_steps + 1):
                state_h_before = _state_hash(state.pp)
                ranked = pol.rank_tactics(state.pp, theorem.full_name, k=top_k)
                origins = getattr(pol, "last_origins", None) or [None] * len(ranked)
                template_sources = (
                    getattr(pol, "last_template_sources", None)
                    or [None] * len(ranked)
                )
                family_sources = (
                    getattr(pol, "last_family_sources", None)
                    or [None] * len(ranked)
                )
                retrieved_premises = (
                    getattr(pol, "last_retrieved_premises", None)
                    or [None] * len(ranked)
                )
                retrieved_forms = (
                    getattr(pol, "last_retrieved_forms", None)
                    or [None] * len(ranked)
                )
                retrieved_shapes = (
                    getattr(pol, "last_retrieved_shapes", None)
                    or [None] * len(ranked)
                )
                # NS4.1 skeleton attribution: parallel lists exposed by
                # the wrapper when use_skeleton_bag is True. Entries that
                # did not come from a Skeleton (generative_topk, legacy-
                # routed, retrieved_premise) carry None.
                skeleton_names = (
                    getattr(pol, "last_skeleton_names", None)
                    or [None] * len(ranked)
                )
                skeleton_shapes = (
                    getattr(pol, "last_skeleton_shapes", None)
                    or [None] * len(ranked)
                )
                skeleton_families = (
                    getattr(pol, "last_skeleton_families", None)
                    or [None] * len(ranked)
                )
                skeleton_specificities = (
                    getattr(pol, "last_skeleton_specificities", None)
                    or [None] * len(ranked)
                )
                skeleton_stable_ids = (
                    getattr(pol, "last_skeleton_stable_ids", None)
                    or [None] * len(ranked)
                )
                skeleton_priorities = (
                    getattr(pol, "last_skeleton_priorities", None)
                    or [None] * len(ranked)
                )
                # Capture activated families once (depends only on theorem name).
                if step == 1:
                    fams = getattr(pol, "last_activated_families", None) or []
                    result["activated_families"] = list(fams)
                    # v4.4 capture the inferred goal shape at step 1.
                    result["goal_shape"] = getattr(
                        pol, "last_goal_shape", "unknown"
                    ) or "unknown"
                # v3.6: accumulate per-call deny-list filter count.
                result["denied_tactic_count"] += int(
                    getattr(pol, "last_denied_count", 0) or 0
                )
                # v4.1: mark retrieval activated if this step's wrapper call
                # produced a non-empty retrieved lemma set.
                if getattr(pol, "last_retrieved_lemma_set", None):
                    result["retrieval_activated"] = True
                # v4.2: accumulate per-call self / unavailable filter counts.
                result["retrieved_premise_filtered_self_count"] += int(
                    getattr(pol, "last_retrieval_filtered_self_count", 0) or 0
                )
                result["retrieved_premise_filtered_unavailable_count"] += int(
                    getattr(pol, "last_retrieval_filtered_unavailable_count", 0) or 0
                )
                # v4.4: accumulate per-call shape-mismatch filter count.
                result["shape_mismatch_filtered_count"] += int(
                    getattr(pol, "last_shape_mismatch_filtered_count", 0) or 0
                )
                # Goal shape on this step (used to tag each retrieved trace).
                step_goal_shape = getattr(pol, "last_goal_shape", "unknown") or "unknown"

                step_succeeded = False
                deferred = None  # (rank, tac, origin, tmpl_src, outcome, state_h_after)
                outcome = None
                already_errored = errored_pairs.get(state_h_before, set())

                for rank, tac in enumerate(ranked):
                    origin = origins[rank] if rank < len(origins) else None
                    tmpl_src = (
                        template_sources[rank]
                        if rank < len(template_sources) else None
                    )
                    fam_src = (
                        family_sources[rank]
                        if rank < len(family_sources) else None
                    )
                    retr_premise = (
                        retrieved_premises[rank]
                        if rank < len(retrieved_premises) else None
                    )
                    skel_name = (
                        skeleton_names[rank]
                        if rank < len(skeleton_names) else None
                    )
                    skel_shape = (
                        skeleton_shapes[rank]
                        if rank < len(skeleton_shapes) else None
                    )
                    skel_family = (
                        skeleton_families[rank]
                        if rank < len(skeleton_families) else None
                    )
                    skel_specificity = (
                        skeleton_specificities[rank]
                        if rank < len(skeleton_specificities) else None
                    )
                    skel_priority = (
                        skeleton_priorities[rank]
                        if rank < len(skeleton_priorities) else None
                    )
                    skel_stable_id = (
                        skeleton_stable_ids[rank]
                        if rank < len(skeleton_stable_ids) else None
                    )
                    retr_form = (
                        retrieved_forms[rank]
                        if rank < len(retrieved_forms) else None
                    )
                    retr_shape = (
                        retrieved_shapes[rank]
                        if rank < len(retrieved_shapes) else None
                    )

                    # v4.3 pre-filter: suppress retrieved `apply LEMMA` if
                    # the same lemma's apply form already bloated the goal
                    # stack earlier on this theorem. The retriever can still
                    # emit `rw [LEMMA]` / `simp [LEMMA]` for the same lemma
                    # — only the apply form is suppressed.
                    if (skip_bloating_apply
                            and origin == "retrieved_premise"
                            and retr_form == "apply"
                            and retr_premise in bloating_apply_lemmas):
                        result["skipped_bloating_apply_count"] += 1
                        synthetic = {
                            "file_path": str(theorem.file_path),
                            "full_name": theorem.full_name,
                            "state_pp": state.pp,
                            "tactic": tac,
                            "result_kind": "SkippedBloatingApply",
                            "proof_finished": False,
                            "step": step,
                            "domain": domain,
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "method": "policy_rollout_topk",
                            "tactic_origin": origin,
                            "tactic_retrieved_premise": retr_premise,
                            "tactic_retrieved_form": retr_form,
                            "tactic_retrieved_shape": retr_shape,
                            "goal_shape": step_goal_shape,
                            "shape_match": (
                                retr_shape is not None
                                and retr_shape == step_goal_shape
                                and retr_shape != "unknown"
                            ),
                            "state_hash_before": state_h_before,
                            "skipped_bloating_apply": True,
                        }
                        append_jsonl(traces_path, synthetic)
                        continue

                    # Pre-filter: skip tactics already known to error on this state.
                    if enable_loop_avoidance and tac in already_errored:
                        result["skipped_repeated_tactic_count"] += 1
                        synthetic = {
                            "file_path": str(theorem.file_path),
                            "full_name": theorem.full_name,
                            "state_pp": state.pp,
                            "tactic": tac,
                            "result_kind": "SkippedKnownError",
                            "proof_finished": False,
                            "step": step,
                            "domain": domain,
                            "run_id": run_id,
                            "episode_id": episode_id,
                            "method": "policy_rollout_topk",
                            "tactic_origin": origin,
                            "skipped_due_to_seen_state": True,
                            "state_hash_before": state_h_before,
                        }
                        if tmpl_src is not None:
                            synthetic["tactic_template_source"] = tmpl_src
                        if fam_src is not None:
                            synthetic["tactic_family_source"] = fam_src
                        if retr_premise is not None:
                            synthetic["tactic_retrieved_premise"] = retr_premise
                        if retr_form is not None:
                            synthetic["tactic_retrieved_form"] = retr_form
                        append_jsonl(traces_path, synthetic)
                        continue

                    # v4.1/v4.2: count attempt before run_transition so REPL
                    # crashes on a retrieved tactic still register as attempted.
                    if origin == "retrieved_premise":
                        result["retrieved_premise_attempt_count"] += 1
                        if retr_form is not None:
                            d = result["retrieved_premise_attempt_by_form"]
                            d[retr_form] = d.get(retr_form, 0) + 1
                        if retr_shape is not None:
                            ds = result["retrieved_shape_counts"]
                            ds[retr_shape] = ds.get(retr_shape, 0) + 1
                    # v5 term_builder: count attempts before run_transition
                    # so REPL crashes still register.
                    if origin == "term_builder":
                        result["term_builder_attempt_count"] += 1
                        # fam_src holds the shape key (e.g. "iff" / "any")
                        # for term_builder entries — surface it once.
                        if fam_src is not None and fam_src not in result["term_builder_shape_keys"]:
                            result["term_builder_shape_keys"].append(fam_src)
                    # NS4.1 skeleton-level attempt counter. Tracks every
                    # skeleton-sourced candidate that reaches run_transition.
                    if skel_name is not None:
                        result["skeleton_attempt_count"] += 1
                        if skel_name not in result["skeletons_seen"]:
                            result["skeletons_seen"].append(skel_name)

                    outcome = run_transition(
                        dojo, theorem, state, tac,
                        step=step,
                        domain=domain,
                        run_id=run_id,
                        episode_id=episode_id,
                        method="policy_rollout_topk",
                    )

                    # Build the trace record (frozen dataclass → dict)
                    record_dict = outcome.record.to_dict()
                    if origin is not None:
                        record_dict["tactic_origin"] = origin
                    if tmpl_src is not None:
                        record_dict["tactic_template_source"] = tmpl_src
                    if fam_src is not None:
                        record_dict["tactic_family_source"] = fam_src
                    if retr_premise is not None:
                        record_dict["tactic_retrieved_premise"] = retr_premise
                    if retr_form is not None:
                        record_dict["tactic_retrieved_form"] = retr_form
                    if retr_shape is not None:
                        record_dict["tactic_retrieved_shape"] = retr_shape
                        record_dict["goal_shape"] = step_goal_shape
                        record_dict["shape_match"] = (
                            retr_shape == step_goal_shape
                            and retr_shape != "unknown"
                        )
                    if skel_name is not None:
                        record_dict["skeleton_name"] = skel_name
                        record_dict["skeleton_shape"] = skel_shape
                        record_dict["skeleton_family"] = skel_family
                        record_dict["skeleton_specificity"] = skel_specificity
                        record_dict["skeleton_priority"] = skel_priority
                        if skel_stable_id is not None:
                            record_dict["skeleton_stable_id"] = skel_stable_id
                    record_dict["state_hash_before"] = state_h_before

                    # REPL crashed — Dojo is dead, abort theorem
                    if outcome.session_dead:
                        append_jsonl(traces_path, record_dict)
                        result["has_error"] = True
                        result["num_steps"] = step
                        result["error_message"] = f"REPL crashed on `{tac}` at step {step}"
                        break

                    if outcome.is_finished:
                        record_dict["state_hash_after"] = None
                        record_dict["produced_seen_state"] = False
                        record_dict["loop_detected"] = False
                        append_jsonl(traces_path, record_dict)
                        result["finished"] = True
                        result["winning_tactic"] = tac
                        result["winning_tactic_origin"] = origin
                        result["winning_tactic_template_source"] = tmpl_src
                        result["winning_tactic_family_source"] = fam_src
                        result["winning_tactic_retrieved_premise"] = retr_premise
                        result["winning_tactic_retrieved_form"] = retr_form
                        result["winning_tactic_skeleton_name"] = skel_name
                        result["winning_tactic_skeleton_shape"] = skel_shape
                        result["winning_tactic_skeleton_family"] = skel_family
                        result["winning_tactic_skeleton_specificity"] = skel_specificity
                        result["winning_tactic_skeleton_priority"] = skel_priority
                        result["num_steps"] = step
                        result["tactics_used"].append(tac)
                        result["tactics_used_origins"].append(origin)
                        if origin == "retrieved_premise":
                            # Closing the goal counts as an advance too.
                            result["retrieved_premise_advanced_count"] += 1
                            if retr_form is not None:
                                d = result["retrieved_premise_advanced_by_form"]
                                d[retr_form] = d.get(retr_form, 0) + 1
                            if retr_shape is not None:
                                ds = result["retrieved_shape_success_counts"]
                                ds[retr_shape] = ds.get(retr_shape, 0) + 1
                        if origin == "term_builder":
                            result["term_builder_advanced_count"] += 1
                            result["term_builder_proved_count"] = 1
                        # NS4.1 skeleton-advanced counter on the close path.
                        if skel_name is not None:
                            result["skeleton_advanced_count"] += 1
                        if rank > 0:
                            result["fallbacks_used"] += 1
                        step_succeeded = True
                        break

                    if outcome.is_error:
                        append_jsonl(traces_path, record_dict)
                        # Remember this (state, tactic) errored so we can skip
                        # it if we ever re-enter this state in a later step.
                        if enable_loop_avoidance and state_h_before is not None:
                            errored_pairs.setdefault(state_h_before, set()).add(tac)
                        continue

                    # ---- non-error, non-finished transition (advance) ----
                    state_h_after = _state_hash(outcome.next_state.pp)
                    record_dict["state_hash_after"] = state_h_after
                    produced_seen = (
                        state_h_after is not None
                        and state_h_after in seen_state_hashes
                    )
                    record_dict["produced_seen_state"] = produced_seen
                    record_dict["loop_detected"] = produced_seen and enable_loop_avoidance
                    # v4.3 goal-shape annotation. num_goals_before/after
                    # come from the TransitionRecord; they're None when
                    # Lean doesn't surface a count. delta/increased fields
                    # are set whenever both are integers.
                    g_before = record_dict.get("num_goals_before")
                    g_after = record_dict.get("num_goals_after")
                    goal_increased = False
                    if isinstance(g_before, int) and isinstance(g_after, int):
                        delta = g_after - g_before
                        record_dict["goal_count_delta"] = delta
                        record_dict["goal_count_increased"] = delta > 0
                        goal_increased = delta > 0
                        if origin == "retrieved_premise" and retr_form == "apply":
                            if delta > 0:
                                result["retrieved_apply_goal_increase_count"] += 1
                            elif delta < 0:
                                result["retrieved_apply_goal_decrease_count"] += 1
                            else:
                                result["retrieved_apply_no_goal_change_count"] += 1

                    # v4.3 bloat-reject: a retrieved `apply LEMMA` that
                    # strictly increases the open-goal count is treated
                    # as if it errored — the advance is not taken, the
                    # lemma joins the per-theorem bloat set, and the
                    # rollout continues at the same state. The trace
                    # record is still written with `bloat_rejected=True`
                    # so the diagnostic is preserved.
                    if (skip_bloating_apply
                            and origin == "retrieved_premise"
                            and retr_form == "apply"
                            and goal_increased
                            and retr_premise):
                        record_dict["bloat_rejected"] = True
                        append_jsonl(traces_path, record_dict)
                        if retr_premise not in bloating_apply_lemmas:
                            bloating_apply_lemmas.add(retr_premise)
                            result["bloating_apply_lemmas"].append(retr_premise)
                        continue

                    append_jsonl(traces_path, record_dict)

                    if produced_seen:
                        result["loop_transition_count"] += 1
                        if enable_loop_avoidance:
                            # Defer the seen-state advance — try other tactics
                            # first; we'll take this as a last resort below.
                            if deferred is None:
                                deferred = (rank, tac, origin, tmpl_src, outcome,
                                            state_h_after)
                            continue
                        # Loop avoidance disabled — fall through and accept.

                    # Accept advance (unseen, or loop avoidance disabled).
                    if state_h_after is not None:
                        seen_state_hashes.add(state_h_after)
                    if not produced_seen:
                        result["unseen_progress_count"] += 1
                    if origin == "retrieved_premise":
                        result["retrieved_premise_advanced_count"] += 1
                        if retr_form is not None:
                            d = result["retrieved_premise_advanced_by_form"]
                            d[retr_form] = d.get(retr_form, 0) + 1
                        if retr_shape is not None:
                            ds = result["retrieved_shape_success_counts"]
                            ds[retr_shape] = ds.get(retr_shape, 0) + 1
                    if origin == "term_builder":
                        result["term_builder_advanced_count"] += 1
                    # NS4.1 skeleton-advanced counter on the non-finishing
                    # advance path.
                    if skel_name is not None:
                        result["skeleton_advanced_count"] += 1
                    state = outcome.next_state
                    result["num_steps"] = step
                    result["tactics_used"].append(tac)
                    result["tactics_used_origins"].append(origin)
                    if rank > 0:
                        result["fallbacks_used"] += 1
                    step_succeeded = True
                    break

                # Last-resort: no unseen advance found; take the deferred
                # seen-state advance if we recorded one.
                if (not step_succeeded and deferred is not None
                        and outcome is not None and not outcome.session_dead):
                    _, tac, origin, _tmpl_src, def_outcome, state_h_after = deferred
                    if state_h_after is not None:
                        seen_state_hashes.add(state_h_after)
                    state = def_outcome.next_state
                    result["num_steps"] = step
                    result["tactics_used"].append(tac)
                    result["tactics_used_origins"].append(origin)
                    step_succeeded = True

                if result["finished"]:
                    break
                if outcome is not None and outcome.session_dead:
                    break

                if not step_succeeded:
                    # All k tactics errored (and no deferred advance).
                    result["has_error"] = True
                    result["num_steps"] = step
                    result["error_message"] = (
                        f"All top-{len(ranked)} tactics errored at step {step}"
                    )
                    break

    except Exception as exc:
        result["skip_reason"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser(description="Rollout policy on all theorems in a set.")
    parser.add_argument("--theorem-set", default="nat_single", choices=list_theorem_sets())
    parser.add_argument("--ckpt-dir", default="clf_ckpt")
    parser.add_argument("--policy-type", default="classifier",
                        choices=["classifier", "generative", "hybrid", "strategic",
                                 "premise_augmented", "translation_guided",
                                 "hybrid_evolved"],
                        help="Policy type: 'classifier', 'generative', 'hybrid', 'strategic', "
                             "'premise_augmented', 'translation_guided', or 'hybrid_evolved' "
                             "(generative + per-candidate fallback_tactics/tactic_templates).")
    parser.add_argument("--strategy-config", default=None,
                        help="Path to a JSON file with 'fallback_tactics' and "
                             "'tactic_templates' arrays. Consumed only by "
                             "--policy-type=hybrid_evolved.")
    parser.add_argument("--enable-loop-avoidance", action="store_true",
                        help="v3.3: track per-theorem seen state hashes; "
                             "prefer tactics producing unseen states; skip "
                             "tactics known to error on the current state.")
    parser.add_argument("--action-space", default="search_v4",
                        help="Action space for hybrid policy fallback.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Try up to k tactics per step before declaring failure.")
    parser.add_argument("--domain", default="mathlib4")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--decode-mode", default="beam", choices=["beam", "sample"],
                        help="Generative decoding strategy. 'beam' is deterministic "
                             "(prior default). 'sample' uses temperature/top-p with --seed.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature when --decode-mode=sample.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Torch seed for sampling reproducibility. "
                             "Ignored when --decode-mode=beam.")
    args = parser.parse_args()

    # Set torch seed early so any model-init randomness is also pinned.
    if args.seed is not None:
        import random
        import torch
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    theorems = get_theorems(args.theorem_set)
    print(f"\n{'='*64}")
    print(f"  LEAN TACTIC POLICY EVALUATION")
    print(f"{'='*64}")
    print(f"  Theorem set : {args.theorem_set} ({len(theorems)} theorems)")
    print(f"  Checkpoint  : {args.ckpt_dir}")
    print(f"  Policy type : {args.policy_type}")
    print(f"  Strategy    : top-{args.top_k} fallback, max {args.max_steps} steps")
    print(f"{'='*64}\n")

    pol = _load_policy(
        args.policy_type, args.ckpt_dir,
        action_space=getattr(args, "action_space", "search_v4"),
        decode_mode=args.decode_mode,
        temperature=args.temperature,
        seed=args.seed,
        strategy_config=args.strategy_config,
    )
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
            "decode_mode": args.decode_mode,
            "temperature": args.temperature,
            "seed": args.seed,
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
            enable_loop_avoidance=args.enable_loop_avoidance,
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

    # Per-origin proved counts (only meaningful for hybrid_evolved).
    proved_by_origin: dict[str, int] = {}
    for r in proved:
        origin = r.get("winning_tactic_origin")
        if origin:
            proved_by_origin[origin] = proved_by_origin.get(origin, 0) + 1
    if proved_by_origin:
        print(f"  Proved by origin:   {proved_by_origin}")

    # v3.3 anti-loop aggregates.
    loop_transition_count = sum(int(r.get("loop_transition_count") or 0) for r in results)
    skipped_repeated_tactic_count = sum(
        int(r.get("skipped_repeated_tactic_count") or 0) for r in results
    )
    unseen_progress_count = sum(
        int(r.get("unseen_progress_count") or 0) for r in results
    )
    if args.enable_loop_avoidance or loop_transition_count:
        print(f"  Loop transitions:   {loop_transition_count}")
        print(f"  Skipped repeats:    {skipped_repeated_tactic_count}")
        print(f"  Unseen progress:    {unseen_progress_count}")

    # v3.4 family-activation aggregates.
    family_activation_counts: dict[str, int] = {}
    family_proved_counts: dict[str, int] = {}
    family_activated_theorems: dict[str, list[str]] = {}
    for r in results:
        for fam in r.get("activated_families") or []:
            family_activation_counts[fam] = family_activation_counts.get(fam, 0) + 1
            family_activated_theorems.setdefault(fam, []).append(r["full_name"])
    for r in proved:
        fsrc = r.get("winning_tactic_family_source")
        if fsrc:
            family_proved_counts[fsrc] = family_proved_counts.get(fsrc, 0) + 1
    if family_activation_counts:
        print(f"  Family activations: {family_activation_counts}")
    if family_proved_counts:
        print(f"  Family proofs:      {family_proved_counts}")

    # v3.6 deny-list aggregate.
    denied_tactic_total = sum(
        int(r.get("denied_tactic_count") or 0) for r in results
    )
    if denied_tactic_total:
        print(f"  Denied tactics:     {denied_tactic_total} (per-theorem deny-list)")

    # v4.1 retrieval aggregates.
    retrieved_premise_activation_count = sum(
        1 for r in results if r.get("retrieval_activated")
    )
    retrieved_premise_attempt_count = sum(
        int(r.get("retrieved_premise_attempt_count") or 0) for r in results
    )
    retrieved_premise_advanced_count = sum(
        int(r.get("retrieved_premise_advanced_count") or 0) for r in results
    )
    retrieved_premise_proved_count = sum(
        1 for r in proved if r.get("winning_tactic_origin") == "retrieved_premise"
    )
    retrieved_premise_wins: list[dict] = [
        {
            "theorem": r["full_name"],
            "premise": r.get("winning_tactic_retrieved_premise"),
            "tactic": r.get("winning_tactic"),
            "form": r.get("winning_tactic_retrieved_form"),
        }
        for r in proved
        if r.get("winning_tactic_origin") == "retrieved_premise"
    ]
    # v4.2 form-level aggregates.
    retrieved_premise_form_counts: dict[str, int] = {}
    retrieved_premise_form_success_counts: dict[str, int] = {}
    for r in results:
        for form, c in (r.get("retrieved_premise_attempt_by_form") or {}).items():
            retrieved_premise_form_counts[form] = (
                retrieved_premise_form_counts.get(form, 0) + int(c or 0)
            )
        for form, c in (r.get("retrieved_premise_advanced_by_form") or {}).items():
            retrieved_premise_form_success_counts[form] = (
                retrieved_premise_form_success_counts.get(form, 0) + int(c or 0)
            )
    retrieved_premise_filtered_self_count = sum(
        int(r.get("retrieved_premise_filtered_self_count") or 0) for r in results
    )
    retrieved_premise_filtered_unavailable_count = sum(
        int(r.get("retrieved_premise_filtered_unavailable_count") or 0)
        for r in results
    )
    # v4.3 bloat-filter aggregates.
    retrieved_apply_goal_increase_count = sum(
        int(r.get("retrieved_apply_goal_increase_count") or 0) for r in results
    )
    retrieved_apply_goal_decrease_count = sum(
        int(r.get("retrieved_apply_goal_decrease_count") or 0) for r in results
    )
    retrieved_apply_no_goal_change_count = sum(
        int(r.get("retrieved_apply_no_goal_change_count") or 0) for r in results
    )
    skipped_bloating_apply_count = sum(
        int(r.get("skipped_bloating_apply_count") or 0) for r in results
    )
    bloating_apply_lemma_counts: dict[str, int] = {}
    for r in results:
        for lem in (r.get("bloating_apply_lemmas") or []):
            bloating_apply_lemma_counts[lem] = (
                bloating_apply_lemma_counts.get(lem, 0) + 1
            )
    # v4.4 shape-filter aggregates.
    retrieved_shape_counts: dict[str, int] = {}
    retrieved_shape_success_counts: dict[str, int] = {}
    shape_mismatch_filtered_count = 0
    for r in results:
        for s, c in (r.get("retrieved_shape_counts") or {}).items():
            retrieved_shape_counts[s] = retrieved_shape_counts.get(s, 0) + int(c or 0)
        for s, c in (r.get("retrieved_shape_success_counts") or {}).items():
            retrieved_shape_success_counts[s] = (
                retrieved_shape_success_counts.get(s, 0) + int(c or 0)
            )
        shape_mismatch_filtered_count += int(
            r.get("shape_mismatch_filtered_count") or 0
        )
    if retrieved_premise_activation_count:
        print(
            f"  Retrieval:          activated on {retrieved_premise_activation_count} theorems, "
            f"{retrieved_premise_attempt_count} tactics attempted, "
            f"{retrieved_premise_advanced_count} advanced, "
            f"{retrieved_premise_proved_count} won"
        )
        print(
            f"  Retrieval filters:  filtered_self={retrieved_premise_filtered_self_count}, "
            f"filtered_unavailable={retrieved_premise_filtered_unavailable_count}"
        )
        if retrieved_premise_form_counts:
            print(f"  Retrieval forms:    {retrieved_premise_form_counts}")
        if retrieved_premise_form_success_counts:
            print(f"  Retrieval form ok:  {retrieved_premise_form_success_counts}")
        if (retrieved_apply_goal_increase_count
                or retrieved_apply_goal_decrease_count
                or retrieved_apply_no_goal_change_count
                or skipped_bloating_apply_count):
            print(
                f"  Apply goal-shape:   inc={retrieved_apply_goal_increase_count}, "
                f"dec={retrieved_apply_goal_decrease_count}, "
                f"same={retrieved_apply_no_goal_change_count}, "
                f"skipped_bloating={skipped_bloating_apply_count}"
            )
        if bloating_apply_lemma_counts:
            print(f"  Bloating lemmas:    {bloating_apply_lemma_counts}")
        if retrieved_shape_counts or shape_mismatch_filtered_count:
            print(
                f"  Shape attempts:     {retrieved_shape_counts}  "
                f"success={retrieved_shape_success_counts}  "
                f"mismatch_filtered={shape_mismatch_filtered_count}"
            )
    tb_attempt = sum(int(r.get("term_builder_attempt_count") or 0) for r in results)
    tb_adv = sum(int(r.get("term_builder_advanced_count") or 0) for r in results)
    tb_proved = sum(int(r.get("term_builder_proved_count") or 0) for r in results)
    if tb_attempt:
        print(
            f"  Term builder:       {tb_attempt} attempts, "
            f"{tb_adv} advanced, {tb_proved} won"
        )
    print(f"{'='*64}\n")

    metrics = {
        "run_id": run_id,
        "method": "policy_rollout_topk",
        "theorem_set": args.theorem_set,
        "ckpt_dir": args.ckpt_dir,
        "policy_type": args.policy_type,
        "max_steps": args.max_steps,
        "top_k": args.top_k,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature,
        "seed": args.seed,
        "strategy_config": args.strategy_config,
        "total_theorems": n,
        "available": n_avail,
        "proved": n_proved,
        "errored": len(errored),
        "exhausted": len(exhausted),
        "skipped": len(skipped),
        "success_rate": (n_proved / n_avail) if n_avail else 0.0,
        "proved_by_origin": proved_by_origin,
        # v3.3 anti-loop aggregates
        "enable_loop_avoidance": args.enable_loop_avoidance,
        "loop_transition_count": loop_transition_count,
        "skipped_repeated_tactic_count": skipped_repeated_tactic_count,
        "unseen_progress_count": unseen_progress_count,
        # v3.4 family-activation aggregates
        "family_activation_counts": family_activation_counts,
        "family_proved_counts": family_proved_counts,
        "family_activated_theorems": family_activated_theorems,
        # v3.6 per-theorem deny-list aggregate
        "denied_tactic_total": denied_tactic_total,
        # v4.1 retrieval aggregates
        "retrieved_premise_activation_count": retrieved_premise_activation_count,
        "retrieved_premise_attempt_count": retrieved_premise_attempt_count,
        "retrieved_premise_advanced_count": retrieved_premise_advanced_count,
        "retrieved_premise_proved_count": retrieved_premise_proved_count,
        "retrieved_premise_wins": retrieved_premise_wins,
        # v4.2 retrieval aggregates
        "retrieved_premise_form_counts": retrieved_premise_form_counts,
        "retrieved_premise_form_success_counts": retrieved_premise_form_success_counts,
        "retrieved_premise_filtered_self_count": retrieved_premise_filtered_self_count,
        "retrieved_premise_filtered_unavailable_count": retrieved_premise_filtered_unavailable_count,
        # v4.3 bloat-filter aggregates
        "retrieved_apply_goal_increase_count": retrieved_apply_goal_increase_count,
        "retrieved_apply_goal_decrease_count": retrieved_apply_goal_decrease_count,
        "retrieved_apply_no_goal_change_count": retrieved_apply_no_goal_change_count,
        "skipped_bloating_apply_count": skipped_bloating_apply_count,
        "bloating_apply_lemma_counts": bloating_apply_lemma_counts,
        # v4.4 shape-filter aggregates
        "retrieved_shape_counts": retrieved_shape_counts,
        "retrieved_shape_success_counts": retrieved_shape_success_counts,
        "shape_mismatch_filtered_count": shape_mismatch_filtered_count,
        # v5 term-mode (term_builder) aggregates
        "term_builder_attempt_count": sum(
            int(r.get("term_builder_attempt_count") or 0) for r in results
        ),
        "term_builder_advanced_count": sum(
            int(r.get("term_builder_advanced_count") or 0) for r in results
        ),
        "term_builder_proved_count": sum(
            int(r.get("term_builder_proved_count") or 0) for r in results
        ),
        "term_builder_wins": [
            {
                "theorem": r["full_name"],
                "tactic": r.get("winning_tactic"),
                "shape_key": r.get("winning_tactic_family_source"),
            }
            for r in proved if r.get("winning_tactic_origin") == "term_builder"
        ],
        # NS4.1 skeleton-level aggregates. Populated only when at least
        # one theorem ran with use_skeleton_bag=True; otherwise the
        # counters are all 0 and dicts are empty. Used by the unification
        # report and by future scoring hooks.
        "skeleton_attempt_count": sum(
            int(r.get("skeleton_attempt_count") or 0) for r in results
        ),
        "skeleton_advanced_count": sum(
            int(r.get("skeleton_advanced_count") or 0) for r in results
        ),
        "skeleton_proved_count": sum(
            1 for r in proved if r.get("winning_tactic_skeleton_name")
        ),
        "skeleton_proved_counts": _agg_by_key(
            proved, "winning_tactic_skeleton_name"
        ),
        "skeleton_proved_counts_by_family": _agg_by_key(
            proved, "winning_tactic_skeleton_family"
        ),
        "skeleton_proved_counts_by_shape": _agg_by_key(
            proved, "winning_tactic_skeleton_shape"
        ),
        "skeleton_specificity_proved_counts": _agg_by_key(
            proved, "winning_tactic_skeleton_specificity"
        ),
        "skeleton_wins": [
            {
                "theorem": r["full_name"],
                "skeleton_name": r.get("winning_tactic_skeleton_name"),
                "shape": r.get("winning_tactic_skeleton_shape"),
                "family": r.get("winning_tactic_skeleton_family"),
                "specificity": r.get("winning_tactic_skeleton_specificity"),
                "tactic": r.get("winning_tactic"),
            }
            for r in proved if r.get("winning_tactic_skeleton_name")
        ],
        "per_theorem": results,
    }
    write_metrics(artifacts["metrics_path"], metrics)

    # v3.2: compact diagnostic for EXHAUSTED theorems — what tactics advanced
    # state without ever closing the goal? Written one level above the
    # eval-XXXX subdir so callers (evolve/evaluator.py) can find it at the
    # candidate root: project/evolve/runs/<run_id>/eval/<cand>/progress_summary.json
    progress: list[dict] = []
    for r in exhausted:
        tactics = r.get("tactics_used") or []
        origins = r.get("tactics_used_origins") or []
        progress.append({
            "theorem": r["full_name"],
            "num_steps": r["num_steps"],
            "advancing_tactics": [
                {"tactic": t, "origin": o, "step": i + 1}
                for i, (t, o) in enumerate(zip(tactics, origins))
            ],
            "any_template_advanced": any(o == "tactic_template" for o in origins),
        })
    parent_dir = Path(args.out_dir)
    if parent_dir.exists() and parent_dir != Path("runs"):
        # Only write when the caller specified an explicit per-candidate
        # out-dir (i.e. via evolve/evaluator.py). Skip for the default
        # "runs/" base to avoid polluting unrelated tooling.
        try:
            progress_path = parent_dir / "progress_summary.json"
            progress_path.write_text(
                json.dumps(progress, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    print(f"Run artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
