"""Incremental, resumable theorem proving pipeline.

Unlike the batch pipelines (run_curriculum.py, run_generative_curriculum.py),
this pipeline is designed for long-term use:

  - Saves progress after EVERY theorem (not just per stage)
  - Never re-searches a theorem that's already been searched
  - Accumulates traces into a single growing corpus
  - Only retrains when new data exists
  - Models are versioned (gen_v1, gen_v2, ...)
  - Can be interrupted and resumed at any point

Workflow:
  1. Register theorems (from tasks.py sets or extracted JSON)
  2. Search unsearched theorems (incremental — skips already-done)
  3. Retrain model on accumulated traces (only if new data)
  4. Evaluate with latest model (on unproved theorems)
  5. Repeat from step 2 with harder theorems

Usage:
  python run_incremental.py status                          # show project state
  python run_incremental.py register --theorem-set curriculum_tier1
  python run_incremental.py register --theorem-set curriculum_tier2
  python run_incremental.py search                          # search all unsearched
  python run_incremental.py search --difficulty easy         # search only easy ones
  python run_incremental.py train                           # retrain if new data
  python run_incremental.py train --type classifier         # train classifier instead
  python run_incremental.py eval                            # eval on unproved theorems
  python run_incremental.py eval --all                      # eval on all theorems
  python run_incremental.py auto                            # full cycle: search → train → eval
  python run_incremental.py auto --rounds 3                 # repeat 3 times
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path

from project_state import ProjectState


def cmd_status(state: ProjectState, args: argparse.Namespace) -> None:
    """Show current project state."""
    print(state.summary())

    # Detailed breakdown
    proved = state.get_proved()
    unproved = state.get_unproved()
    unsearched = state.get_unsearched()

    if proved:
        print(f"\n  Proved ({len(proved)}):")
        for name in proved[:20]:
            t = state.theorems[name]
            tac = (t.get("proof_tactics") or "")[:50]
            print(f"    {name:<40s}  {tac}")
        if len(proved) > 20:
            print(f"    ... and {len(proved) - 20} more")

    if unproved:
        print(f"\n  Searched but unproved ({len(unproved)}):")
        for name in unproved[:10]:
            print(f"    {name}")

    if unsearched:
        print(f"\n  Unsearched ({len(unsearched)}):")
        for name in unsearched[:10]:
            t = state.theorems[name]
            print(f"    {name:<40s}  [{t.get('difficulty', '?')}]")
        if len(unsearched) > 10:
            print(f"    ... and {len(unsearched) - 10} more")


def cmd_register(state: ProjectState, args: argparse.Namespace) -> None:
    """Register theorems from a theorem set or JSON file."""
    if args.theorem_set:
        from tasks import get_theorems
        configs = get_theorems(args.theorem_set)
        state.register_theorems_from_configs(configs)
        print(f"Registered {len(configs)} theorems from set '{args.theorem_set}'")

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        theorems = data.get("theorems", [])
        for t in theorems:
            state.register_theorem(
                full_name=t["full_name"],
                file_path=t["file_path"],
                difficulty=t.get("difficulty", "unknown"),
                available=t.get("available"),
            )
        print(f"Registered {len(theorems)} theorems from {args.from_json}")

    state.save()
    print(state.summary())


def cmd_search(state: ProjectState, args: argparse.Namespace) -> None:
    """Search unsearched theorems incrementally."""
    from lean_dojo import Dojo

    from actions import get_action_space
    from core_types import TheoremConfig
    from env import make_repo, make_theorem, run_transition
    from search_generate_traces import search_and_log_for_theorem
    from trace_io import append_jsonl

    unsearched = state.get_unsearched(difficulty=args.difficulty)
    if not unsearched:
        print("No unsearched theorems. Register more with 'register' command.")
        return

    limit = args.limit or len(unsearched)
    targets = unsearched[:limit]
    print(f"Searching {len(targets)} unsearched theorems (of {len(unsearched)} total)")

    actions = get_action_space(args.action_space)
    run_id = f"search-{uuid.uuid4().hex[:8]}"

    for i, name in enumerate(targets):
        t = state.theorems[name]
        cfg = TheoremConfig(file_path=t["file_path"], full_name=name)

        print(f"\n[{i+1}/{len(targets)}] Searching: {name}")

        # Write to a temp file, then append to corpus
        tmp_trace = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir=state.project_dir
        )
        tmp_path = tmp_trace.name
        tmp_trace.close()

        # Clear temp file
        open(tmp_path, "w").close()

        ok = search_and_log_for_theorem(
            cfg, tmp_path, run_id,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            actions=actions,
        )

        if ok:
            # Count traces written
            count = sum(1 for line in open(tmp_path) if line.strip())

            # Append to cumulative corpus
            state.append_traces(tmp_path)
            state.mark_searched(name, traces_added=count)
            print(f"  Added {count} traces (corpus total: {state.total_traces})")
        else:
            state.mark_unavailable(name, reason="search failed or unavailable")
            print(f"  Unavailable — marked and skipped")

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        # Save after EVERY theorem — crash-safe
        state.save()

    print(f"\nSearch complete. {state.summary()}")


def cmd_train(state: ProjectState, args: argparse.Namespace) -> None:
    """Train or retrain model on accumulated traces."""
    import subprocess

    py = sys.executable
    model_type = args.type

    if not state.needs_retrain(model_type):
        print(f"No new traces since last {model_type} model. Skipping retrain.")
        print(f"  (Use --force to retrain anyway)")
        if not args.force:
            return

    version = state.next_model_version(model_type)
    ckpt_dir = state.model_dir(version, model_type)
    print(f"Training {model_type} model v{version} on {state.total_traces} traces")

    if model_type == "gen":
        # Build seq2seq dataset
        seq2seq_data = str(Path(state.project_dir) / f"seq2seq_data_v{version}.jsonl")
        subprocess.run([
            py, "build_seq2seq_dataset.py",
            "--in", state.traces_path,
            "--out", seq2seq_data,
            "--min-goal-drop", "1",
        ], check=True)

        # Train CodeT5
        subprocess.run([
            py, "train_tactic_generator.py",
            "--data", seq2seq_data,
            "--model", args.base_model,
            "--output-dir", ckpt_dir,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--val-split", "0.1",
        ], check=True)

    elif model_type == "decoder":
        # Build seq2seq dataset (same format works for decoder)
        seq2seq_data = str(Path(state.project_dir) / f"seq2seq_data_v{version}.jsonl")
        subprocess.run([
            py, "build_seq2seq_dataset.py",
            "--in", state.traces_path,
            "--out", seq2seq_data,
            "--min-goal-drop", "1",
        ], check=True)

        # Train GPT-2 decoder
        base = args.base_model if args.base_model != "t5-small" else "gpt2"
        subprocess.run([
            py, "train_decoder_policy.py",
            "--data", seq2seq_data,
            "--model", base,
            "--output-dir", ckpt_dir,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--val-split", "0.1",
        ], check=True)

    elif model_type == "classifier":
        # Build SFT dataset
        sft_data = str(Path(state.project_dir) / f"sft_data_v{version}.jsonl")
        subprocess.run([
            py, "build_sft_dataset.py",
            "--in", state.traces_path,
            "--out", sft_data,
            "--action-space", args.action_space,
            "--min-goal-drop", "1",
            "--max-per-label", "64",
        ], check=True)

        # Train DistilBERT
        subprocess.run([
            py, "train_action_classifier.py",
            "--sft-path", sft_data,
            "--action-space", args.action_space,
            "--output-dir", ckpt_dir,
            "--seed", str(args.seed),
            "--val-split", "0.1",
        ], check=True)

    # Register model
    base_model_name = {
        "gen": args.base_model,
        "decoder": args.base_model if args.base_model != "t5-small" else "gpt2",
        "classifier": "distilbert-base-uncased",
    }.get(model_type, args.base_model)

    state.register_model(
        version=version,
        model_type=model_type,
        ckpt_dir=ckpt_dir,
        base_model=base_model_name,
        trained_on_traces=state.total_traces,
    )
    state.save()
    print(f"Model {model_type} v{version} saved to {ckpt_dir}")


def cmd_eval(state: ProjectState, args: argparse.Namespace) -> None:
    """Evaluate latest model on theorems."""
    from lean_dojo import Dojo

    from core_types import TheoremConfig
    from env import make_repo, make_theorem, run_transition
    from trace_io import append_jsonl

    latest = state.latest_model
    if latest is None:
        print("No trained model. Run 'train' first.")
        return

    model_type = latest["type"]
    ckpt_dir = latest["ckpt_dir"]
    print(f"Evaluating {model_type} v{latest['version']} from {ckpt_dir}")

    # Load policy
    use_strategic = getattr(args, "use_strategic", False)
    use_premise_aug = getattr(args, "use_premise_augmented", False)
    if use_strategic:
        from strategic_policy import StrategicPolicy
        pol = StrategicPolicy(
            base_policy="hybrid" if model_type in ("gen", "decoder") else "action_space",
            gen_ckpt_dir=ckpt_dir,
            action_space=getattr(args, "action_space", "strategic_v5"),
            traces_path=state.traces_path,
        )
        print(f"Using strategic policy (backward reasoning + premise retrieval)")
    elif use_premise_aug and model_type in ("gen", "decoder"):
        from generative_policy import PremiseAugmentedPolicy
        pol = PremiseAugmentedPolicy(
            ckpt_dir=ckpt_dir,
            premise_index_path=str(Path(state.project_dir) / "premise_index.json"),
            traces_path=state.traces_path,
        )
        print(f"Using premise-augmented generative policy")
    elif model_type in ("gen", "decoder"):
        from generative_policy import GenerativePolicy
        pol = GenerativePolicy(ckpt_dir=ckpt_dir)
    else:
        from policy import Policy
        pol = Policy(ckpt_dir=ckpt_dir)

    # Select theorems to evaluate
    if args.all:
        targets = [
            name for name, t in state.theorems.items()
            if t.get("available") is not False
        ]
    else:
        # Only evaluate unproved theorems
        targets = state.get_unproved()
        if not targets:
            # If nothing is unproved but unsearched exist, try searched ones
            targets = [
                name for name, t in state.theorems.items()
                if t.get("available") is not False and t["searched"]
            ]

    if not targets:
        print("No theorems to evaluate.")
        return

    print(f"Evaluating {len(targets)} theorems")
    repo = make_repo()
    run_id = f"eval-{uuid.uuid4().hex[:8]}"

    proved_this_round = 0
    for i, name in enumerate(targets):
        t = state.theorems[name]
        cfg = TheoremConfig(file_path=t["file_path"], full_name=name)

        print(f"\n[{i+1}/{len(targets)}] {name}")

        try:
            theorem = make_theorem(repo, cfg)
            with Dojo(theorem) as (dojo, init_state):
                step_state = init_state
                solved = False

                for step in range(1, args.max_steps + 1):
                    ranked = pol.rank_tactics(step_state.pp, name, k=args.top_k)

                    step_ok = False
                    for tac in ranked:
                        outcome = run_transition(
                            dojo, theorem, step_state, tac,
                            step=step, run_id=run_id, method="incremental_eval",
                        )

                        if outcome.session_dead:
                            print(f"  [CRASH] REPL died — aborting")
                            break

                        if outcome.is_finished:
                            state.mark_proved(name, tactic=tac, steps=step)
                            proved_this_round += 1
                            print(f"  PROVED in {step} step(s): {tac}")
                            solved = True
                            step_ok = True
                            break

                        if not outcome.is_error:
                            step_state = outcome.next_state
                            step_ok = True
                            break

                    if solved or outcome.session_dead or not step_ok:
                        break

                if not solved:
                    print(f"  NOT proved ({step} steps)")

        except Exception as exc:
            print(f"  SKIP: {str(exc)[:80]}")

        # Save after every theorem
        state.save()

    total_proved = len(state.get_proved())
    total_avail = sum(
        1 for t in state.theorems.values() if t.get("available") is not False
    )
    print(f"\nThis round: {proved_this_round} newly proved")
    print(f"Overall: {total_proved}/{total_avail} proved ({total_proved/total_avail:.0%})" if total_avail else "")


def cmd_selfplay(state: ProjectState, args: argparse.Namespace) -> None:
    """Self-play refinement: model generates tactics → verify in Lean → add new successes to corpus.

    This creates a virtuous cycle where the model discovers novel tactics
    that work, and those get folded back into the training data.
    """
    from lean_dojo import Dojo

    from core_types import TheoremConfig
    from env import make_repo, make_theorem, run_transition
    from trace_io import append_jsonl

    latest = state.latest_model
    if latest is None:
        print("No trained model. Run 'train' first.")
        return

    model_type = latest["type"]
    ckpt_dir = latest["ckpt_dir"]

    # Load generative policy (or hybrid)
    if args.use_hybrid:
        from hybrid_policy import HybridPolicy
        pol = HybridPolicy(gen_ckpt_dir=ckpt_dir, action_space=args.action_space)
        print(f"Self-play with hybrid policy (gen v{latest['version']} + {args.action_space})")
    elif model_type == "gen":
        from generative_policy import GenerativePolicy
        pol = GenerativePolicy(ckpt_dir=ckpt_dir)
        print(f"Self-play with generative model v{latest['version']}")
    else:
        from policy import Policy
        pol = Policy(ckpt_dir=ckpt_dir)
        print(f"Self-play with classifier v{latest['version']}")

    # Target: unproved theorems (where we can potentially find new proofs)
    targets = state.get_unproved()
    if not targets:
        targets = [
            name for name, t in state.theorems.items()
            if t.get("available") is not False and t["searched"]
        ]

    if not targets:
        print("No theorems to self-play on.")
        return

    limit = args.limit or len(targets)
    targets = targets[:limit]
    print(f"Self-play on {len(targets)} theorems")

    repo = make_repo()
    run_id = f"selfplay-{uuid.uuid4().hex[:8]}"
    new_traces = 0
    proved_this_round = 0

    for i, name in enumerate(targets):
        t = state.theorems[name]
        cfg = TheoremConfig(file_path=t["file_path"], full_name=name)

        print(f"\n[{i+1}/{len(targets)}] {name}")

        try:
            theorem = make_theorem(repo, cfg)
            with Dojo(theorem) as (dojo, init_state):
                step_state = init_state
                solved = False

                for step in range(1, args.max_steps + 1):
                    # Generate more diverse tactics for self-play
                    if hasattr(pol, 'generate_tactics'):
                        # Use sampling for diversity in self-play
                        ranked = pol.generate_tactics(
                            step_state.pp, name,
                            num_samples=args.top_k,
                            num_beams=0,  # sampling mode
                            temperature=args.temperature,
                        )
                    else:
                        ranked = pol.rank_tactics(step_state.pp, name, k=args.top_k)

                    step_ok = False
                    for tac in ranked:
                        outcome = run_transition(
                            dojo, theorem, step_state, tac,
                            step=step, run_id=run_id, method="selfplay",
                        )

                        if outcome.session_dead:
                            break

                        # Log ALL non-error transitions as new training data
                        if not outcome.is_error:
                            append_jsonl(state.traces_path, outcome.record)
                            new_traces += 1

                        if outcome.is_finished:
                            state.mark_proved(name, tactic=tac, steps=step)
                            proved_this_round += 1
                            print(f"  PROVED in {step} step(s): {tac}")
                            solved = True
                            step_ok = True
                            break

                        if not outcome.is_error:
                            step_state = outcome.next_state
                            step_ok = True
                            break

                    if solved or outcome.session_dead or not step_ok:
                        break

                if not solved:
                    print(f"  not proved ({step} steps)")

        except Exception as exc:
            print(f"  SKIP: {str(exc)[:80]}")

        state.save()

    # Update trace count
    state._data["total_traces"] += new_traces
    counts = state._data["traces_by_method"]
    counts["selfplay"] = counts.get("selfplay", 0) + new_traces
    state.save()

    print(f"\nSelf-play complete:")
    print(f"  New traces added: {new_traces}")
    print(f"  Proved this round: {proved_this_round}")
    print(f"  Total traces: {state.total_traces}")


def cmd_build_premise_index(state: ProjectState, args: argparse.Namespace) -> None:
    """Build premise retrieval index from accumulated traces."""
    from premise_retriever import PremiseRetriever

    out_path = args.out or str(Path(state.project_dir) / "premise_index.json")
    retriever = PremiseRetriever()
    retriever.build_index_from_traces(state.traces_path)
    retriever.save_index(out_path)
    print(f"Premise index saved to {out_path}")

    # Quick test
    test_state = "α : Type u\ns t : Set α\n⊢ s ∪ t = t ∪ s"
    premises = retriever.retrieve(test_state, k=10)
    print(f"\nTest retrieval (Set.union_comm-like state):")
    for i, p in enumerate(premises, 1):
        print(f"  {i}. {p}")


def cmd_scale(state: ProjectState, args: argparse.Namespace) -> None:
    """Scale up: extract theorems from more files → register → search → train → eval.

    One command to go from current state to 200+ theorems.
    """
    import subprocess

    py = sys.executable

    print(f"\n{'='*64}")
    print(f"  SCALING UP: extract → register → search → train → eval")
    print(f"{'='*64}")

    # Step 1: Extract theorems
    discovered_path = str(Path(state.project_dir) / "discovered_theorems.json")

    if not args.skip_extraction:
        extract_cmd = [
            py, "extract_theorems.py",
            "--preset", args.preset,
            "--out", discovered_path,
        ]
        if args.check_availability:
            extract_cmd.append("--check-availability")
        if args.max_per_file > 0:
            extract_cmd.extend(["--max-per-file", str(args.max_per_file)])

        print(f"\n--- EXTRACT PHASE ---")
        try:
            subprocess.run(extract_cmd, check=True)
        except subprocess.CalledProcessError:
            print("[WARN] Extraction failed — continuing with existing theorems")
    else:
        print("\n--- SKIP EXTRACTION (reusing existing) ---")

    # Step 2: Register extracted theorems
    if Path(discovered_path).exists():
        print(f"\n--- REGISTER PHASE ---")
        cmd_register(state, argparse.Namespace(
            theorem_set="", from_json=discovered_path
        ))

    # Step 3: Search unsearched (incremental — skips already-done)
    print(f"\n--- SEARCH PHASE ---")
    search_args = argparse.Namespace(
        difficulty=args.difficulty,
        limit=args.limit,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        action_space=args.action_space,
    )
    cmd_search(state, search_args)

    # Step 4: Train (only if new data)
    print(f"\n--- TRAIN PHASE ---")
    train_args = argparse.Namespace(
        type=args.type,
        base_model=args.base_model,
        action_space=args.action_space,
        epochs=args.epochs,
        seed=args.seed,
        force=args.force,
    )
    cmd_train(state, train_args)

    # Step 5: Eval
    print(f"\n--- EVAL PHASE ---")
    eval_args = argparse.Namespace(
        all=True,
        max_steps=args.max_steps,
        top_k=args.top_k,
    )
    cmd_eval(state, eval_args)

    print(f"\n{'='*64}")
    print(f"  SCALING COMPLETE")
    print(f"{'='*64}")
    print(state.summary())


def cmd_auto(state: ProjectState, args: argparse.Namespace) -> None:
    """Run a full cycle: search → train → eval. Repeat for --rounds."""
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*64}")
        print(f"  AUTO ROUND {round_num}/{args.rounds}")
        print(f"{'='*64}")

        state.advance_curriculum(state.curriculum_stage + 1,
                                 note=f"auto round {round_num}")

        # Search
        print("\n--- SEARCH PHASE ---")
        cmd_search(state, args)

        # Train (only if new data)
        print("\n--- TRAIN PHASE ---")
        cmd_train(state, args)

        # Eval
        print("\n--- EVAL PHASE ---")
        cmd_eval(state, args)

        print(f"\n--- Round {round_num} complete ---")
        print(state.summary())

    state.save()


def main():
    parser = argparse.ArgumentParser(
        description="Incremental, resumable theorem proving pipeline.",
    )
    parser.add_argument("--project", default="project",
                        help="Project directory for persistent state.")

    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show project state.")

    # register
    p_reg = sub.add_parser("register", help="Register theorems.")
    p_reg.add_argument("--theorem-set", default="")
    p_reg.add_argument("--from-json", default="",
                       help="JSON file from extract_theorems.py")

    # search
    p_search = sub.add_parser("search", help="Search unsearched theorems.")
    p_search.add_argument("--difficulty", default=None,
                          help="Only search this difficulty tier.")
    p_search.add_argument("--limit", type=int, default=0,
                          help="Max theorems to search this run (0 = all).")
    p_search.add_argument("--beam-width", type=int, default=24)
    p_search.add_argument("--max-depth", type=int, default=6)
    p_search.add_argument("--action-space", default="search_v4")

    # train
    p_train = sub.add_parser("train", help="Train/retrain model.")
    p_train.add_argument("--type", default="gen", choices=["gen", "decoder", "classifier"])
    p_train.add_argument("--base-model", default="t5-small",
                         help="Base model: t5-small (seq2seq), gpt2 (decoder), distilbert (classifier)")
    p_train.add_argument("--action-space", default="search_v4")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--force", action="store_true",
                         help="Retrain even if no new data.")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate latest model.")
    p_eval.add_argument("--all", action="store_true",
                        help="Eval all theorems, not just unproved.")
    p_eval.add_argument("--max-steps", type=int, default=8)
    p_eval.add_argument("--top-k", type=int, default=8)
    p_eval.add_argument("--use-strategic", action="store_true",
                        help="Use strategic policy (backward reasoning + premise retrieval).")
    p_eval.add_argument("--use-premise-augmented", action="store_true",
                        help="Use premise-augmented generative policy (requires premise-trained model).")
    p_eval.add_argument("--action-space", default="strategic_v5",
                        help="Action space for strategic/hybrid policy.")

    # auto
    p_auto = sub.add_parser("auto", help="Full cycle: search → train → eval.")
    p_auto.add_argument("--rounds", type=int, default=1,
                        help="Number of search-train-eval cycles.")
    p_auto.add_argument("--difficulty", default=None)
    p_auto.add_argument("--limit", type=int, default=0)
    p_auto.add_argument("--beam-width", type=int, default=24)
    p_auto.add_argument("--max-depth", type=int, default=6)
    p_auto.add_argument("--action-space", default="search_v4")
    p_auto.add_argument("--type", default="gen", choices=["gen", "classifier"])
    p_auto.add_argument("--base-model", default="t5-small")
    p_auto.add_argument("--epochs", type=int, default=15)
    p_auto.add_argument("--seed", type=int, default=42)
    p_auto.add_argument("--force", action="store_true")
    p_auto.add_argument("--max-steps", type=int, default=8)
    p_auto.add_argument("--top-k", type=int, default=8)

    # selfplay
    p_sp = sub.add_parser("selfplay", help="Self-play: model generates → verify → add to corpus.")
    p_sp.add_argument("--limit", type=int, default=0)
    p_sp.add_argument("--max-steps", type=int, default=8)
    p_sp.add_argument("--top-k", type=int, default=12)
    p_sp.add_argument("--temperature", type=float, default=1.2,
                      help="Sampling temperature for diversity (>1 = more diverse).")
    p_sp.add_argument("--use-hybrid", action="store_true",
                      help="Use hybrid policy (generative + fixed actions).")
    p_sp.add_argument("--action-space", default="search_v4")

    # scale
    p_scale = sub.add_parser("scale", help="Scale up: extract → register → search → train → eval.")
    p_scale.add_argument("--preset", default="core",
                         choices=["core", "extended", "all_known"],
                         help="File preset for theorem extraction.")
    p_scale.add_argument("--skip-extraction", action="store_true")
    p_scale.add_argument("--check-availability", action="store_true")
    p_scale.add_argument("--max-per-file", type=int, default=0)
    p_scale.add_argument("--difficulty", default=None)
    p_scale.add_argument("--limit", type=int, default=0)
    p_scale.add_argument("--beam-width", type=int, default=24)
    p_scale.add_argument("--max-depth", type=int, default=6)
    p_scale.add_argument("--action-space", default="search_v4")
    p_scale.add_argument("--type", default="gen", choices=["gen", "decoder", "classifier"])
    p_scale.add_argument("--base-model", default="t5-small")
    p_scale.add_argument("--epochs", type=int, default=15)
    p_scale.add_argument("--seed", type=int, default=42)
    p_scale.add_argument("--force", action="store_true")
    p_scale.add_argument("--max-steps", type=int, default=8)
    p_scale.add_argument("--top-k", type=int, default=8)

    # build-premise-index
    p_prem = sub.add_parser("build-premise-index",
                            help="Build premise retrieval index from accumulated traces.")
    p_prem.add_argument("--out", default="",
                        help="Output path (default: project/premise_index.json)")

    args = parser.parse_args()
    state = ProjectState(args.project)

    commands = {
        "status": cmd_status,
        "register": cmd_register,
        "search": cmd_search,
        "train": cmd_train,
        "eval": cmd_eval,
        "auto": cmd_auto,
        "selfplay": cmd_selfplay,
        "scale": cmd_scale,
        "build-premise-index": cmd_build_premise_index,
    }
    commands[args.command](state, args)


if __name__ == "__main__":
    main()
