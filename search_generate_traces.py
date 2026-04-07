import argparse
import uuid
from dataclasses import dataclass

from lean_dojo import Dojo

from actions import get_action_space, list_action_spaces
from env import make_repo, make_theorem, run_transition
from evaluate_traces import evaluate
from experiment_io import init_run_artifacts, write_metrics
from tasks import get_theorems
from trace_io import append_jsonl


@dataclass
class Node:
    state: object | None
    history: list[str]
    finished: bool


def _is_missing_trace_artifact_error(exc: Exception) -> bool:
    msg = str(exc)
    return exc.__class__.__name__ == "DojoInitError" or ".ast.json" in msg


def _availability_reason(exc: Exception) -> str:
    if _is_missing_trace_artifact_error(exc):
        return "missing_trace_artifact"
    if "Failed to locate the theorem" in str(exc):
        return "theorem_not_found"
    return "other_init_error"


def _partition_available_theorems(theorems):
    repo = make_repo()
    available = []
    unavailable = []
    for cfg in theorems:
        try:
            theorem = make_theorem(repo, cfg)
            with Dojo(theorem):
                pass
            available.append(cfg)
        except Exception as exc:
            unavailable.append({
                "file_path": cfg.file_path,
                "full_name": cfg.full_name,
                "reason": _availability_reason(exc),
                "error": str(exc),
            })
    return available, unavailable


def search_and_log_for_theorem(
    cfg,
    out_path: str,
    run_id: str,
    beam_width: int = 16,
    max_depth: int = 4,
    actions: list[str] | None = None,
) -> bool:
    actions = actions or get_action_space("core_v1")
    repo = make_repo()
    theorem = make_theorem(repo, cfg)
    episode_id = cfg.full_name

    try:
        with Dojo(theorem) as (dojo, init_state):
            print(f"\n=== Theorem: {cfg.full_name} ===")
            print(init_state.pp)
            print(f"#goals: {init_state.num_goals}\n")

            beam: list[Node] = [Node(state=init_state, history=[], finished=False)]

            for depth in range(1, max_depth + 1):
                print(f"\n==== Depth {depth} ====")
                new_beam: list[Node] = []

                for node in beam:
                    if node.finished or node.state is None:
                        new_beam.append(node)
                        continue

                    for tac in actions:
                        outcome = run_transition(
                            dojo,
                            theorem,
                            node.state,
                            tac,
                            step=depth,
                            run_id=run_id,
                            episode_id=episode_id,
                            method="beam_search",
                        )
                        if outcome.is_error:
                            continue

                        goals_before = outcome.record.num_goals_before
                        goals_after = outcome.record.num_goals_after
                        if goals_before is not None and goals_after is not None and goals_after <= goals_before:
                            append_jsonl(out_path, outcome.record)

                        new_history = node.history + [tac]

                        if outcome.is_finished:
                            print(f"FOUND PROOF at depth {depth} with history {new_history}")
                            new_beam.append(Node(state=None, history=new_history, finished=True))
                        else:
                            print(f"  OK: `{tac}` : {goals_before} -> {goals_after} goals")
                            new_beam.append(Node(state=outcome.next_state, history=new_history, finished=False))

                if not new_beam:
                    print("No valid successors, stop.")
                    break

                def score(node: Node):
                    if node.finished:
                        return -1000
                    return node.state.num_goals

                new_beam.sort(key=score)
                beam = new_beam[:beam_width]
                print(f"Beam size: {len(beam)}, finished: {sum(n.finished for n in beam)}")
    except Exception as exc:
        if _is_missing_trace_artifact_error(exc):
            print(
                f"[WARN] Skip theorem {cfg.full_name}: {exc}\n"
                "       Hint: this usually means LeanDojo trace artifacts (*.ast.json) "
                "for that file are missing in cache."
            )
            return False
        raise

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate traces from beam search.")
    parser.add_argument("--theorem-set", default="toy_search")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--fail-on-skip", action="store_true", help="Exit non-zero if any theorem is skipped.")
    parser.add_argument("--fail-on-unavailable", action="store_true", help="Exit non-zero if availability precheck filters any theorem.")
    parser.add_argument("--action-space", default="search_v2", choices=list_action_spaces())
    args = parser.parse_args()

    run_id = f"search-{uuid.uuid4().hex[:8]}"
    artifacts = init_run_artifacts(
        base_dir=args.out_dir,
        method="beam_search",
        run_id=run_id,
        config={
            "method": "beam_search",
            "theorem_set": args.theorem_set,
            "beam_width": args.beam_width,
            "max_depth": args.max_depth,
            "action_space": args.action_space,
        },
    )

    trace_path = args.out or artifacts["traces_path"]
    open(trace_path, "w", encoding="utf-8").close()

    requested_theorems = get_theorems(args.theorem_set)
    available_theorems, unavailable = _partition_available_theorems(requested_theorems)
    if unavailable:
        print(f"[WARN] Availability precheck filtered {len(unavailable)} theorem(s).")
        for item in unavailable:
            print(f"  - {item['full_name']}: {item['reason']}")

    if args.fail_on_unavailable and unavailable:
        raise SystemExit(4)

    n_ok = 0
    n_skipped_runtime = 0
    actions = get_action_space(args.action_space)
    for theorem in available_theorems:
        ok = search_and_log_for_theorem(
            theorem,
            trace_path,
            run_id,
            args.beam_width,
            args.max_depth,
            actions,
        )
        if ok:
            n_ok += 1
        else:
            n_skipped_runtime += 1

    metrics = evaluate(trace_path)
    metrics["run_summary"] = {
        "requested_theorems": len(requested_theorems),
        "available_theorems": len(available_theorems),
        "processed_theorems": n_ok,
        "skipped_theorems_runtime": n_skipped_runtime,
        "unavailable_theorems": len(unavailable),
        "unavailable_details": unavailable,
    }
    write_metrics(artifacts["metrics_path"], metrics)
    print(f"\nRun artifacts: {artifacts['run_dir']}")
    print(
        "Processed theorems: "
        f"{n_ok}, unavailable(precheck): {len(unavailable)}, runtime skipped: {n_skipped_runtime}"
    )

    if args.fail_on_skip and (n_skipped_runtime > 0 or unavailable):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
