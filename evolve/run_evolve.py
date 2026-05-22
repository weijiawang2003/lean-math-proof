"""run_evolve.py — AlphaEvolve-style outer loop for Lean proof-search strategies.

  LLM/heuristic mutator  ->  candidate strategy  ->  Lean evaluator  ->
  scoring + selection  ->  next generation

Version 1 uses a deterministic heuristic mutator and (by default) a dry-run
evaluator, so the whole loop runs with no Lean, no GPU and no external API.

Examples
--------
  # fast smoke test, no Lean
  python -m evolve.run_evolve --dry-run --generations 2 --population-size 4 --survivors 2

  # full dry-run on the curriculum
  python -m evolve.run_evolve --dry-run --theorem-set curriculum_all \\
      --generations 5 --population-size 8 --survivors 3

  # real Lean evaluation (slow; needs LeanDojo + the checkpoint)
  python -m evolve.run_evolve --theorem-set curriculum_all \\
      --policy-type generative --ckpt-dir project/models/gen_v5 \\
      --generations 5 --population-size 8 --survivors 3
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evolve.candidate import SearchCandidate
from evolve.evaluator import REPO_ROOT, check_ckpt_exists, evaluate_candidate
from evolve.mutator import mutate_candidate
from evolve.population import (
    CandidateRecord,
    DEFAULT_POPULATION_PATH,
    append_record,
    select_top,
)
from evolve.template_verifier import (
    default_type_mismatch,
    default_unavailable,
    filter_templates,
    verification_summary,
)


# v4.6 template variants. Each entry is a (description, div_templates,
# div_budget) triple. The div family is the only family we vary in this
# sweep — mod / AM_GM are held constant. `v45` is the v4.5 reference and
# matches the hard-coded list in `make_seed_candidate`. Other variants
# are derivable by subset.
TEMPLATE_VARIANTS: dict[str, dict] = {
    "v45": {
        "description": "v4.5 reference; full 19-template div family; no verification",
        "verification_enabled": False,
        "div_templates": None,  # use the seed default
        "div_budget": None,
    },
    "verified": {
        "description": (
            "verified-conservative: v4.5 div family with template verification "
            "enabled; constants in _UNAVAILABLE_LEMMAS / _TYPE_MISMATCH_CONSTANTS "
            "are filtered out at config-load time"
        ),
        "verification_enabled": True,
        "div_templates": None,  # let verifier strip from default
        "div_budget": None,
    },
    "constructor": {
        "description": (
            "constructor-only: iff-decomposition + omega/simp_all. Tests whether "
            "iff-shaped div theorems benefit from the iff constructor split alone."
        ),
        "verification_enabled": True,
        "div_templates": [
            "omega",
            "simp",
            "simp_all",
            "constructor <;> intro h_split <;> omega",
            "constructor <;> intro h_split <;> simp_all",
            "constructor <;> intro h_split <;> simp_all <;> omega",
        ],
        "div_budget": 8,
    },
    "div-rewrite": {
        "description": (
            "div-rewrite-only: simp/rw with div_eq_of_lt + div_lt_iff_lt_mul "
            "variants. Tests rewrite-side closure without iff machinery."
        ),
        "verification_enabled": True,
        "div_templates": [
            "omega",
            "simp",
            "simp_all",
            "simp [Nat.div_eq_of_lt]",
            "simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]",
            "rw [Nat.div_eq_of_lt]",
            "rw [Nat.div_lt_iff_lt_mul']",
            "rw [Nat.div_lt_iff_lt_mul]",
        ],
        "div_budget": 10,
    },
    "mixed-small": {
        "description": (
            "mixed-small: union of constructor-only + div-rewrite; tests the "
            "best of both subsets against the v4.5 reference."
        ),
        "verification_enabled": True,
        "div_templates": [
            "omega",
            "simp",
            "simp_all",
            "simp [Nat.div_eq_of_lt]",
            "simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]",
            "rw [Nat.div_eq_of_lt]",
            "rw [Nat.div_lt_iff_lt_mul']",
            "rw [Nat.div_lt_iff_lt_mul]",
            "constructor <;> intro h_split <;> omega",
            "constructor <;> intro h_split <;> simp_all",
            "constructor <;> intro h_split <;> simp_all <;> omega",
        ],
        "div_budget": 14,
    },
    "verified-no-rw-eq": {
        # Hypothesis test: in v4.5 the family tactic `rw [Nat.div_eq_of_lt]`
        # fires first on iff goals like ⊢ a/b < c ↔ ... and introduces a
        # stray side-goal (1→2) that derails subsequent retrieval. Drop just
        # that template from the verified-conservative list and see if the
        # +1 closure (Nat.div_lt_iff_lt_mul') reproduces with the rest of
        # the v4.5 div family intact.
        "description": (
            "verified-no-rw-eq: v4.5 div family minus `rw [Nat.div_eq_of_lt]`"
            " and the four verifier-filtered constants; isolates the family"
            " ordering effect that closes Nat.div_lt_iff_lt_mul'."
        ),
        "verification_enabled": True,
        "div_templates": [
            "omega",
            "simp",
            "simp_all",
            "simp [Nat.div_eq_of_lt]",
            "simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]",
            # "rw [Nat.div_eq_of_lt]" intentionally removed
            "rw [Nat.div_lt_iff_lt_mul']",
            "rw [Nat.div_lt_iff_lt_mul]",
            "simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
            "simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
            "simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]",
            "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
            "constructor <;> intro h_split <;> omega",
            "constructor <;> intro h_split <;> simp_all",
            "induction {hyp_le} <;> simp_all",
        ],
        "div_budget": 18,
    },
}


def _resolve(path: str | Path) -> Path:
    """Resolve a path against the repo root if it is relative, so the script
    works regardless of the current working directory."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _apply_template_variant(
    base_div_templates: list[str],
    base_div_budget: int,
    variant_name: str,
) -> tuple[list[str], int, str, dict]:
    """Apply a v4.6 template variant to the seed candidate's div family.

    Returns (div_templates, div_budget, description_suffix, diagnostic).
    The diagnostic dict is the `verification_summary` output (or an empty
    placeholder for variant 'v45' which disables verification).

    Variants:
      v45         : v4.5 reference. Returns (base_div_templates,
                    base_div_budget, "", empty_diag).
      verified    : v4.5 templates filtered by `template_verifier`.
      constructor : prescribed minimal iff-constructor list.
      div-rewrite : prescribed minimal rewrite list.
      mixed-small : union of constructor + div-rewrite.

    For 'verified' and the three prescribed variants, the prescribed
    template list is passed through `filter_templates` so a user-supplied
    list still benefits from the unavailable-constant filter. The
    `v45` variant is the only one that skips verification.
    """
    spec = TEMPLATE_VARIANTS.get(variant_name)
    if spec is None:
        raise ValueError(
            f"Unknown --template-variant: {variant_name!r}. "
            f"Choices: {sorted(TEMPLATE_VARIANTS)}"
        )

    if not spec["verification_enabled"]:
        # v45: keep base as-is.
        return base_div_templates, base_div_budget, spec["description"], {
            "variant": variant_name,
            "verification_enabled": False,
            "template_count": len(base_div_templates),
            "filtered_template_count": 0,
            "filtered_templates": [],
            "filtered_template_constants": [],
        }

    candidates = list(spec["div_templates"] if spec["div_templates"] is not None else base_div_templates)
    unavailable = default_unavailable()
    type_mismatch = default_type_mismatch()
    kept, dropped = filter_templates(candidates, unavailable, type_mismatch)
    diag = verification_summary(candidates, unavailable, type_mismatch)
    diag["variant"] = variant_name
    diag["verification_enabled"] = True
    diag["dropped_diagnostics"] = dropped

    budget_raw = spec["div_budget"]
    if budget_raw is None:
        budget = base_div_budget
    else:
        budget = int(budget_raw)
    return kept, budget, spec["description"], diag


def make_seed_candidate(
    policy_type: str,
    ckpt_dir: str,
    template_variant: str = "constructor",
) -> tuple[SearchCandidate, dict]:
    """Generation-0 candidate: the current baseline wrapper.

    Defaults mirror the known gen_v5 baseline (top-k=8, max-steps=8 on
    curriculum_all). For policy_type=='hybrid_evolved' the seed also ships
    a fallback list tuned for the kind of arithmetic theorems gen_v5
    misses (omega/simp_arith/norm_num close most Nat.Defs goals). For
    other policy types the fallback list is still set so it's available
    to a future wrapper, but it has no effect on the underlying eval.
    """
    if policy_type == "hybrid_evolved":
        # v3.4 seed — v3.3's 10/15 ordering carried forward as the generic
        # fallback layer, plus theorem-name-aware tactic families targeting
        # the five unsolved theorems (Nat.div_*, Nat.AM_GM, Nat.add_mod_eq_ite).
        # Family-specific tactics are tried *before* the generic fallbacks
        # for any theorem whose name matches the family key.
        fallback_tactics = [
            "simp [Nat.add_comm, Nat.add_assoc, Nat.left_comm]",
            "simp_all",
            "simp",
            "omega",
            "ac_rfl",
            "simp [Nat.mul_comm, Nat.mul_assoc, Nat.left_comm]",
            "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
            "simp [Nat.add_mod, Nat.mod_eq_of_lt]",
            "rw [Nat.add_comm]",
            "simp_arith",
            "simp [Nat.add_mod, Nat.mod_eq_of_lt] at *",
            "simp [Nat.add_comm, Nat.add_assoc, Nat.left_comm, Nat.add_mod, Nat.mod_eq_of_lt]",
        ]
        tactic_templates = [
            "induction {var} with | zero => simp | succ n ih => simp [ih]",
            "cases {var} <;> simp",
            "by_cases h : {var} = 0 <;> simp [h] <;> omega",
        ]
        # Dict insertion order = activation order. Declare more-specific
        # keys before less-specific ones so e.g. add_mod_eq_ite tactics
        # queue ahead of generic mod tactics on Nat.add_mod_eq_ite.
        #
        # v3.5 cleanup applied to the v3.4 library:
        # - AM_GM: removed nlinarith / ring_nf / positivity / nlinarith
        #   [sq_nonneg (a-b)] — all four reported `unknown tactic` in this
        #   evaluation environment per v3.4 traces. Keep the layer minimal
        #   so AM_GM matches *something* (for trace continuity) without
        #   spending budget on dead tactics.
        # - add_mod_eq_ite family DISABLED. The v3.4 crash came from the
        #   `<;>`-chained by_cases+simp_all+omega combinator; v3.5 replaced
        #   that with non-chained variants and *still* observed a
        #   DojoCrashError on Nat.add_mod_eq_ite — this time from plain
        #   `simp_all [Nat.add_mod, Nat.mod_eq_of_lt]` applied to its
        #   ite-shaped goal. The crash is contained (Dojo opens a fresh
        #   session per theorem, so subsequent theorems are unaffected),
        #   but per spec we no longer ship a dedicated family for this
        #   theorem. A per-theorem deny-list would be the proper fix and
        #   is left for v3.6+. Note: the same tactic is still in the
        #   generic fallback list and the `mod` family — both will be
        #   tried on Nat.add_mod_eq_ite and both will hit the same crash.
        theorem_family_tactics = {
            "AM_GM": [
                "omega",
                "simp",
                "simp_all",
            ],
            "div": [
                "omega",
                "simp",
                "simp_all",
                "simp [Nat.div_eq_of_lt]",
                "simp [Nat.div_eq_of_lt, Nat.lt_of_lt_of_le]",
                "rw [Nat.div_eq_of_lt]",
                "rw [Nat.div_lt_iff_lt_mul']",
                "rw [Nat.div_lt_iff_lt_mul]",
                "rw [Nat.div_le_iff_le_mul]",
                "exact Nat.div_le_div_right ‹_›",
                "apply Nat.div_le_div_right",
                # v4.5 structured templates. `{hyp_pos}` / `{hyp_le}` are
                # rendered to the actual hypothesis names found in the
                # state (e.g. `hb` / `h`); templates referencing absent
                # hypotheses are skipped silently (not emitted as
                # malformed Lean). See `_render_template` /
                # `_extract_hypotheses` in evolve/strategy_wrapper.py.
                "simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
                "simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]",
                "simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]",
                "rw [Nat.div_lt_iff_lt_mul {hyp_pos}, Nat.mul_one]",
                "constructor <;> intro h_split <;> omega",
                "constructor <;> intro h_split <;> simp_all",
                "induction {hyp_le} <;> simp_all",
                "induction {hyp_le} with | refl => exact Nat.le_refl _ | step h_step ih => exact ih.trans (Nat.div_le_succ_div _ _)",
            ],
            "mod": [
                "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
                "simp [Nat.add_mod, Nat.mod_eq_of_lt]",
                "omega",
            ],
        }
        family_budgets = {
            "AM_GM": 8,
            "div": 20,  # v4.5: bumped from 12 to fit the 8 new structured templates
            "mod": 12,
        }
        # v3.6 per-theorem deny-list. simp_all [Nat.add_mod, Nat.mod_eq_of_lt]
        # is a winning tactic for Nat.add_mod_eq_add_mod_left/right (via the
        # mod family), but it crashes the Dojo REPL when applied to the
        # ite-shaped goal of Nat.add_mod_eq_ite. We don't want to remove
        # it globally — just deny it for that one theorem.
        theorem_tactic_denylist = {
            "Nat.add_mod_eq_ite": [
                "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
            ],
        }
        max_extra_tactics_per_state = 10
        timeout_per_theorem = 60
        # v4.1 → v4.2: premise retrieval on the div family with hygiene
        # filters. retrieval_top_k=8 (lowered from 10) × 3 forms (rw/simp/
        # apply) = up to 24 retrieved tactics per state — fewer than v4.1's
        # 40, focused on the forms most likely to make progress on iff /
        # propositional div lemmas. The cap auto-bumps while retrieval is
        # active so non-div theorems retain v3.6 budgeting unchanged.
        retrieval_enabled = True
        retrieval_top_k = 8
        # v4.2: drop "exact" from the default forms — none of v4.1's 783
        # attempts proved via `exact LEMMA`, and the form generates many
        # type-mismatch errors on iff/prop lemmas. Keep rw/simp/apply.
        retrieval_tactic_forms: list[str] = ["rw", "simp", "apply"]
        retrieval_filter_self = True
        retrieval_filter_unavailable = True
        # v4.3: goal-shape filter for retrieved-apply. Per-theorem only —
        # if `apply LEMMA` ever produces a strictly larger open-goal stack,
        # subsequent `apply LEMMA` candidates are pre-filtered on that
        # theorem. The lemma is NOT globally banned; rw/simp forms still
        # flow. Suppresses the pathological `apply Nat.lt_of_lt_of_le`
        # bloat that consumed most of v4.2's retrieval search budget.
        retrieval_skip_bloating_apply = True
        # v4.4: shape-aware retrieval. Classifies the current goal's head
        # connective and gates which forms (rw/simp/apply/exact) each
        # retrieved lemma emits, by lemma-shape vs goal-shape compatibility.
        # On the 6 div theorems (4 iff, 1 le, 1 lt), suppresses the
        # majority of guaranteed-fail `apply iff_lemma` emissions that
        # v4.3 was still trying.
        retrieval_shape_filter = True
        description = (
            "v4.5 hybrid_evolved seed: v3.6 library + div-family premise "
            "retrieval with self/unavailable/bloating-apply filters and "
            "shape-aware form emission; plus 8 structured div-family "
            "templates (induction on hyp_le, iff-constructor splits, "
            "rw-with-positivity-hypothesis chains)."
        )
        # v4.6 template variants. Picks a subset / verified version of
        # the div family. The seed default ('v45') leaves the family
        # untouched and emits an empty diagnostic.
        new_div_templates, new_div_budget, variant_desc, template_diag = (
            _apply_template_variant(
                theorem_family_tactics["div"],
                family_budgets["div"],
                template_variant,
            )
        )
        theorem_family_tactics["div"] = new_div_templates
        family_budgets["div"] = new_div_budget
        if template_variant != "v45":
            description = (
                f"v4.6/{template_variant} hybrid_evolved seed: "
                + variant_desc
            )
    else:
        fallback_tactics = ["simp", "aesop", "omega", "norm_num", "rfl"]
        tactic_templates = []
        theorem_family_tactics = {}
        family_budgets = {}
        theorem_tactic_denylist = {}
        max_extra_tactics_per_state = None
        timeout_per_theorem = 20
        retrieval_enabled = False
        retrieval_top_k = 0
        retrieval_tactic_forms = []
        retrieval_filter_self = True
        retrieval_filter_unavailable = True
        retrieval_skip_bloating_apply = True
        retrieval_shape_filter = True
        description = "Baseline wrapper: top-k=8, max-steps=8 (gen_v5 reference)."
        template_diag = {
            "variant": template_variant,
            "verification_enabled": False,
            "template_count": 0,
            "filtered_template_count": 0,
            "filtered_templates": [],
            "filtered_template_constants": [],
        }

    cand = SearchCandidate(
        name="seed-baseline",
        description=description,
        policy_type=policy_type,
        ckpt_dir=ckpt_dir,
        top_k=8,
        max_steps=8,
        timeout_per_theorem=timeout_per_theorem,
        fallback_tactics=fallback_tactics,
        tactic_templates=tactic_templates,
        max_extra_tactics_per_state=max_extra_tactics_per_state,
        theorem_family_tactics=theorem_family_tactics,
        family_budgets=family_budgets,
        theorem_tactic_denylist=theorem_tactic_denylist,
        retrieval_enabled=retrieval_enabled,
        retrieval_top_k=retrieval_top_k,
        retrieval_tactic_forms=retrieval_tactic_forms,
        retrieval_filter_self=retrieval_filter_self,
        retrieval_filter_unavailable=retrieval_filter_unavailable,
        retrieval_skip_bloating_apply=retrieval_skip_bloating_apply,
        retrieval_shape_filter=retrieval_shape_filter,
        metadata={"role": "seed", "template_variant": template_variant},
    )
    return cand, template_diag


def _fmt_record(rank: int, rec: CandidateRecord) -> str:
    m = rec.metrics
    proved = f"{m.proved_count}/{m.attempted_count}"
    return (
        f"  {rank:<5}{rec.candidate.name[:27]:<29}{rec.generation:<5}"
        f"{proved:<9}{m.progress_count:<7}{m.timeout_count:<6}{rec.score:<10.1f}"
    )


def print_leaderboard(records: list[CandidateRecord], title: str) -> None:
    ranked = sorted(records, key=lambda r: r.score, reverse=True)
    print()
    print(f"  {title}")
    print(
        f"  {'rank':<5}{'candidate':<29}{'gen':<5}"
        f"{'proved':<9}{'prog':<7}{'t/o':<6}{'score':<10}"
    )
    print("  " + "-" * 70)
    for i, r in enumerate(ranked, 1):
        print(_fmt_record(i, r))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaEvolve-style evolutionary search over Lean proof-search strategies."
    )
    parser.add_argument("--theorem-set", default="curriculum_all",
                        help="Theorem set name passed to the evaluator.")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=8,
                        help="New candidates evaluated per generation.")
    parser.add_argument("--survivors", type=int, default=3,
                        help="Top candidates kept as parents for the next generation.")
    parser.add_argument("--policy-type", default="generative",
                        help="Policy type for the seed candidate / real eval.")
    parser.add_argument("--ckpt-dir", default="project/models/gen_v5",
                        help="Checkpoint directory for the seed candidate.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use deterministic fake metrics; never invoke Lean.")
    parser.add_argument("--population-path", default=str(DEFAULT_POPULATION_PATH),
                        help="JSONL file that accumulates candidate records.")
    parser.add_argument("--eval-timeout-seconds", type=int, default=None,
                        help="Hard wall-clock cap (seconds) per real-eval subprocess. "
                             "Default: timeout_per_theorem × n_theorems × 1.05 + 60. "
                             "On timeout the candidate is recorded with "
                             "timeout_count = n_theorems and a heavy score penalty.")
    parser.add_argument("--template-variant", default="constructor",
                        choices=sorted(TEMPLATE_VARIANTS),
                        help="v4.6 sweep selector. constructor (default in v4.7+) = "
                             "iff-constructor div family, 26/38 on nat_defs_medium. "
                             "v45 = pre-v4.6 baseline. verified = v45 templates with "
                             "unavailable-constant filter. div-rewrite / mixed-small "
                             "/ verified-no-rw-eq = ablation variants.")
    args = parser.parse_args()

    run_id = (
        f"evolve-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    )
    run_root = _resolve(Path("project/evolve/runs") / run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    pop_path = _resolve(args.population_path)

    config = {
        "run_id": run_id,
        "theorem_set": args.theorem_set,
        "generations": args.generations,
        "population_size": args.population_size,
        "survivors": args.survivors,
        "policy_type": args.policy_type,
        "ckpt_dir": args.ckpt_dir,
        "dry_run": args.dry_run,
        "population_path": str(pop_path),
        "eval_timeout_seconds": args.eval_timeout_seconds,
        "template_variant": args.template_variant,
    }
    (run_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 72)
    print("  LeanEvolve — evolutionary search over proof-search strategies")
    print("=" * 72)
    print(f"  run id        : {run_id}")
    print(f"  mode          : {'DRY-RUN (no Lean)' if args.dry_run else 'REAL (Lean eval)'}")
    print(f"  theorem set   : {args.theorem_set}")
    print(f"  generations   : {args.generations}")
    print(f"  population    : {args.population_size} per generation")
    print(f"  survivors     : {args.survivors}")
    print(f"  population log: {pop_path}")
    if not args.dry_run:
        print(f"  eval timeout  : "
              f"{args.eval_timeout_seconds if args.eval_timeout_seconds is not None else 'derived (per_theorem × n × 1.05 + 60s)'}")
    print("=" * 72)

    # Pre-flight: verify the seed's generative checkpoint exists. Children
    # inherit ckpt_dir from the seed (the mutator never touches it), so this
    # one check catches the whole run before any compute is spent.
    seed_for_check, template_diag = make_seed_candidate(
        args.policy_type, args.ckpt_dir, template_variant=args.template_variant,
    )
    (run_root / "template_verification.json").write_text(
        json.dumps(template_diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  template var  : {args.template_variant}")
    if template_diag.get("verification_enabled"):
        print(
            f"  templates     : kept {template_diag['template_count'] - template_diag['filtered_template_count']}"
            f" / dropped {template_diag['filtered_template_count']}"
            f" ; bad constants {template_diag['filtered_template_constants']}"
        )
    if not args.dry_run:
        try:
            check_ckpt_exists(seed_for_check)
        except FileNotFoundError as exc:
            print(f"\n[run_evolve] PRE-FLIGHT CHECK FAILED: {exc}")
            raise SystemExit(2)

    run_records: list[CandidateRecord] = []

    def evaluate_and_record(cand: SearchCandidate, generation: int) -> CandidateRecord:
        cand.metadata = dict(cand.metadata)
        cand.metadata["evolve_run_id"] = run_id
        eval_out = str(run_root / "eval" / cand.name)
        metrics = evaluate_candidate(
            cand, args.theorem_set, eval_out,
            dry_run=args.dry_run,
            eval_timeout_seconds=args.eval_timeout_seconds,
        )
        rec = CandidateRecord.build(
            generation, cand, metrics,
            run_dir=None if args.dry_run else eval_out,
        )
        append_record(pop_path, rec)
        run_records.append(rec)
        m = rec.metrics
        print(
            f"    [{cand.name:<27}] proved {m.proved_count}/{m.attempted_count}"
            f"  progress {m.progress_count}  steps {m.total_steps}"
            f"  score {rec.score:.1f}"
        )
        return rec

    # ---- generation 0: the seed -----------------------------------------
    print("\nGeneration 0 — seed candidate")
    seed_cand, _ = make_seed_candidate(
        args.policy_type, args.ckpt_dir, template_variant=args.template_variant,
    )
    evaluate_and_record(seed_cand, 0)
    print_leaderboard(run_records, "Leaderboard after generation 0")

    # ---- generations 1..N -----------------------------------------------
    for gen in range(1, args.generations + 1):
        survivors = select_top(run_records, args.survivors)
        print(
            f"Generation {gen} — {len(survivors)} survivor(s) -> "
            f"{args.population_size} new candidate(s)"
        )
        for idx in range(args.population_size):
            parent = survivors[idx % len(survivors)]
            child = mutate_candidate(parent.candidate, gen, idx)
            evaluate_and_record(child, gen)
        print_leaderboard(run_records, f"Leaderboard after generation {gen}")

    # ---- wrap up ---------------------------------------------------------
    best = select_top(run_records, 1)[0]
    seed_rec = run_records[0]
    best.candidate.save_json(run_root / "best_candidate.json")

    summary = {
        "run_id": run_id,
        "config": config,
        "candidates_evaluated": len(run_records),
        "seed_score": seed_rec.score,
        "seed_proved": seed_rec.metrics.proved_count,
        "best_candidate": best.candidate.name,
        "best_score": best.score,
        "best_proved": best.metrics.proved_count,
        "best_generation": best.generation,
        "improvement_over_seed": round(best.score - seed_rec.score, 3),
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 72)
    print("  EVOLVE RUN COMPLETE")
    print("=" * 72)
    print(f"  candidates evaluated : {len(run_records)}")
    print(
        f"  seed                 : proved {seed_rec.metrics.proved_count}"
        f"/{seed_rec.metrics.attempted_count}  score {seed_rec.score:.1f}"
    )
    print(
        f"  best ({best.candidate.name}) : proved "
        f"{best.metrics.proved_count}/{best.metrics.attempted_count}"
        f"  score {best.score:.1f}  (gen {best.generation})"
    )
    delta = best.score - seed_rec.score
    verdict = "beat" if delta > 0 else ("tied" if delta == 0 else "did NOT beat")
    print(f"  best {verdict} the seed (Δscore = {delta:+.1f})")
    print(f"  artifacts            : {run_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
