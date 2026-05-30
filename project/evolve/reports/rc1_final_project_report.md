# RC1 — final project report

## 1. Executive summary

This project evolved an automated Lean/Mathlib proof-search system from a raw
generative policy into a layered **production wrapper stack (RC1)** that adds
deterministic, namespace-gated, provably-safe improvements on top of a strong
base. The headline result:

- **RC1 = NS9 base ⊕ WX3 Multiset induction oracle ⊕ MX2 narrow Set.Finite aesop
  fallback**, with the AX4 learned predictor and SX1 sequence search kept
  **off by default**.
- On the surfaces where it acts, **RC1 adds +15 wins beyond the NS9 wrapper**
  (+12 Multiset from WX3, +3 Set.Finite from MX2), with **0 regressions** and
  **0 off-gate emissions**; canonical floors (medium 37/38, large 49/65,
  demo 11/15) are preserved exactly.
- The scientific arc established *where* each technique pays: short stable
  tactics → raw SFT; variable-dependent tactics → state-aware symbolic actions;
  deterministic oracle wrappers beat learned selectors when emission is free;
  and strong-base-policy namespaces are best served by cheap battery tactics,
  while weak structural namespaces (quotient types) benefit from symbolic actions.

## 2. Baselines

- **Raw model** — the routed generative policy (NS24 router over per-namespace
  gen_v5 checkpoints). Decodes top-k tactics per state; no wrapper.
- **NS9 wrapper** — the evolved base genome: generative top-k + retrieval +
  tactic templates + family tactics + per-state budget + fallbacks, deterministic
  and shape-aware. This is the production baseline RC1 builds on; its genome is
  never modified.

## 3. Learning track (fine-tunes)

- **NS15 (Nat)** — successful raw-tactic distillation; the Nat route.
- **NS22 (Int/omega)** — successful: short `omega`-family tactics transfer as
  raw SFT labels.
- **NS24 (Int minimal-omega aggregate)** — confirmed the Int/omega surface is
  **saturated** (57→58, near-null); the relabeled iff-group was already solved by
  NS22's fallback. No new short-token SFT family has appeared since Int/omega.
  Lesson: raw-tactic SFT works only for short, stable, variable-independent
  tactics, and that well is largely dry.

## 4. Wrapper / symbolic track

- **WX1 (Option)** — state-aware `cases <var> <;> simp` wrapper: +19 Option wins
  beyond NS9, 0 regressions. Wrapper-ready, not SFT-ready.
- **WX2 (List)** — the same pattern generalizes to List: +10 wins; combined
  WX1+WX2 = +29 beyond NS9. Quotient types (Multiset) excluded — raw `cases`
  does not apply.
- **AX1 (symbolic abstraction)** — a typed `SymbolicAction` layer
  (`CASES_SIMP[List,simp_all]`, etc.): makes the variable-dependent family a
  stable *label* while the wrapper instantiates the variable from the state.
  Reproduces WX2 exactly; the 27 WX wins collapse to 4 symbolic labels.
- **AX2 (cap)** — RED: mining could not grow the single-shot symbolic-label
  dataset past ~27 (Option exhausted; fresh List wins are multi-step). The
  symbolic layer looked capped — *by namespace choice, as WX3 then showed*.
- **WX3 (Multiset breakthrough)** — a quotient-aware
  `induction {var} using Multiset.induction_on <;> simp_all` action opened a
  fresh namespace: **+25 wins beyond NS9, 0 regressions, 20 clean single-shot
  symbolic labels** — the first surface to clear AX2's null result. GREEN.
- **AX3 / AX4 (learned selector proof-of-concept)** — trained the first learned
  symbolic-action predictor (TF-IDF char_wb + logistic regression over the proof
  state). AX4 reached GREEN (46 labels) and **cleared the held-out promotion bar**
  (retain 53.8%, 0 regressions, 0 non-Multiset FP) — but for a single *free*
  additive action the deterministic oracle still dominates (retains 100% vs 54%),
  so the predictor stays **off by default**.
- **SX1 (sequence search — negative)** — depth-2 symbolic sequences. The existing
  best-first search already performs the depth-2 follow-up (~9 follow-ups per
  advanced symbolic state), so a fixed-battery sequence mode is subsumed: 0 net
  wins over production. Gate B (dataset-generation), kept off by default.
- **MX1 (frontier saturation)** — live LeanDojo mining over fresh Finset/Set
  frontiers (Multiset/List/Option exhausted). New Finset/Set ext/cases actions
  fire but never close (Finset) or are `aesop`-over-attributed (Set): **0 clean
  new symbolic labels**. The symbolic-action layer is **namespace-saturated**.
- **MX2 (Set aesop patch)** — acted on MX1's cheap follow-up: a narrow
  `Set.Finite`/`toFinset` aesop fallback (NS19 pattern) captures the 2 clean-aesop
  Set misses, 0 regressions, 0 off-Set emissions. The broad `Set.` gate added no
  extra wins (overfiring only) → Gate B, narrow patch.

## 5. RC1 production stack

**Included** (`rc1_production_wrapper.json`):
1. **NS9 base wrapper** — the proven baseline genome.
2. **WX3 Multiset induction oracle** — `MULTISET_INDUCTION_SIMP[Multiset,simp_all]`
   + `[,simp]`, gated to `Multiset.`; free always-emit, oracle beats the learned
   selector.
3. **MX2 narrow Set aesop fallback** — `aesop` gated to `Set.Finite.`/`Set.toFinset`.

**Excluded:** AX4 learned predictor (off), SX1 sequence search (off), broad
`Set.` aesop (overfires), MX1 Finset/Set ext/cases actions (never close / base
saturates).

**Final benchmark** (`rc1_full_benchmark_meta.json`; RC1 composed from the
namespace-disjoint WX3 and MX2 deltas + a live confirmation run):

| surface | A raw | B NS9 | C RC1 | Δ vs NS9 | regr |
|---|---|---|---|---|---|
| Multiset induction heldout | 10 | 10 | 14 | **+4** | 0 |
| Multiset induction heldout2 | 12 | 12 | 19 | **+7** | 0 |
| ax3 Multiset heldout | 0 | 0 | 1 | **+1** | 0 |
| Set.Finite known | 0 | 0 | 2 | **+2** | 0 |
| Set.Finite frontier | 3 | 3 | 4 | **+1** | 0 |
| demo_v1 (floor) | — | 11 | 11 | 0 | 0 |
| nat_defs_medium (floor) | — | 37 | 37 | 0 | 0 |
| ns17_set_extra (control) | — | 18 | 18 | 0 | 0 |
| ns17_finset_extra (control) | — | 15 | 15 | 0 | 0 |

**Component ablation** (`rc1_component_ablation.md`): WX3 contributes **+12**
Multiset wins, MX2 contributes **+3** Set.Finite wins; disjoint namespace gates ⇒
**no negative interaction**, RC1 gain = WX3 + MX2 = **+15**.

## 6. Main scientific lessons

1. **Short, stable, variable-independent tactics are raw-SFT-ready**
   (`omega`/`aesop`: NS15/NS22) — but that family saturates quickly (NS24).
2. **Variable-dependent tactics need symbolic actions**, not SFT: the wrapper
   reads the variable from the live state (`cases <var>`, `induction <var> using
   Multiset.induction_on`) while the label stays stable (WX1/WX2/AX1/WX3).
3. **Deterministic oracle wrappers beat learned selectors when emission is free**
   — WX3's always-emit Multiset action retains 100% of wins at zero cost; AX4's
   learned selector retains only 54%, so it stays off (AX3/AX4).
4. **Learned selectors matter only when emission has cost** — i.e. under
   multi-action / costly search; SX1 showed the current search already subsumes
   the cheap depth-2 case, so there is no cost yet for selectivity to save.
5. **Minimal-tactic relabeling prevents false family attribution** — repeatedly
   decisive (NS23, CX3, MX1, MX2): MX1's "symbolic" Set wins and MX2's 3rd win
   were really plain `aesop`/`assumption`. Always relabel before crediting a family.
6. **Match the technique to the base policy's weakness**: strong-base-policy
   namespaces (Finset/Set, well-served by the routed generative policy) prefer
   cheap namespace-gated battery tactics (`aesop`, as NS21/MX2 did); weak
   structural namespaces (the Multiset quotient) benefit from symbolic actions
   (WX3). Symbolic-action learning is reserved for the latter.

## 7. Next research directions

- **Larger symbolic-action learner** — grow the Multiset clean-label pool well
  beyond 46 (broader discovered catalog / cross-namespace induction transfer) so
  a learned selector has the data to matter; revisit AX4 promotion only with a
  genuinely costly emission setting.
- **Learned selector for costly search only** — apply selectivity where emission
  is expensive (multi-action search, term synthesis), not to single free actions.
- **Broader namespace discovery** — the symbolic-action layer pays on
  weak-base-policy structural surfaces; the open question is finding the *next*
  such namespace (quotients, subtypes, order structures) where the base policy is
  weak and a stable symbolic label exists.
- **Cheap battery sweeps** — for strong-base namespaces, continue the NS21/MX2
  pattern (namespace-gated `aesop`/battery additions) rather than symbolic
  learning; these are low-risk, additive, and occasionally net new wins.
- **Write-up / cleanup** — the arc (NS→WX→AX→SX→MX→RC1) is a coherent story about
  *where* SFT vs wrapper vs symbolic vs battery each win; worth consolidating into
  a paper-style report.
