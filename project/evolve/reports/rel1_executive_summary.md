# REL1 — executive summary

## Project in one paragraph

This project builds an automated theorem-proving system for Lean 4 / Mathlib: a
routed generative tactic policy wrapped by an evolved, deterministic search
layer. Starting from a strong base wrapper (**NS9**), it adds only proven,
namespace-gated, additive improvements and ships them as a single
release-candidate config (**RC1**). The central scientific contribution is a
**symbolic-action abstraction** — typed, state-instantiated tactics such as
`induction {v} using Multiset.induction_on <;> simp_all` — which makes
variable-dependent proof steps stable, learnable labels while the wrapper fills
the variable from the live proof state. RC1 combines NS9 with a Multiset
induction oracle (WX3) and a narrow `Set.Finite` `aesop` fallback (MX2),
delivering **+15 proofs beyond NS9 with zero regressions and zero off-gate
emissions**, while a learned symbolic-action predictor (AX4) and a depth-2
sequence search (SX1) are kept off by default after rigorous evaluation showed
the deterministic oracle wins when emission is free.

## Main result

| config | wins (measured surfaces) | Δ vs NS9 | regressions |
|---|---|---|---|
| NS9 baseline wrapper | 106 | — | — |
| **RC1 production wrapper** | **121** | **+15** | **0** |

- **WX3 Multiset induction oracle: +12** (held-out Multiset surfaces).
- **MX2 narrow Set.Finite aesop fallback: +3** (the `Set.Finite.toFinset` misses).
- Disjoint namespace gates ⇒ additive, no negative interaction.
- Off-gate emissions = 0; canonical floors preserved (demo 11/15, medium 37/38,
  large 49/65).

## Key architectural findings

1. **Short, stable, variable-independent tactics are raw-SFT-ready**
   (`omega`/`aesop`: NS15/NS22) — but that family saturates fast (NS24).
2. **Variable-dependent tactics need symbolic actions, not SFT** — the wrapper
   reads the variable from the state; a 60M model cannot reliably memorize a
   variable-fill compound (WX1/WX2/AX1).
3. **Multiset is the main symbolic-action payoff** — a quotient-aware
   `induction_on` action opened a fresh namespace where the base policy is weak
   (WX3: +25 beyond NS9, 20 clean labels), unlike the exhausted Option/List.
4. **A learned selector works, but the deterministic oracle wins when emission is
   free** — AX4's predictor cleared the held-out promotion bar (retain 53.8%, 0
   regressions) yet the always-emit oracle retains 100% at zero cost, so the
   predictor stays off; selectivity only pays under costly search.
5. **Sequence search gave no net gain over the production wrapper** — the
   best-first search already performs the depth-2 follow-up, so SX1's fixed
   battery is subsumed (0 net wins); kept off by default.
6. **Set/Finset are better handled by narrow `aesop`-style fallbacks** — on
   strong-base-policy namespaces, cheap namespace-gated battery tactics capture
   the residual headroom (MX2 for Set, NS21 for Finset); symbolic ext/cases
   actions there fire but never close (MX1).

## What ships

- **Production:** the RC1 deterministic wrapper (`rc1_production_wrapper.json`).
- **Experimental (off by default):** AX4 learned predictor, SX1 sequence search.
- **Not promoted:** broad `Set.` aesop (overfires), MX1 Set/Finset symbolic ext
  actions, SX1 sequence production flag.
