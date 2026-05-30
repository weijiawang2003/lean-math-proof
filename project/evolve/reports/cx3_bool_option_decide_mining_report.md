# CX3 — Bool/Option decide-family mining

**Arc type:** mining-only (no training). **Branch:**
`cx3-bool-option-decide-mining`. **Router used for eval:** `ns24_router`
(NS24-promoted; Bool/Option fall through to the default checkpoint
`gen_v5_ns12_balanced`). **Wrapper:** NS9 best genome.

## 0. NS24 recap — why a fresh namespace

NS24 confirmed the **Int omega surface is saturated**: training the
NS23-repaired minimal-omega aggregate reproduced NS22 (57 → 58/156, +1)
because NS22's `fallback_omega` ablation had already absorbed the omega
policy. The minimal-tactic principle (NS23) is an attribution/gating
step, not a win-generator on an already-absorbed family. The natural
next move is a **fresh namespace with no base prior** — the setting
where NS15/NS22-style absorption actually produced broad transfer. CX3
probes Bool/Option for a short-tactic family analogous to Int/omega.

## 1. Catalog audit (Stage 2)

`scripts/cx3_bool_option_catalog_audit.py` →
`project/data/cx3_bool_option_catalog_audit_meta.json`,
`project/evolve/reports/cx3_bool_option_catalog_audit.md`.

| | Bool | Option | total |
|---|---:|---:|---:|
| candidates (catalog + source scan) | 42 | 83 | 125 |
| already used/probed in prior sets | 35 | 4 | 39 |
| **fresh unused** | **7** | **79** | **86** |
| &nbsp;&nbsp;of which verified-available | 0 | 42 | 42 |
| &nbsp;&nbsp;of which needs-probe | 7 | 37 | 44 |

> **Bool is exhausted.** Every verified-available Bool theorem
> (`Mathlib/Data/Bool/Basic.lean`) was already consumed by
> `cx1_bool_option_int` in CX1. The fresh surface is essentially
> **Option** (map/bind/pmap/pbind/isSome/isNone/getD/elim/orElse). The
> 7 "fresh" Bool only come from additional-file source scans.

## 2. Theorem sets built (Stage 3)

`scripts/build_cx3_theorem_sets.py` →
`project/evolve/routing/cx3_theorem_sets.json` (loaded by
`tasks._load_cx3_sets`). All 86 fresh candidates, partitioned disjointly:

| set | n | intent |
|---|---:|---|
| `cx3_bool_decide_easy` | 5 | Bool props → `decide` |
| `cx3_bool_simp_medium` | 2 | Bool/Set → `simp`/`ext` |
| `cx3_option_simp_easy` | 32 | Option simp surface |
| `cx3_option_cases_medium` | 11 | Option map/bind needing case split |
| `cx3_bool_option_mixed` | 36 | remaining fresh Option |

## 3. Raw vs wrapper matrix (Stage 4)

`scripts/cx3_extract_probe.py` →
`project/data/cx3_bool_option_probe_meta.json`. Eval `--top-k 8
--max-steps 8`, raw routed vs NS9-wrapper+routed.

| set | total | avail | raw wins | wrap wins | **wrapper-only** |
|---|---:|---:|---:|---:|---:|
| cx3_bool_decide_easy | 5 | 5 | 2 | 2 | **0** |
| cx3_bool_simp_medium | 2 | 2 | 1 | 1 | **0** |
| cx3_option_simp_easy | 32 | 32 | 15 | 15 | **0** |
| cx3_option_cases_medium | 11 | 11 | 7 | 7 | **0** |
| cx3_bool_option_mixed | 36 | 33 | 18 | 18 | **0** |
| **total** | 86 | 83 | **43** | **43** | **0** |

**The raw routed model and the NS9 wrapper solve an identical 43/83
theorems** (verified: `raw_solved == wrap_solved`, no raw-only, no
wrapper-only). On Bool/Option the wrapper is a **complete no-op** — its
17-skeleton strategy adds nothing the default generative model
(`gen_v5_ns12_balanced`) doesn't already emit. 3 discovered names did
not load (`Option.get`, `Option.map`, `Option.map_map` — deprecated /
signature-shadowed), counted unavailable.

**Wrapper-only wins = 0 ⇒ there is nothing to distill from the
wrapper.** This is the headline CX3 result.

## 4. Minimal-tactic relabeling (Stage 5 — mandatory post-NS23/NS24)

With wrapper-only == 0, the gate-relevant question shifts to
**headroom**: of the currently-unsolved theorems, how many does a short
tactic close from the initial state (i.e. the model simply isn't
emitting the right tactic)? `scripts/cx3_relabel_minimal_tactics.py
--key relabel_candidates` ran the battery (`assumption, rfl, decide,
simp, simp_all, norm_num, tauto, aesop, constructor<;>simp,
constructor<;>decide, intros<;>cases<var><;>simp_all`, + wrapper
fallback) from the initial state over **all 83 available** Bool/Option
theorems. → `project/data/cx3_minimal_tactic_labels.json`,
`project/data/cx3_minimal_family_pools_meta.json`.

- **Resolved by battery: 59/83** (vs 43 by the model — the battery
  closes 16 more).
- **Already-solved minimal families** (43): `fallback_rfl` 18,
  `aesop` 14, `simp_other` 7, `simp_all` 1, Bool {rfl,simp,aesop} 3.
  The model already emits rfl/aesop/simp on the easy surface —
  consistent with wrapper-only == 0.
- **Headroom** (unsolved-by-model but battery-closeable, 16):

  | minimal family | unique | count gate (≥5) |
  |---|---:|:---:|
  | `cases_simp \| Option` | **13** | ✓ |
  | `fallback_rfl \| Option` | 2 | ✗ |
  | `aesop \| Bool` | 1 | ✗ |

- **Hard tail** (24): no battery tactic closes them (need specific
  lemmas / rewrites) — e.g. `Option.map_injective`, `orElse_eq_some`,
  `mem_pmem`, `Bool.forall_bool`. Not a short-tactic family.

## 5. Gate decision (Stage 6)

**No clean short-token training gate is met.** Two independent reasons:

1. **Wrapper-only gate: empty (0).** The spec's primary gate requires
   ≥5 wrapper-only wins of one family. There are zero. The wrapper is a
   no-op here, so there is no wrapper trick to distill.

2. **The only count-meeting headroom pool fails the "short stable
   tactic" criterion.** `cases_simp | Option` has 13 unique, same
   family, same namespace — but its minimal tactic is the **compound**
   `intros <;> cases <var> <;> simp_all`, with a **per-theorem variable
   name** (`a`/`i`/`o`/`x`/`y`) and ~35 chars. Plain
   `simp`/`simp_all`/`aesop`/`decide` were all tried first and
   **genuinely fail** (aesop: 10 LeanError + 3 partial-progress, **zero
   timeouts** — not a timeout artifact). This is precisely the
   structured-tactic class that NS22 found does **not** memorize at the
   60M-param scale (NS22's `iff_5x`/`iff_10x` nulled out). The
   minimal-tactic relabel — the mandatory NS23 discipline — did exactly
   its job: it **prevented a likely-wasteful NS25** by revealing that
   the only headroom family is structured, not a short token like
   `omega`/`aesop`.

Expected high-value pools from the spec (decide/Bool, simp/Bool,
simp/Option, cases_simp/Option): the first three are empty as
short-token wrapper-only pools; the fourth exists only as a
structured-tactic headroom pool, which is the trap NS22 documented.

## 6. Preservation smoke (Stage 7)

CX3 changes **no** router, checkpoint, or genome (mining only), so
preservation is identical to NS24 by construction. Confirmed from the
existing NS24-router runs:

| config | demo_v1 | nat_defs_medium |
|---|---:|---:|
| routed raw (ns24_router) | 10/15 | 23/38 |
| NS9 wrapper + ns24_router | 11/15 | 37/38 |

Unchanged from NS24. NS24 router/checkpoints/artifacts intact.

## 7. Recommendation

**Bool/Option short-token fresh-namespace mining returns negative.** The
default model already covers the easy Bool/Option surface; the wrapper
adds nothing; the only count-meeting headroom is a structured
cases-split tactic of the NS22 non-memorizable class.

- **Do not** launch a naive NS25 imitation run on `cases_simp/Option` —
  by NS22 precedent it is likely to null out, and it would re-make the
  mistake NS23/NS24 exist to prevent.
- **One genuinely novel research bet, if pursued:** the cases-split
  variable is **state-readable** (the first Option/Bool binder), so
  unlike NS22's fixed 49-char template, a *state-conditioned*
  generative model could in principle learn
  `intros <;> cases <binder> <;> simp_all`. An NS25 here would be a
  deliberate **test of whether state-conditioned variable-fill compound
  tactics memorize at 60M** — a distinct hypothesis from NS22's fixed
  template, not a clear-cut gate. High oversample, flagged as a bet.
- **Otherwise pivot:** a different fresh namespace (List/Multiset
  short-tactic surface is large and unprobed by this lens), or
  genuinely-unseen held-out Int (the CX2 audit left ~50 sub-bitwise/dvd
  candidates), rather than more Bool/Option.

Two `fallback_rfl | Option` theorems (`Option.elim'_none`,
`Option.elim'_some`) are closed by plain `rfl` yet the model missed them
— a trivial decode-variance curiosity, not a pool.

## Artifacts

- `scripts/cx3_bool_option_catalog_audit.py`,
  `scripts/build_cx3_theorem_sets.py`,
  `scripts/cx3_extract_probe.py`,
  `scripts/cx3_relabel_minimal_tactics.py`,
  `scripts/cx3_run_eval.sh`
- `project/evolve/routing/cx3_theorem_sets.json`
- `project/data/cx3_bool_option_catalog_audit_meta.json`,
  `project/data/cx3_bool_option_probe_meta.json`,
  `project/data/cx3_minimal_tactic_labels.json`,
  `project/data/cx3_minimal_family_pools_meta.json`
- Eval traces/logs under `project/evolve/eval_runs/cx3_*` (gitignored).
