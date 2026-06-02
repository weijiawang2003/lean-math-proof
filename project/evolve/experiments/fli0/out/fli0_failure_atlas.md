# FLI0 Failure Atlas

_A researcher-facing map of where the RC5 hybrid search fails and what intermediate lemmas might help. Language is deliberately hedged — these are candidate shapes, not verified requirements._

## 1. Overview

FLI0 mines the RC5V2 (complete) and RC5V3 (partial-raw) hybrid-search artifacts for theorems the full stack (RC2 → RC4 static → safe dynamic) could not prove, and groups them by the *kind* of gap each exhibits.

## 2. Source stages used

- **RC5V2** — complete, committed attribution (149 eligible, 8 solved).
- **RC5V3** — `PARTIAL_ARTIFACTS_AVAILABLE` (raw B1/B3/B5 results only; analysis layer never produced). A B5 network outage produced ~112 infra-only records, separated out.
- RC5V2 ∩ RC5V3 = 0 (disjoint fresh frontiers).

## 3. Failure corpus size

- total failures extracted: **455**
- clean failures (math, not infra/timeout/unknown-name): **327**
- seed cases selected for FLI1: **40**

## 4. Namespace distribution (clean failures)

| namespace | clean failures |
|---|---|
| Finset | 107 |
| Nat | 87 |
| Set | 57 |
| List | 41 |
| Multiset | 35 |

## 5. Main failure patterns (clean failures)

| pattern | clean count |
|---|---|
| MAP_FILTER_BIND_BRIDGE | 101 |
| ORDER_STRUCTURE_GAP | 42 |
| SUBSET_BRIDGE | 39 |
| IFF_SPLIT | 36 |
| LOW_SIGNAL | 24 |
| NEEDS_REVIEW | 23 |
| NAT_ARITH_GAP | 21 |
| MEMBERSHIP_BRIDGE | 20 |
| INDUCTION_GENERALIZATION | 11 |
| SINGLETON_CHARACTERIZATION | 6 |
| DISJOINT_BRIDGE | 3 |
| EXTENSIONALITY_NEEDED | 1 |

## 6. High-value seed cases

40 seeds selected (clean + fresh + readable + invention-friendly pattern). Full records in `cases/fli0_seed_cases.json`.

| id | theorem | ns | pattern | conf |
|---|---|---|---|---|
| FLI0-S01 | `Finset.biUnion_nonempty` | Finset | MEMBERSHIP_BRIDGE | high |
| FLI0-S02 | `Finset.card_le_card` | Finset | SUBSET_BRIDGE | high |
| FLI0-S03 | `Finset.card_le_one` | Finset | MEMBERSHIP_BRIDGE | high |
| FLI0-S04 | `Finset.card_le_one_iff` | Finset | MEMBERSHIP_BRIDGE | high |
| FLI0-S05 | `Finset.card_le_one_iff_subsingleton_coe` | Finset | SINGLETON_CHARACTERIZATION | high |
| FLI0-S06 | `Finset.card_le_one_of_subsingleton` | Finset | SINGLETON_CHARACTERIZATION | high |
| FLI0-S07 | `Finset.card_singleton_inter` | Finset | SINGLETON_CHARACTERIZATION | high |
| FLI0-S08 | `Finset.disjoint_range_addLeftEmbedding` | Finset | DISJOINT_BRIDGE | high |
| FLI0-S09 | `Finset.disjoint_range_addRightEmbedding` | Finset | DISJOINT_BRIDGE | high |
| FLI0-S10 | `Finset.eq_of_subset_of_card_le` | Finset | SUBSET_BRIDGE | high |
| FLI0-S11 | `Finset.exists_of_one_lt_card_pi` | Finset | SUBSET_BRIDGE | high |
| FLI0-S12 | `Finset.exists_subset_card_eq` | Finset | SUBSET_BRIDGE | high |
| FLI0-S13 | `Finset.filter_attach'` | Finset | SUBSET_BRIDGE | high |
| FLI0-S14 | `Finset.map_eq_of_subset` | Finset | SUBSET_BRIDGE | high |
| FLI0-S15 | `Multiset.forall_mem_map_iff` | Multiset | MEMBERSHIP_BRIDGE | high |
| FLI0-S16 | `Multiset.mem_filter` | Multiset | MEMBERSHIP_BRIDGE | high |
| FLI0-S17 | `Multiset.mem_filterMap` | Multiset | MEMBERSHIP_BRIDGE | high |
| FLI0-S18 | `Multiset.one_le_count_iff_mem` | Multiset | MEMBERSHIP_BRIDGE | high |
| FLI0-S19 | `Set.InjOn.image_diff_subset` | Set | SUBSET_BRIDGE | high |
| FLI0-S20 | `Set.InjOn.image_inter` | Set | SUBSET_BRIDGE | high |
| … | _(20 more)_ | | | |

## 7. Examples of residual goals

> **Residual goal states are unavailable in all artifacts** (the dynamic logs record tactic *outcomes*, not post-tactic goals). FLI0 reasons from the theorem statement, feature vector, retrieved lemmas, and which tactic families failed. Capturing residual goals for the chosen seeds (via a short live re-run) is an explicit FLI1 step.

Representative seed statements:
- `Finset.biUnion_nonempty` — `@[simp] lemma biUnion_nonempty : (s.biUnion t).Nonempty ↔ ∃ x ∈ s, (t x).Nonempty`
- `Finset.card_le_card` — `@[gcongr] theorem card_le_card : s ⊆ t → s.card ≤ t.card`
- `Finset.card_le_one` — `theorem card_le_one : s.card ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b`
- `Finset.card_le_one_iff` — `theorem card_le_one_iff : s.card ≤ 1 ↔ ∀ {a b}, a ∈ s → b ∈ s → a = b`
- `Finset.card_le_one_iff_subsingleton_coe` — `theorem card_le_one_iff_subsingleton_coe : s.card ≤ 1 ↔ Subsingleton (s : Type _)`

## 8. Candidate missing-lemma shapes

### MAP_FILTER_BIND_BRIDGE  (101 clean)

The goal talks about membership in (or equality of) an image / map / filter / bind / biUnion of a container. `simp`/`aesop` stall because they lack a membership-unfolding lemma for that specific transformer, so the elementwise structure never gets exposed.

- **example:** `List.bind_ret_eq_map`
  - statement: `@[deprecated bind_pure_eq_map (since`
  - why search failed: dynamic search (RC5V2) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: a membership lemma for the map/filter/bind/image, e.g. `x ∈ f <$> s ↔ ∃ y ∈ s, f y = x`.

### ORDER_STRUCTURE_GAP  (42 clean)

Order/lattice goals where the tactic families have no order-specific structural route. Lower priority for lemma invention (often not a single missing lemma).

- **example:** `Finset.card_le_two`
  - statement: `theorem card_le_two : card {a, b} ≤ 2`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: an order/lattice structural lemma; current tactic batteries have no order-specific route.

### SUBSET_BRIDGE  (39 clean)

The goal is (or reduces to) a subset relation. Search fails when it cannot turn `s ⊆ t` into the pointwise `∀ x, x ∈ s → x ∈ t` form (or a pair/singleton subset characterization) that downstream automation can chew on.

- **example:** `Finset.card_le_card`
  - statement: `@[gcongr] theorem card_le_card : s ⊆ t → s.card ≤ t.card`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: a `s ⊆ t ↔ ∀ x, x ∈ s → x ∈ t` / pair-subset characterization for this type.

### IFF_SPLIT  (36 clean)

The goal is a biconditional. Single-shot tactics fail to make progress because the two directions need different reasoning; a `constructor` split plus a one-directional bridge lemma on each side is the likely shape.

- **example:** `Finset.card_eq_zero`
  - statement: `@[simp] lemma card_eq_zero : s.card = 0 ↔ s = ∅`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: no new lemma per se — a `constructor`/two-direction split helper or the missing one-directional bridge lemma each side needs.

### NAT_ARITH_GAP  (21 clean)

Nat/Int arithmetic beyond the current omega/nlinarith reach. Usually not a reusable bridge lemma.

- **example:** `Nat.div_add_mod'`
  - statement: `lemma div_add_mod' (a b : ℕ) : a / b * b + a % b = a`
  - why search failed: dynamic search (RC5V2) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: arithmetic beyond the current search (omega/nlinarith gap), not an obvious reusable bridge.

### MEMBERSHIP_BRIDGE  (20 clean)

The goal is an iff whose left side is a membership statement about a derived structure. The missing piece appears to be a `x ∈ … ↔ <condition>` rewrite that bridges the membership to an elementwise predicate.

- **example:** `Finset.card_le_one`
  - statement: `theorem card_le_one : s.card ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: a `x ∈ <transformed container> ↔ <elementwise condition>` rewrite lemma.

### INDUCTION_GENERALIZATION  (11 clean)

A List/Multiset goal over a recursive constructor (cons/append/foldr…) where plain simp/aesop cannot fold through the recursion; a generalized induction helper with a stronger hypothesis seems needed.

- **example:** `List.dedup_append`
  - statement: `theorem dedup_append (l₁ l₂ : List α) : dedup (l₁ ++ l₂) = l₁ ∪ dedup l₂`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: a generalized induction helper (stronger IH over the recursive structure) that the elementwise goal can fold through.

### SINGLETON_CHARACTERIZATION  (6 clean)

The goal characterizes a singleton (or a `card ≤ 1` / subsingleton condition). The candidate is a `… = {x} ↔ …` (or `y ∈ {x} ↔ y = x`) lemma for the container.

- **example:** `Finset.card_singleton_inter`
  - statement: `theorem card_singleton_inter [DecidableEq α] : ({a} ∩ s).card ≤ 1`
  - why search failed: dynamic search (RC5V3) result was 'failed'; retrieval surfaced lemmas but none closed it
  - candidate lemma shape: a `… = {x} ↔ …` (or `y ∈ {x} ↔ y = x`) characterization lemma for this container.

## 9. What FLI1 should try next

1. Re-run the 40 seeds live to **capture residual goals** (the one missing ingredient).
2. For the bridge patterns (membership / subset / disjoint / map-filter-bind), **synthesize the candidate `↔` lemma**, prove it (or retrieve it), add it as a gated `simp [L]` enabling action, and check whether the downstream theorem now closes — the RC4B/RC4C deployment pattern.
3. Start with the highest-confidence, most-clustered families (Finset/List membership & subset bridges) where one invented lemma may rescue several theorems.
4. Defer ORDER_STRUCTURE_GAP / NAT_ARITH_GAP (rarely a single missing lemma).

## 10. Caveats

- No residual goals → pattern labels are inferred from statement + retrieval, not from a stuck goal state. Conservative by design.
- RC5V3 is partial; its B5 network outage means some V3 theorems only have B1 live data.
- Several Finset seeds are `card_*` near-variants (one invented lemma may cover the cluster — or none generalize); treat the cluster as a single bet, not many.
- Labels are multi-signal heuristics; a label says a failure *suggests* a lemma family, never that it *requires* one.
