# FLI1 candidate lemma summary

- candidates: 40 | high-conf/low-risk: 29
- by pattern: {'SUBSET_BRIDGE': 8, 'MAP_FILTER_BIND_BRIDGE': 8, 'MEMBERSHIP_BRIDGE': 7, 'INDUCTION_GENERALIZATION': 6, 'IFF_SPLIT': 4, 'SINGLETON_CHARACTERIZATION': 4, 'DISJOINT_BRIDGE': 2, 'EXTENSIONALITY_NEEDED': 1}
- by namespace: {'Finset': 14, 'List': 14, 'Multiset': 4, 'Nat': 4, 'Set': 4} | confidence: {'high': 29, 'low': 4, 'medium': 7}

| id | seed | ns | pattern | conf | lemma goal |
|---|---|---|---|---|---|
| FLI1-L01 | FLI0-S01 | Finset | MEMBERSHIP_BRIDGE | high | `∃ x ∈ s, (t x).Nonempty` |
| FLI1-L02 | FLI0-S02 | Finset | SUBSET_BRIDGE | high | `s.card ≤ t.card` |
| FLI1-L03 | FLI0-S03 | Finset | MEMBERSHIP_BRIDGE | high | `∀ a ∈ s, ∀ b ∈ s, a = b` |
| FLI1-L04 | FLI0-S04 | Finset | MEMBERSHIP_BRIDGE | high | `∀ {a b : α}, a ∈ s → b ∈ s → a = b` |
| FLI1-L05 | FLI0-S08 | Finset | DISJOINT_BRIDGE | high | `∀ ⦃a_1 : ℕ⦄, a_1 < a → ∀ x < b, ¬a + x = a_1` |
| FLI1-L06 | FLI0-S09 | Finset | DISJOINT_BRIDGE | high | `∀ ⦃a_1 : ℕ⦄, a_1 < a → ∀ x < b, ¬x + a = a_1` |
| FLI1-L07 | FLI0-S10 | Finset | SUBSET_BRIDGE | high | `s = t` |
| FLI1-L08 | FLI0-S11 | Finset | SUBSET_BRIDGE | high | `∃ i, 1 < (image (fun x => x i) s).card ∧ ∀ (ai : α i), filter (fun x => x i = ai` |
| FLI1-L09 | FLI0-S12 | Finset | SUBSET_BRIDGE | high | `∃ t ⊆ s, t.card = n` |
| FLI1-L10 | FLI0-S13 | Finset | SUBSET_BRIDGE | high | `filter p s.attach = map { toFun := Subtype.map id ⋯, inj' := ⋯ } (filter (fun x ` |
| FLI1-L11 | FLI0-S14 | Finset | SUBSET_BRIDGE | high | `map f s = s` |
| FLI1-L12 | FLI0-S21 | List | MAP_FILTER_BIND_BRIDGE | high | `map (fun i => f ↑i) l.attach = map f l` |
| FLI1-L13 | FLI0-S22 | List | MAP_FILTER_BIND_BRIDGE | high | `map Subtype.val l.attach = l` |
| FLI1-L14 | FLI0-S23 | List | MAP_FILTER_BIND_BRIDGE | high | `l.bind f = l.bind g` |
| FLI1-L15 | FLI0-S24 | List | MAP_FILTER_BIND_BRIDGE | high | `l >>= f = l.bind f` |
| FLI1-L16 | FLI0-S25 | List | MAP_FILTER_BIND_BRIDGE | high | `l.bind (pure ∘ f) = map f l` |
| FLI1-L17 | FLI0-S26 | List | MAP_FILTER_BIND_BRIDGE | high | `l.bind (List.ret ∘ f) = map f l` |
| FLI1-L18 | FLI0-S27 | List | MAP_FILTER_BIND_BRIDGE | high | `count (f x) (map f l) = count x l` |
| FLI1-L19 | FLI0-S33 | List | MAP_FILTER_BIND_BRIDGE | high | `filterMap f l = filterMap g l` |
| FLI1-L20 | FLI0-S15 | Multiset | MEMBERSHIP_BRIDGE | high | `∀ x ∈ s, p (f x)` |
| FLI1-L21 | FLI0-S16 | Multiset | MEMBERSHIP_BRIDGE | high | `a ∈ s ∧ p a` |
| FLI1-L22 | FLI0-S17 | Multiset | MEMBERSHIP_BRIDGE | high | `∃ a ∈ s, f a = some b` |
| FLI1-L23 | FLI0-S18 | Multiset | MEMBERSHIP_BRIDGE | high | `a ∈ s` |
| FLI1-L24 | FLI0-S37 | Nat | IFF_SPLIT | high | `m.Coprime n` |
| FLI1-L25 | FLI0-S38 | Nat | IFF_SPLIT | high | `m.Coprime n` |
| FLI1-L26 | FLI0-S39 | Nat | IFF_SPLIT | high | `m.Coprime n` |
| FLI1-L27 | FLI0-S40 | Nat | IFF_SPLIT | high | `m.Coprime n` |
| FLI1-L28 | FLI0-S19 | Set | SUBSET_BRIDGE | high | `f '' (s \ t) = f '' s \ f '' t` |
| FLI1-L29 | FLI0-S20 | Set | SUBSET_BRIDGE | high | `f '' (s ∩ t) = f '' s ∩ f '' t` |
| FLI1-L30 | FLI0-S05 | Finset | SINGLETON_CHARACTERIZATION | low | `Subsingleton { x // x ∈ s }` |
| FLI1-L31 | FLI0-S06 | Finset | SINGLETON_CHARACTERIZATION | low | `∀ {a b : α}, a ∈ s → b ∈ s → a = b` |
| FLI1-L32 | FLI0-S07 | Finset | SINGLETON_CHARACTERIZATION | low | `({a} ∩ s).card.le 0` |
| FLI1-L33 | FLI0-S35 | Set | EXTENSIONALITY_NEEDED | medium | `x ∈ s → x ∉ t` |
| FLI1-L34 | FLI0-S36 | Set | SINGLETON_CHARACTERIZATION | low | `⋂₀ (s \ {univ}) = ⋂₀ s` |
| FLI1-L35 | FLI0-S28 | List | INDUCTION_GENERALIZATION | medium | `(l₁ ++ l₂).dedup = l₁ ∪ l₂.dedup` |
| FLI1-L36 | FLI0-S29 | List | INDUCTION_GENERALIZATION | medium | `(a :: a :: as).dedup = (a :: as).dedup` |
| FLI1-L37 | FLI0-S30 | List | INDUCTION_GENERALIZATION | medium | `(a :: l).dedup = l.dedup` |
| FLI1-L38 | FLI0-S31 | List | INDUCTION_GENERALIZATION | medium | `(a :: l).dedup = a :: l.dedup` |
| FLI1-L39 | FLI0-S32 | List | INDUCTION_GENERALIZATION | medium | `(a :: l).dedup = a :: l.dedup` |
| FLI1-L40 | FLI0-S34 | List | INDUCTION_GENERALIZATION | medium | `(l₁ ++ l₂).getLast ⋯ = l₂.getLast h` |
