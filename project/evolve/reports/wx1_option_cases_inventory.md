# WX1 — CX3 Option cases headroom inventory

Pool: **cases_simp|Option** (CX3 headroom — theorems the routed model fails but `cases <var> <;> simp_all` closes). Total **13**, with an extractable Option/Bool context variable: **13**.

| theorem | set | raw | wrap | option vars | cx3 minimal tactic |
|---|---|:---:|:---:|---|---|
| `Option.bind_congr'` | cx3_option_cases_medium | n | n | `x, y` | `intros <;> cases y <;> simp_all` |
| `Option.bnot_isNone` | cx3_option_simp_easy | n | n | `a` | `intros <;> cases a <;> simp_all` |
| `Option.bnot_isSome` | cx3_option_simp_easy | n | n | `a` | `intros <;> cases a <;> simp_all` |
| `Option.casesOn'_eq_elim` | cx3_option_simp_easy | n | n | `a` | `intros <;> cases a <;> simp_all` |
| `Option.elim'_eq_elim` | cx3_option_simp_easy | n | n | `a` | `intros <;> cases a <;> simp_all` |
| `Option.elim_apply` | cx3_option_simp_easy | n | n | `i` | `intros <;> cases i <;> simp_all` |
| `Option.elim_comp` | cx3_option_simp_easy | n | n | `i` | `intros <;> cases i <;> simp_all` |
| `Option.eq_none_or_eq_some` | cx3_bool_option_mixed | n | n | `a` | `intros <;> cases a <;> simp_all` |
| `Option.isSome_map` | cx3_option_simp_easy | n | n | `o` | `intros <;> cases o <;> simp_all` |
| `Option.map_bind` | cx3_option_cases_medium | n | n | `x` | `intros <;> cases x <;> simp_all` |
| `Option.map_bind'` | cx3_option_cases_medium | n | n | `x` | `intros <;> cases x <;> simp_all` |
| `Option.orElse_none'` | cx3_option_simp_easy | n | n | `x` | `intros <;> cases x <;> simp_all` |
| `Option.pmap_bind` | cx3_option_cases_medium | n | n | `x` | `intros <;> cases x <;> simp_all` |

## Goal snippets

- `Option.bind_congr'` — vars `['x', 'y']` — `⊢ x.bind f = y.bind g`
- `Option.bnot_isNone` — vars `['a']` — `⊢ (!a.isNone) = a.isSome`
- `Option.bnot_isSome` — vars `['a']` — `⊢ (!a.isSome) = a.isNone`
- `Option.casesOn'_eq_elim` — vars `['a']` — `⊢ a.casesOn' b f = a.elim b f`
- `Option.elim'_eq_elim` — vars `['a']` — `⊢ Option.elim' b f a = a.elim b f`
- `Option.elim_apply` — vars `['i']` — `⊢ i.elim x f y = i.elim (x y) fun j => f j y`
- `Option.elim_comp` — vars `['i']` — `⊢ (i.elim (h x) fun j => h (f j)) = h (i.elim x f)`
- `Option.eq_none_or_eq_some` — vars `['a']` — `⊢ a = none ∨ ∃ x, a = some x`
- `Option.isSome_map` — vars `['o']` — `⊢ (Option.map f o).isSome = o.isSome`
- `Option.map_bind` — vars `['x']` — `⊢ Option.map f (x >>= g) = do`
- `Option.map_bind'` — vars `['x']` — `⊢ Option.map f (x.bind g) = x.bind fun a => Option.map f (g a)`
- `Option.orElse_none'` — vars `['x']` — `⊢ (x.orElse fun x => none) = x`
- `Option.pmap_bind` — vars `['x']` — `⊢ pmap f (x >>= g) H = do`
