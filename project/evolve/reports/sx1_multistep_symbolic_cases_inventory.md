# SX1 multi-step symbolic-assisted case inventory

Mined offline from the existing oracle/symbolic trace corpus (`*_wx3ind_*` Multiset, `*_ax1sym_*` Option/List). No live Lean was run in this stage.

## Totals

- symbolic-action firings scanned: **812**
- single-shot closes (symbolic => ProofFinished): **31**
- advanced (symbolic => TacticState): **217** (of which 5 were later closed by the search)
- **unique multistep symbolic-assisted cases: 5** ({'AX4': 3, 'AX2': 2})

## Key finding

> The existing NS9/WX3 best-first search already explores follow-up tactics from advanced symbolic states; every multistep case below was already CLOSED by that search. Sequence mode does not add these as new wins — it makes the two-step shape explicit/learnable.

## Multistep cases

| theorem | ns | arc | first symbolic tactic | closing tactic | closer origin | raw/ns9 solved? |
|---|---|---|---|---|---|---|
| `Multiset.mem_sigma` | Multiset | AX4 | `induction s using Multiset.induction_on <;> simp_all` | `aesop` | generative_topk | False |
| `Multiset.mem_add` | Multiset | AX4 | `induction s using Multiset.induction_on <;> simp_all` | `aesop` | generative_topk | False |
| `Multiset.mem_map` | Multiset | AX4 | `induction s using Multiset.induction_on <;> simp_all` | `aesop` | generative_topk | False |
| `List.headI_dedup` | List | AX2 | `cases l <;> simp_all` | `cases l <;> simp_all` | wrapper_symbolic_action | False |
| `List.tail_dedup` | List | AX2 | `cases l <;> simp_all` | `cases l <;> simp_all` | wrapper_symbolic_action | False |

## Interpretation

Every multistep close above was produced by the existing best-first search (the closing tactic appears in the trace). The symbolic first action *advances* the state; the search then finds the closer (base-model `aesop`, a re-applied symbolic action, etc.). The SX1 sequence schema turns this implicit two-step behaviour into an explicit, namespace-gated, depth-2 object — its value is **selectivity / learnability**, not new raw search reach (see decision gate in the report).
