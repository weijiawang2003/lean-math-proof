# TR6 optional retrain (EXPLORATORY — TR4 model not replaced)

| set | n | pos | pos-ns | PR(thm) | PR(ns) | PR(cl) | top5 |
|---|---|---|---|---|---|---|---|
| tr4_only | 4737 | 23 | 4 | 0.5216 | 0.0079 | 0.4772 | 1.0 |
| tr4_plus_tr5 | 5542 | 36 | 4 | 0.2995 | 0.0055 | 0.516 | 0.7692 |
| tr4_plus_tr6 | 7195 | 44 | 7 | 0.5189 | 0.0095 | 0.0139 | 0.8529 |
| tr4_plus_tr5_tr6 | 8000 | 57 | 7 | 0.3718 | 0.01 | 0.2814 | 0.9118 |

- Δ PR-AUC by-theorem (TR4→TR4+TR6): **-0.0027**
- Δ PR-AUC **by-namespace** (TR4→TR4+TR6): **0.0016** (the key generalization metric)
