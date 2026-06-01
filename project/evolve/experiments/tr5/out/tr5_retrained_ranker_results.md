# TR5 optional retrain (EXPLORATORY — TR4 model not replaced)

| set | n | pos | PR-AUC(thm) | ROC(thm) | PR-AUC(ns) | PR-AUC(cluster) | top5 rec |
|---|---|---|---|---|---|---|---|
| tr4_only | 4737 | 23 | 0.5216 | 0.7745 | 0.0079 | 0.4772 | 1.0 |
| tr4_plus_tr5 | 5542 | 36 | 0.2995 | 0.7096 | 0.0055 | 0.516 | 0.7692 |

- Δ PR-AUC (by theorem), TR4→TR4+TR5: **-0.2221**
