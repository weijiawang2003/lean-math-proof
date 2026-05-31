# TR4 training results

- examples 4737 | success pos 23 | credit pos 22

## OOF classification (leakage-free, group=theorem)

| model | PR-AUC | ROC-AUC | prec@10 | rec@20 | rec@50 | credit PR-AUC |
|---|---|---|---|---|---|---|
| heuristic | 0.0169 | 0.4392 | 0.1 | 0.0435 | 0.0435 | 0.0174 |
| logistic | 0.1038 | 0.8128 | 0.1 | 0.1304 | 0.3478 | 0.1075 |
| sgd | 0.0141 | 0.7449 | 0.0 | 0.0 | 0.0 | 0.0137 |
| hgb | 0.5216 | 0.7745 | 0.9 | 0.5217 | 0.5652 | 0.5435 |

## Grouped generalization PR-AUC

| model | by_theorem | by_namespace | by_cluster |
|---|---|---|---|
| logistic | 0.1038 | 0.0175 | 0.3364 |
| sgd | 0.0138 | 0.0085 | 0.0081 |
| hgb | 0.5216 | 0.0079 | 0.3357 |
