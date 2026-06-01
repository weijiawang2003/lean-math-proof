# TR4 per-theorem ranking eval

- source: tr3 | theorems: 92 | with success: 13

## First-success rank (lower = better)

| ordering | n | mean | median |
|---|---|---|---|
| original_order | 13 | 16.46 | 8.0 |
| random_expected | 13 | 8.73 | 4.5 |
| heuristic | 13 | 5.15 | 2.0 |
| logistic | 13 | 2.69 | 2.0 |
| sgd | 13 | 8.15 | 2.0 |
| hgb | 13 | 2.08 | 2.0 |

## Top-k success recovery (frac of theorems-with-success)

| ordering | top1 | top3 | top5 |
|---|---|---|---|
| heuristic | 0.385 | 0.615 | 0.769 |
| logistic | 0.385 | 0.769 | 0.846 |
| sgd | 0.462 | 0.538 | 0.615 |
| hgb | 0.462 | 0.769 | 1.0 |
| original_order | 0.385 | 0.385 | 0.462 |
