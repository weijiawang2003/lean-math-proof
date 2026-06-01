# RC4D full canonical floor benchmark

- config: hybrid_evolved, top-k 8, max-steps 8
- **all floors pass (RC4D ≥ RC2, no regression): True**
- total regressions: 0

| floor | n | RC2 | RC4D | delta | regressed | gained | pass |
|---|---|---|---|---|---|---|---|
| demo_v1 | 15 | 12 | 12 | 0 | 0 | 0 | True |
| nat_defs_medium | 38 | 37 | 37 | 0 | 0 | 0 | True |
| nat_defs_large_v5 | 65 | 49 | 49 | 0 | 0 | 0 | True |
