# v5 autonomous research final report — v5-auto-20260522-095802-1fcaa0

- theorem set: nat_defs_medium
- total runtime: 0.86h
- cycles: 12

## scoreboard
| # | variant | dir | proved | Δ | prog | err | tb attempts/adv/won | newly proved |
|---|---------|-----|--------|----|------|-----|---------------------|--------------|
| 1 | v5-00-baseline-repro | baseline | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 2 | v5-01-div-hyp-pos | B | 26 | +0 | 6 | 6 | 0/0/0 | — |
| 3 | v5-02-mul-family | B | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 4 | v5-03-split-ifs | B | 26 | +0 | 5 | 7 | 0/0/0 | — |
| 5 | v5-04-term-iff-basic | A | 26 | +0 | 5 | 7 | 122/16/14 | — |
| 6 | v5-05-term-iff-adv | A | 26 | +0 | 5 | 7 | 138/16/14 | — |
| 7 | v5-06-term-iff-hyp | A | 26 | +0 | 5 | 7 | 54/14/14 | — |
| 8 | v5-07-term-dvd | A | 26 | +0 | 5 | 7 | 57/19/0 | — |
| 9 | v5-08-pow-sqrt | B | 26 | +0 | 3 | 9 | 0/0/0 | — |
| 10 | v5-09-skeleton-mut | C | 26 | +0 | 5 | 7 | 102/16/14 | — |
| 11 | v5-10-combo-minimal | B | 26 | +0 | 6 | 6 | 122/14/14 | — |
| 12 | v5-11-combo-aggressive | B | 26 | +0 | 4 | 8 | 47/14/14 | — |

## best candidate

- name: `v5-01-div-hyp-pos`
- direction: B
- proved: **26**  (Δ +0)
- description: Restore v45 {hyp_pos} div templates
