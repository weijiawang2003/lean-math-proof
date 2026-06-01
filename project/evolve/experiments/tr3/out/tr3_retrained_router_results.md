# TR3 retrained router (EXPLORATORY)

- component sizes: TR1=57, SF5=20, TR3=92
- **exploratory only — not production routing**

| combo | n | labels | LOO macroF1 | LOO acc | top3 | grouped LONO F1 |
|---|---|---|---|---|---|---|
| TR1 | 57 | 7 | 0.484 | 0.702 | 0.912 | 0.137 |
| TR1+SF5 | 77 | 10 | 0.463 | 0.649 | 0.883 | 0.083 |
| TR1+SF5+TR3 | 169 | 12 | 0.567 | 0.781 | 0.888 | 0.057 |
