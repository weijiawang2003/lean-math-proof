# RC5S vs RC5H comparison

- **verdict: SAFETY_HARDENING_SUCCESS**
- programs: RC5H 1792 → RC5S 828 (strict-filtered, 523 removed)
- off-policy: RC5H ~74 → RC5S **0**
- global stalls: RC5H pervasive @B10+ → RC5S **none**
- timeout handling: process-group kill, 0 bounded kills, max wall 12.3s (cap 60s)
- true wins preserved: **3/3**
- B20: RC5H unrunnable → RC5S disabled; B10: B5_ONLY

## Detail

| metric | RC5H | RC5S |
|---|---|---|
| programs | 1792 | 828 |
| off-policy | 74 | 0 |
| true wins | 3 | 3 (3 recovered) |
| global stalls | pervasive | none |
| max wall (s) | 150 (cap) | 12.3 |
| probes/true win | — | 276.0 |
