# WX2 — cases/induction catalog audit

Does the WX1 state-aware Option cases pattern generalize? Audit of fresh (unused) candidates in inductive namespaces.

| namespace | available | fresh unused | cases-friendly | buckets |
|---|---:|---:|:---:|---|
| Option | 46 | 0 | yes | {} |
| List | 260 | 165 | yes | {'list_induction': 14, 'list_cases': 151} |
| Bool | 35 | 0 | yes | {} |
| Sum | 0 | 0 | yes | {} |
| Prod | 5 | 1 | yes | {'prod_cases': 1} |
| Multiset | 260 | 251 | NO (quotient) | {'multiset_quotient_excluded': 251} |

**Verdict:** the fresh cases-friendly surface is dominated by **List** (165 fresh). Option and Bool are exhausted (0 fresh — consumed by CX3); Sum is absent; Prod is tiny; Multiset is a quotient type and excluded. WX2 generalization is therefore primarily a **List** test.
