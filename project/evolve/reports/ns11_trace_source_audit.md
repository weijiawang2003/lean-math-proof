# NS11 trace source audit

- total traces.jsonl files: **368**
- total episodes (across files): **13773**
- total close transitions:    **11915**
- total advance transitions:  **5624**

| run group | files | episodes | close | advance | theorem sets | skel meta | origins (close+adv) |
|---|---:|---:|---:|---:|---|---|---|
| `skeleton_runs/ns5-20260523-050214-0ec613` | 171 | 6654 | 6308 | 839 | large?, nat_defs_medium | no | tactic_template:5021, generative_topk:870, fallback_tactic:763, family_tactic:367, retrieved_premise:126 |
| `ns6_runs/ns6-20260523-144409-49e91b` | 23 | 952 | 867 | 167 | large?, nat_defs_medium | no | tactic_template:696, generative_topk:163, fallback_tactic:107, family_tactic:42, retrieved_premise:26 |
| `ns7_runs/ns7-20260523-161321-709316` | 19 | 748 | 705 | 94 | large?, nat_defs_medium | yes | tactic_template:565, generative_topk:103, fallback_tactic:81, family_tactic:38, retrieved_premise:12 |
| `ns9_runs/ns8-20260523-234547-fc1277` | 17 | 724 | 665 | 152 | large?, nat_defs_medium | yes | tactic_template:535, generative_topk:139, fallback_tactic:83, family_tactic:34, retrieved_premise:26 |
| `ns8_runs/ns8-20260523-215711-3051c3` | 9 | 368 | 345 | 64 | large?, nat_defs_medium | yes | tactic_template:275, generative_topk:63, fallback_tactic:41, family_tactic:18, retrieved_premise:12 |
| `autonomous_runs/v5-auto-20260522-095802-1fcaa0` | 12 | 456 | 312 | 679 | ? | no | fallback_tactic:448, generative_topk:231, family_tactic:119, term_builder:109, retrieved_premise:81 |
| `autonomous_runs/v5-followup-20260522-103058-537f36` | 11 | 418 | 298 | 542 | ? | no | fallback_tactic:395, generative_topk:159, tactic_template:145, family_tactic:80, retrieved_premise:61 |
| `runs/evolve-20260522-072211-b7f1fc` | 9 | 342 | 208 | 657 | ? | no | fallback_tactic:548, generative_topk:173, family_tactic:80, retrieved_premise:63, tactic_template:1 |
| `autonomous_runs/v5-ns3-20260522-193323-3294ab` | 6 | 228 | 195 | 167 | ? | no | tactic_template:154, fallback_tactic:108, generative_topk:54, family_tactic:32, retrieved_premise:14 |
| `autonomous_runs/v5-wave4-20260522-111556-3063e7` | 6 | 228 | 168 | 285 | ? | no | fallback_tactic:234, generative_topk:86, tactic_template:58, family_tactic:46, retrieved_premise:29 |
| `autonomous_runs/v5-wave6-20260522-121607-e99584` | 5 | 190 | 155 | 192 | ? | no | tactic_template:130, fallback_tactic:126, generative_topk:46, family_tactic:30, retrieved_premise:15 |
| `autonomous_runs/v5-wave5-20260522-115150-c14a93` | 5 | 190 | 151 | 196 | ? | no | fallback_tactic:131, tactic_template:123, generative_topk:49, family_tactic:30, retrieved_premise:14 |
| `ns6_runs/baseline` | 2 | 102 | 86 | 36 | large?, nat_defs_medium | no | tactic_template:72, generative_topk:28, fallback_tactic:13, retrieved_premise:5, family_tactic:4 |
| `ns7_runs/baseline` | 2 | 102 | 86 | 36 | large?, nat_defs_medium | yes | tactic_template:72, generative_topk:28, fallback_tactic:13, retrieved_premise:5, family_tactic:4 |
| `ns9_runs/baseline` | 2 | 102 | 86 | 36 | large?, nat_defs_medium | yes | tactic_template:72, generative_topk:28, fallback_tactic:13, retrieved_premise:5, family_tactic:4 |
| `skeleton_runs/ns5-20260523-045640-5cb0cc` | 2 | 102 | 86 | 29 | large?, nat_defs_medium | no | tactic_template:70, fallback_tactic:22, generative_topk:15, family_tactic:4, retrieved_premise:4 |
| `autonomous_runs/v5-ns3-20260522-222000-9beeab` | 2 | 76 | 74 | 8 | ? | no | tactic_template:58, generative_topk:10, fallback_tactic:8, family_tactic:4, retrieved_premise:2 |
| `autonomous_runs/v5-ns3-20260522-195455-7f1508` | 2 | 76 | 69 | 32 | ? | no | tactic_template:52, generative_topk:19, fallback_tactic:14, family_tactic:12, retrieved_premise:4 |
| `runs/evolve-20260521-034204-247625` | 2 | 60 | 50 | 0 | ? | no | ?:50 |
| `autonomous_runs/large_v5_master` | 1 | 64 | 43 | 70 | large_v5 | no | fallback_tactic:54, tactic_template:33, generative_topk:15, family_tactic:6, retrieved_premise:5 |
| `autonomous_runs/large_v5_master_steps16` | 1 | 64 | 43 | 126 | large_v5 | no | fallback_tactic:102, tactic_template:33, generative_topk:19, family_tactic:10, retrieved_premise:5 |
| `autonomous_runs/large_v5_kitchen` | 1 | 64 | 41 | 75 | large_v5 | no | fallback_tactic:54, tactic_template:33, generative_topk:16, retrieved_premise:7, family_tactic:6 |
| `runs/evolve-20260521-093316-8d8595` | 4 | 60 | 40 | 28 | ? | no | fallback_tactic:32, family_tactic:20, generative_topk:16 |
| `runs/evolve-20260521-081120-d15c23` | 4 | 60 | 38 | 70 | ? | no | fallback_tactic:100, generative_topk:8 |
| `autonomous_runs/v5-ns3-20260522-200519-d374c5` | 1 | 38 | 37 | 4 | ? | no | tactic_template:29, generative_topk:5, fallback_tactic:4, family_tactic:2, retrieved_premise:1 |
| `runs/evolve-20260521-055157-c9f991` | 4 | 60 | 37 | 0 | ? | no | fallback_tactic:34, generative_topk:3 |
| `skeleton_runs/ns5-20260523-045212-793700` | 1 | 38 | 37 | 4 | nat_defs_medium | no | tactic_template:29, generative_topk:5, fallback_tactic:4, family_tactic:2, retrieved_premise:1 |
| `autonomous_runs/v5-ns3-20260522-200134-2c809a` | 1 | 38 | 36 | 11 | ? | no | tactic_template:27, generative_topk:9, family_tactic:6, fallback_tactic:4, retrieved_premise:1 |
| `runs/evolve-20260521-070057-0175d2` | 4 | 52 | 35 | 16 | ? | no | fallback_tactic:45, generative_topk:6 |
| `runs/evolve-20260521-045407-dbcd53` | 4 | 60 | 34 | 0 | ? | no | fallback_tactic:32, generative_topk:2 |
| `autonomous_runs/ns1_v5_27_repro` | 1 | 38 | 31 | 37 | ? | no | fallback_tactic:25, tactic_template:24, generative_topk:10, family_tactic:6, retrieved_premise:3 |
| `autonomous_runs/ns1_v5_31_iff_reorder_fixed` | 1 | 38 | 31 | 37 | ? | no | fallback_tactic:25, tactic_template:24, generative_topk:10, family_tactic:6, retrieved_premise:3 |
| `autonomous_runs/ns3_5_v5_27_sanity` | 1 | 38 | 31 | 37 | ? | no | fallback_tactic:25, tactic_template:24, generative_topk:10, family_tactic:6, retrieved_premise:3 |
| `autonomous_runs/v5_27_repro` | 1 | 38 | 31 | 37 | ? | no | fallback_tactic:25, tactic_template:24, generative_topk:10, family_tactic:6, retrieved_premise:3 |
| `runs/evolve-20260522-062048-3673c6` | 1 | 38 | 26 | 55 | ? | no | fallback_tactic:46, generative_topk:19, family_tactic:9, retrieved_premise:7 |
| `runs/evolve-20260521-182223-1f6a34` | 1 | 38 | 25 | 34 | ? | no | fallback_tactic:35, generative_topk:13, family_tactic:11 |
| `runs/evolve-20260521-184742-70ac3e` | 1 | 38 | 25 | 42 | ? | no | fallback_tactic:35, generative_topk:17, family_tactic:15 |
| `runs/evolve-20260521-233937-cf2370` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:51, generative_topk:18, family_tactic:16, retrieved_premise:6 |
| `runs/evolve-20260522-024446-58dcb7` | 1 | 38 | 25 | 47 | ? | no | fallback_tactic:35, generative_topk:17, family_tactic:15, retrieved_premise:5 |
| `runs/evolve-20260522-025521-bc3e5a` | 1 | 38 | 25 | 58 | ? | no | fallback_tactic:35, generative_topk:17, retrieved_premise:16, family_tactic:15 |
| `runs/evolve-20260522-034524-d10acf` | 1 | 38 | 25 | 42 | ? | no | fallback_tactic:35, generative_topk:17, family_tactic:15 |
| `runs/evolve-20260522-044315-1c0395` | 1 | 38 | 25 | 60 | ? | no | fallback_tactic:53, generative_topk:17, family_tactic:15 |
| `runs/evolve-20260522-050325-0fe236` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:58, generative_topk:17, family_tactic:15, tactic_template:1 |
| `runs/evolve-20260522-061049-9be813` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:58, generative_topk:17, family_tactic:15, tactic_template:1 |
| `runs/evolve-20260522-061553-122264` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:58, generative_topk:17, family_tactic:15, tactic_template:1 |
| `runs/evolve-20260522-062457-6e9f3e` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:58, generative_topk:17, family_tactic:15, tactic_template:1 |
| `runs/evolve-20260522-062940-b62ae9` | 1 | 38 | 25 | 66 | ? | no | fallback_tactic:58, generative_topk:17, family_tactic:15, tactic_template:1 |
| `runs/evolve-20260522-064654-332077` | 1 | 38 | 25 | 64 | ? | no | fallback_tactic:50, generative_topk:19, family_tactic:12, retrieved_premise:7, tactic_template:1 |
| `runs/evolve-20260521-042816-615c15` | 3 | 45 | 24 | 0 | ? | no | ?:24 |
| `runs/evolve-20260521-043356-11736a` | 3 | 45 | 24 | 0 | ? | no | fallback_tactic:24 |
| `autonomous_runs/nat_defs_subset_master` | 1 | 15 | 12 | 18 | nat_defs_subset | no | fallback_tactic:8, tactic_template:7, generative_topk:7, family_tactic:6, retrieved_premise:2 |
| `autonomous_runs/demo_v1_master` | 1 | 15 | 11 | 3 | demo_v1 | no | generative_topk:13, family_tactic:1 |
| `autonomous_runs/demo_v1_raw_baseline` | 1 | 15 | 10 | 3 | demo_v1 | no | generative_topk:13 |
| `runs/evolve-20260521-181556-250a59` | 1 | 15 | 10 | 7 | ? | no | fallback_tactic:8, family_tactic:5, generative_topk:4 |
| `autonomous_runs/large_v5_raw_baseline` | 1 | 64 | 4 | 12 | large_v5 | no | generative_topk:16 |
| `eval_runs/gen_v5_plus1_raw_medium` | 1 | 38 | 4 | 10 | nat_defs_medium | no | ?:14 |
| `autonomous_runs/nat_defs_medium_raw_baseline` | 1 | 38 | 3 | 5 | nat_defs_medium | no | generative_topk:8 |
| `eval_runs/gen_v5_raw_medium` | 1 | 38 | 3 | 5 | nat_defs_medium | no | ?:8 |
| `runs/evolve-20260521-035516-0fd6ca` | 3 | 45 | 0 | 0 | ? | no |  |
