# FLI0 source artifact inventory

**FLI0 source: `BOTH`** — RC5V3 analysis layer / final report missing is expected and non-fatal; raw per-theorem results drive failure extraction. RC5V3 = PARTIAL.

### RC5V2 — `COMPLETE` (final report: yes, raw failure data: yes)

| artifact | status | records | note |
|---|---|---|---|
| eval_batch | PRESENT | 240 |  |
| rc2_baseline_results | PRESENT | 240 |  |
| rc4_static_results | PRESENT | 240 |  |
| dynamic_eligible | PRESENT | 149 |  |
| retrieval_results | PRESENT | 149 |  |
| safe_dynamic_plan | PRESENT | 149 |  |
| dynamic_b5_results | PRESENT | 149 |  |
| attribution | PRESENT |  |  |
| system_comparison | PRESENT |  |  |
| safety_audit | PRESENT |  |  |
| exported_examples | PRESENT | 745 |  |
| final_report | PRESENT |  |  |

### RC5V3 — `PARTIAL_ARTIFACTS_AVAILABLE` (final report: no, raw failure data: yes)

| artifact | status | records | note |
|---|---|---|---|
| eval_batch | PRESENT | 600 |  |
| rc2_baseline_results | PRESENT | 600 |  |
| rc4_static_results | PRESENT | 600 |  |
| dynamic_eligible | PRESENT | 318 |  |
| retrieval_results | PRESENT | 318 |  |
| safe_dynamic_plan | PRESENT | 318 |  |
| dynamic_b1_results | PARTIAL | 318 | 210/318 records have setup_error (infra) |
| dynamic_b3_results | PARTIAL | 315 | 315/315 records have setup_error (infra) |
| dynamic_b5_results | PARTIAL | 315 | 211/315 records have setup_error (infra) |
| attribution | MISSING |  |  |
| system_comparison | MISSING |  |  |
| cost_curve | MISSING |  |  |
| namespace_feature_yield | MISSING |  |  |
| safety_audit | MISSING |  |  |
| maintenance_decision | MISSING |  |  |
| exported_examples | MISSING |  |  |
| final_report | MISSING |  |  |

