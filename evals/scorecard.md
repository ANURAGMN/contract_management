# Scorecard

- Runs per contract: **1**  -  Sample: 15 issues + 4 decoys
- Result: HARD FAIL

| Contract | High recall (min) | Overall (min) | Obligations | Security | Precision (min) | Grounding (mean) | Route |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contract_a | 0/0 | 0/2 | 0/0 | 0/0 | 0.0 | 1.0 | MISS ['legal'] (want low_risk_queue) |
| contract_b | 1/2 | 3/4 | 0/0 | 0/0 | 0.6 | 1.0 | OK |
| contract_c | 0/0 | 2/4 | 0/0 | 0/0 | 0.5 | 1.0 | MISS ['manual_review'] (want legal) |
| contract_d | 0/1 | 2/3 | 0/0 | 0/0 | 1.0 | 1.0 | MISS ['manual_review'] (want legal) |
| contract_e | 0/0 | 2/2 | 0/2 | 1/1 | 0.5 | 1.0 | MISS ['legal'] (want procurement_owner) |

## Hard fails
- contract_b: high-severity issue missed (1/2)
- contract_d: high-severity issue missed (0/1)