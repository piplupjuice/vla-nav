# Full test-matrix analysis (recomputed from results/raw/*/episodes.jsonl)

`SPL_proxy` = mean of `success * d_straight / max(path, d_straight)`; the denominator ignores walls, so it is an OPTIMISTIC bound, not true SPL.

## Coverage

- conditions present: ['A_in_dist', 'B_env_shift', 'C_instr_shift', 'D_recovery']
- all four protocol conditions present

| condition | regime | n | SR | Wilson 95% CI | SPL_proxy | inf/ep | replans/ep | lat mean s | collisions/ep | d_final m |
|---|---|---|---|---|---|---|---|---|---|---|
| A_in_dist | B1 chunked-frozen | 16 | 0.00 | [0.00,0.19] | 0.000 | 6.9 | 0.0 | 4.38 | 29.2 | 1.83 |
| A_in_dist | B2 fixed-frequency (official) | 16 | 0.00 | [0.00,0.19] | 0.000 | 41.2 | 0.0 | 4.42 | 30.0 | 1.65 |
| A_in_dist | B3 reliability-triggered | 16 | 0.00 | [0.00,0.19] | 0.000 | 16.8 | 15.4 | 4.42 | 30.0 | 1.65 |
| B_env_shift | B1 chunked-frozen | 16 | 0.06 | [0.01,0.28] | 0.000 | 7.6 | 0.0 | 4.44 | 21.0 | 2.14 |
| B_env_shift | B2 fixed-frequency (official) | 16 | 0.06 | [0.01,0.28] | 0.000 | 42.2 | 0.0 | 4.41 | 27.8 | 2.01 |
| B_env_shift | B3 reliability-triggered | 16 | 0.06 | [0.01,0.28] | 0.000 | 16.4 | 14.2 | 4.43 | 26.7 | 2.25 |
| C_instr_shift | B1 chunked-frozen | 16 | 0.06 | [0.01,0.28] | 0.062 | 5.9 | 0.0 | 3.91 | 27.6 | 1.73 |
| C_instr_shift | B2 fixed-frequency (official) | 16 | 0.00 | [0.00,0.19] | 0.000 | 40.1 | 0.0 | 3.99 | 30.0 | 1.87 |
| C_instr_shift | B3 reliability-triggered | 16 | 0.06 | [0.01,0.28] | 0.062 | 15.4 | 14.1 | 4.12 | 27.8 | 1.83 |
| D_recovery | B1 chunked-frozen | 16 | 0.00 | [0.00,0.19] | 0.000 | 6.4 | 0.0 | 4.11 | 30.0 | 2.55 |
| D_recovery | B2 fixed-frequency (official) | 16 | 0.00 | [0.00,0.19] | 0.000 | 42.1 | 0.0 | 4.04 | 29.6 | 2.46 |
| D_recovery | B3 reliability-triggered | 16 | 0.00 | [0.00,0.19] | 0.000 | 17.1 | 15.3 | 4.04 | 29.9 | 2.62 |

## Pooled across conditions

| regime | n | SR | Wilson 95% CI | SPL_proxy | inf/ep |
|---|---|---|---|---|---|
| B1 chunked-frozen | 64 | 0.03 | [0.01,0.11] | 0.016 | 6.7 |
| B2 fixed-frequency (official) | 64 | 0.02 | [0.00,0.08] | 0.000 | 41.4 |
| B3 reliability-triggered | 64 | 0.03 | [0.01,0.11] | 0.016 | 16.4 |

## Paired comparisons (same episodes; exact McNemar)

| condition | A vs B | n pairs | only A | only B | McNemar p | delta SR (A-B) | 95% bootstrap CI | inf ratio A/B |
|---|---|---|---|---|---|---|---|---|
| A_in_dist | RELIABILITY vs FIXED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 0.41 |
| A_in_dist | RELIABILITY vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 2.44 |
| A_in_dist | FIXED vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 6.00 |
| B_env_shift | RELIABILITY vs FIXED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 0.39 |
| B_env_shift | RELIABILITY vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 2.17 |
| B_env_shift | FIXED vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 5.59 |
| C_instr_shift | RELIABILITY vs FIXED | 16 | 1 | 0 | 1.000 | +0.06 | [+0.00,+0.19] | 0.38 |
| C_instr_shift | RELIABILITY vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 2.62 |
| C_instr_shift | FIXED vs CHUNKED | 16 | 0 | 1 | 1.000 | -0.06 | [-0.19,+0.00] | 6.82 |
| D_recovery | RELIABILITY vs FIXED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 0.41 |
| D_recovery | RELIABILITY vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 2.68 |
| D_recovery | FIXED vs CHUNKED | 16 | 0 | 0 | 1.000 | +0.00 | [+0.00,+0.00] | 6.61 |
