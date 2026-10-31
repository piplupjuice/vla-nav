# Pilot analysis (recomputed from results/raw/episodes.jsonl)

SPL shown is the *optimistic straight-line shortest-path proxy* (walls ignored): SPL_proxy = mean over episodes of [ success * d_straight / max(path_taken, d_straight) ].

| regime | n | SR | Wilson 95% CI | SPL_proxy | inference/ep | replans/ep | lat mean s | collisions/ep | dist_final mean m |
|---|---|---|---|---|---|---|---|---|---|
| chunked_frozen | 10 | 0.00 | [0.00,0.28] | 0.000 | 5.2 | 0.0 | 4.51 | 23.5 | 1.76 |
| fixed_frequency | 10 | 0.10 | [0.02,0.40] | 0.100 | 35.8 | 0.0 | 4.63 | 23.5 | 1.85 |
| reliability_triggered | 10 | 0.00 | [0.00,0.28] | 0.000 | 13.7 | 12.1 | 4.63 | 22.6 | 1.75 |

provenance (env_family/policy_is_model in rows): ['ai2thor/model=True']
