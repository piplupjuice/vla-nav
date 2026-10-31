# Pilot analysis (recomputed from results/raw/episodes.jsonl)

SPL shown is the *optimistic straight-line shortest-path proxy* (walls ignored): SPL_proxy = mean over episodes of [ success * d_straight / max(path_taken, d_straight) ].

| regime | n | SR | Wilson 95% CI | SPL_proxy | inference/ep | replans/ep | lat mean s | collisions/ep | dist_final mean m |
|---|---|---|---|---|---|---|---|---|---|
| chunked_frozen | 10 | 0.00 | [0.00,0.28] | 0.000 | 4.0 | 0.0 | 3.81 | 30.0 | 1.52 |
| fixed_frequency | 10 | 0.00 | [0.00,0.28] | 0.000 | 30.0 | 0.0 | 4.10 | 30.0 | 1.52 |
| reliability_triggered | 10 | 0.00 | [0.00,0.28] | 0.000 | 15.0 | 14.0 | 4.10 | 30.0 | 1.52 |

provenance (env_family/policy_is_model in rows): ['ai2thor/model=True']
