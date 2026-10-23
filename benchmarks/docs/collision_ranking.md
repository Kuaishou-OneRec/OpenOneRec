# Optional collision-aware PID ranking

A generated SID can map to several catalog items. After selecting one PID, ranking
it with the whole SID's log probability ignores that ambiguity. This optional
strategy retains the empirical share of the selected PID:

```
ranking_score = raw_logprob + weight * log(selected_count / total_group_count)
```

Counts are `count_after_downsample`, summed over **unique PIDs** within the SID.
The PID with highest downsampled count is selected, with smallest PID breaking
ties. At weight 1 this is an empirical conditional-probability factor; other
weights are calibration parameters. The mapping frequencies are not guaranteed
to be user-conditional probabilities. This does not remove SID collisions.

No model parameters, codebooks, or generated candidates are changed. Raw
`logprobs` are preserved. Invalid or unmapped SIDs go last; equal ranking scores
retain input order. Exact repeated PID records are deduplicated; conflicting
counts, negative/nonfinite counts, and all-zero groups raise errors.
Existing selection and PID-resolution defaults remain unchanged.

## Run on saved generations

From the repository root, with benchmark dependencies installed:

```python
import json
from pathlib import Path
from benchmark.tasks.v1_0.recommendation.evaluator import RecommendationEvaluator

saved = json.loads(Path("results/OneRec-1.7B/ad/test_generated.json").read_text())
config = {
    "name": "ad",
    "evaluation_config": {
        "k_values": [1, 32],
        "evaluation_mode": "both",
        "select_k": "pid_probability",
        "pid_probability_weight": 1.0,
        "sid_to_pid_strategy": "most_popular_after_downsampling_deterministic",
    },
}
evaluator = RecommendationEvaluator(
    samples=saved["samples"], task_config=config,
    data_dir="raw_data/onerec_data/benchmark_data", overwrite=True,
)
metrics, per_sample_metrics = evaluator.evaluate()
print(metrics)
```

Run with `PYTHONPATH=benchmarks python your_script.py`. Saved generations must
contain one finite raw logprob per candidate. Set weight to 0 for the same
valid-candidate, deterministic-resolver baseline. Both SID and PID metrics use
the same reordered list. Changing the ranking settings invalidates cached
metrics for this strategy. In an application, use
`collision_ranking.rank_pid_candidates` to obtain indices, canonical PIDs, raw
logprobs and separate ranking scores directly.

## Local quality study (not an official leaderboard result)

A frozen OneRec-1.7B generated one shared beam-128 pool for 2,048 new advertising
users. Every method returned 32 candidates and used the **same deterministic PID
resolver**. Weights/codebooks stayed frozen. The cohort excluded 6,144 users
previously inspected in local experiments. The previously inspected 2,048 users
were explicitly reused as 1,024 adaptation-training / 1,024 development users;
this is not the unchanged zero-shot benchmark protocol.

Development selected weight 2 from a fixed 16-configuration count/smoothing/weight
grid. The selected settings were frozen before final outcomes. A separate
9-feature residual calibrator was the study's primary method; it did **not** show
a clear gain. The collision method was a prespecified secondary contrast, with
Bonferroni correction for the two fixed Recall comparisons.

| Same-pool method | PID Recall@32 |
| --- | ---: |
| Valid-only raw score, deterministic resolver | 7.3394% |
| Empirical PID share, weight 1 | 7.6426% |
| Empirical PID share, development-selected weight 2 | 7.7642% |
| Previously frozen learned combined-feature baseline | 7.5367% |
| Demote every SID with multiple unique PIDs | 7.2559% |
| Train-only ad PID count sensitivity, weight 2, Laplace 1 | 7.7258% |

The selected method changes Recall by **+0.4248 percentage points (+5.79%
relative)**. A 10,000-resample paired-user bootstrap gives a 97.5% interval of
**[+0.0601, +0.8076] percentage points** for this fixed contrast. Recall improved
for 46 users, declined for 29, and was unchanged for the rest. Distinct predicted
PID coverage changed from 1,598 to 1,566.

This sample does not establish superiority over weight 1 or the learned combined
baseline; those difference intervals include zero. SID Recall and custom PID
NDCG have positive point differences but inconclusive intervals. This is not a
claim of universal or online CTR/ROAS/LTV improvement.

The released mapping's count window was not established from its documentation.
The train-only sensitivity above instead uses positive ad PID counts from
161,818 training users, disjoint from all official advertising test UIDs, with
fixed smoothing/weights chosen before final analysis. It is auxiliary evidence,
not a replacement selected after seeing final outcomes. Supply only counts
available at recommendation time in a deployment.

The released catalog contains 1,841 exact repeated PID records across 1,840 SIDs;
these must not be interpreted as distinct items. An initial row-sum prototype
was withdrawn and corrected **before** final evaluation. The original and
corrected development point metrics happened to be identical in this cohort.

The actual evaluator was cross-checked against the research implementation for
every final user, including PID Recall/Pass and SID Recall. The patch also adapts
Recall to the ordered target lists already returned by the parsers, preserving
first-target order. This contract repair is not counted as a model-quality gain.

Runtime: one RTX 4090, BF16, vLLM 0.7.3 V0 Transformers fallback, batch 2,
2,048-token context. These results are limited to one model/task/cohort/backend.

## Validation

```bash
PYTHONPATH=benchmarks python -m unittest discover -s benchmarks/tests -v
```

Tests cover group versus item mass, duplicate catalog records, deterministic
resolution, raw-score preservation, weight-zero parity, input validation, full
SID/PID evaluator integration, cache invalidation, and ordered-target Recall.

Collision-aware evaluation and item conditioning have prior work, including
[CCE/ZCR](https://arxiv.org/abs/2605.25330) and
[EPIC](https://arxiv.org/abs/2609.03522). This is an optional benchmark integration,
not a claim to a new tokenizer or posterior-inference framework.
