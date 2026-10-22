# Compact beam search for short SID generation

Recommendation prompts can contain long histories while the generated SID has
only three tokens. In vLLM 0.7.3, temporary beam candidates carry complete token
lists, and final beams eagerly decode the full history into text. This benchmark
uses continuation tokens and cumulative logprobs, so those copies and full-text
decodes add host overhead without contributing to its predictions.

The optional compact implementation shares the immutable prompt, stores only a
generated suffix in each beam, and materializes full tokens at model/output
boundaries. Full text is decoded only when `.text` is requested. This changes
Python state management, not the model or search budget.

## Enable

Install the versions in the benchmark README (including vLLM 0.7.3), then add
`--use_compact_beam_search true` to a beam-search evaluation command:

```bash
cd benchmarks
VLLM_USE_V1=0 python scripts/ray-vllm/evaluate.py \
  --model_path /path/to/OneRec-1.7B \
  --data_dir /path/to/benchmark_data \
  --output_dir ./results/compact \
  --task_types ad --sample_size 256 \
  --dtype bfloat16 --worker_batch_size 4 \
  --num_beams 32 --num_return_sequences 32 \
  --max_logprobs 64 --use_compact_beam_search true
```

The default remains native vLLM beam search. In thinking mode the flag applies
only to the beam-search stage; thinking sampling is unchanged. The compact
worker raises errors instead of treating a failed batch as an empty success.

## Compatibility

Search transitions follow [vLLM v0.7.3 LLM.beam_search](https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/llm.py).
Both paths use the same `2 * beam_width` logprob budget, prompt-inclusive length
penalty, EOS handling, stable tie order, and aborted-row handling. There is no
catalog filtering, score renormalization, reduced model-call budget, or weight
update. Output token, score, logprob-history and text fields are compatible with
this benchmark's read-only consumption; this is not a replacement for vLLM's
entire mutable output API. Nonempty text/token prompts are supported; multimodal
payloads are rejected. Other vLLM versions/backends require renewed validation.

## Validate and reproduce

```bash
python -m unittest discover -s benchmarks/tests -p 'test_compact*.py' -v
```

These CPU-only tests compare against the installed native vLLM method across
324 combinations of beam width, depth, length penalty, EOS behavior and aborted
rows, including stable score ties. They check model request batches and logprob
budgets as well as output tokens, order, scores, logprob histories and text.
Worker tests exercise opt-in routing, unchanged defaults and error handling.
The reference test is skipped when vLLM is not installed; it allocates no model.

For paired GPU measurement, use an existing benchmark `test_generated.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/scripts/ray-vllm/compare_compact.py \
  --model /path/to/OneRec-1.7B \
  --source /path/to/ad/test_generated.json \
  --mapping /path/to/benchmark_data/sid2pid.json \
  --output /path/to/new-comparison-directory \
  --samples 4096 --split test --batch-size 4 --beam-width 32
```

The input has a `samples` object whose values contain `metadata.uid`,
`metadata.answer_pid` (or `answer_iid`), `ground_truth`, and either
`prompt_token_ids` including `<|sid_begin|>` or `prompt` excluding that marker.
The marker is appended for string prompts. This driver measures the three-token
SID stage, not thinking generation or the full Ray/data-loading pipeline.

Users are selected by a fixed hash of UID and seed, independent of labels.
Method order alternates by batch; prefix caches are reset before each arm.
Timing includes generation, continuation decoding and length extraction, with
CUDA synchronization. The driver aborts on any candidate/order/score/text or
model-budget mismatch. It saves source hashes, runtime, aggregate metrics,
per-batch timings and a paired batch bootstrap interval. Saved per-user files
contain evaluation data and should remain in the authorized data environment.
`pid_ndcg_custom` is a diagnostic with an explicit tie policy in the protocol,
not a replacement for the official evaluator. Exact output parity is the quality
acceptance criterion.

## Measurement scope

Development measurement on 256 held-out users, one RTX 4090, official pretrained
OneRec-1.7B, BF16, vLLM 0.7.3 V0 with its Transformers fallback, beam 32, batch 4,
three generated tokens, context limit 2048:

| Method | Sum of measured batch times | Users/second |
| --- | ---: | ---: |
| Native | 42.2774 s | 6.0552 |
| Compact | 35.8174 s | 7.1474 |

All 256 candidate lists, their order and cumulative scores matched exactly.
Measured time fell 15.28%; the paired batch bootstrap 95% interval was
5.66–23.15%. This is a single-device development result, not online P99, a
training speedup, or a recommendation-quality gain. No additional training is
required. Hardware, input length and backend affect the benefit.

A subsequent frozen confirmation used 4,096 users from the saved `ad` test
evaluation, selected by UID hash with seed 20261022 and the same configuration.
Input lengths were 203–1,242 tokens (median 722). This measured:

| Method | Sum of measured batch times | Users/second | Batch p50 | Batch p95 |
| --- | ---: | ---: | ---: | ---: |
| Native | 706.5593 s | 5.7971 | 0.6082 s | 1.0801 s |
| Compact | 595.0439 s | 6.8835 | 0.5084 s | 0.9751 s |

Time decreased **15.78%** (paired batch bootstrap 95% interval **13.62–17.88%**);
throughput increased **18.74%**. All 4,096 candidate lists, their order and
cumulative scores were exactly equal. Both methods made 3,072 model calls,
scored 266,240 prefixes and returned 17,039,360 logprob entries. Shared custom
PID Recall@32 was 6.7251%; no recommendation-quality improvement is claimed.
These are batch-stage measurements from one paired run on one device/backend.
