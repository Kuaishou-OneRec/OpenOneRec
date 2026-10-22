"""Exact-output, paired timing comparison on saved recommendation prompts."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path


def load_implementation(path):
    spec = importlib.util.spec_from_file_location('compact_benchmark_impl', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compact_beam_search


PATTERN = re.compile(r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>')


def quality(texts, sample, mapping, k):
    truth_sid = set(PATTERN.findall(sample['ground_truth'].split('</think>')[-1]))
    raw = sample['metadata'].get('answer_pid')
    if raw is None:
        raw = sample['metadata'].get('answer_iid', [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    truth_pid = {int(value) for value in raw if value is not None and int(value) != 0}
    if not truth_sid or not truth_pid:
        raise ValueError('Empty truth requires an explicit evaluation policy')
    pids, sids = [], []
    for text in texts[:k]:
        matches = PATTERN.findall(text)
        sid = matches[0] if len(matches) == 1 else None
        sids.append(sid)
        key = str(sum(value * multiplier for value, multiplier in
                      zip(map(int, sid), [8192 ** 2, 8192, 1]))) if sid else None
        items = mapping.get(key, [])
        if items:
            maximum = max(item['count_after_downsample'] for item in items)
            pid = min(int(item.get('pid', item.get('iid', 0))) for item in items
                      if item['count_after_downsample'] == maximum)
        else:
            pid = 0
        pids.append(pid)
    seen, dcg = set(), 0.0
    for rank, pid in enumerate(pids, 1):
        if pid in truth_pid and pid not in seen:
            dcg += 1 / math.log2(rank + 1)
        seen.add(pid)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(truth_pid)) + 1))
    return {'sid_recall': len(set(sids) & truth_sid) / len(truth_sid),
            'pid_recall': len(set(pids) & truth_pid) / len(truth_pid),
            'pid_pass': float(bool(set(pids) & truth_pid)), 'pid_ndcg_custom': dcg / ideal}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--mapping', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--beam-width', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--split', choices=['development', 'test'], default='development')
    a = p.parse_args()
    if a.batch_size < 1 or a.beam_width < 1:
        p.error('Batch size and beam width must be positive')
    os.environ['VLLM_USE_V1'] = '0'
    impl = Path(__file__).resolve().parents[2] / 'benchmark/compact_beam_search.py'
    compact = load_implementation(impl)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    for path in [Path(__file__), impl]:
        shutil.copyfile(path, out / 'code' / path.name)
    source = json.loads(Path(a.source).read_text())['samples']
    items = sorted(source.items(), key=lambda item: hashlib.sha256(
        f"{a.seed}:{item[1]['metadata']['uid']}".encode()).digest())
    if not 0 < a.samples <= len(items):
        p.error('Invalid sample count')
    items = items[:a.samples]
    uids = [int(sample['metadata']['uid']) for _, sample in items]
    assert len(uids) == len(set(uids))
    protocol = {**vars(a), 'selection': 'fixed UID hash, independent of labels',
                'uids_sha256': hashlib.sha256(json.dumps(uids).encode()).hexdigest(),
                'invariants': 'same model, BF16, V0 engine, beam, 2B logprob budget and context',
                'cache': 'prefix caching enabled, reset before each method',
                'measurement': 'generation plus worker-style continuation decode and length extraction',
                'quality': 'strict token/order/cumulative-logprob equality required before computing shared metrics',
                'pid_tie': 'minimum PID among maximum count_after_downsample',
                'code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (out / 'code').iterdir()}}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams
    import numpy as np
    import torch

    llm = LLM(model=a.model, dtype='bfloat16', trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=2048, max_logprobs=2 * a.beam_width, gpu_memory_utilization=.6,
              enable_prefix_caching=True, enable_chunked_prefill=False, seed=a.seed)
    tokenizer = llm.get_tokenizer()
    params = BeamSearchParams(beam_width=a.beam_width, max_tokens=3)
    mapping = json.loads(Path(a.mapping).read_text())
    times = {'native': [], 'compact': []}
    counter = {'model_calls': 0, 'scored_prefixes': 0, 'returned_logprob_entries': 0}
    counts = {mode: {key: 0 for key in counter} for mode in times}
    original_generate = llm.generate
    def tracked_generate(*args, **kwargs):
        inputs = args[0] if args else kwargs['prompts']
        counter['model_calls'] += 1
        counter['scored_prefixes'] += len(inputs)
        outputs = original_generate(*args, **kwargs)
        counter['returned_logprob_entries'] += sum(
            len(position) for output in outputs for sequence in output.outputs
            for position in (sequence.logprobs or []) if position is not None)
        return outputs
    llm.generate = tracked_generate
    records = []
    for start in range(0, len(items), a.batch_size):
        batch = items[start:start + a.batch_size]
        ids = [list(sample['prompt_token_ids']) if 'prompt_token_ids' in sample
               else tokenizer.encode(sample['prompt'] + '<|sid_begin|>', add_special_tokens=True)
               for _, sample in batch]
        if any(len(tokens) + 3 > 2048 for tokens in ids):
            raise ValueError('Context budget exceeded')
        prompts = [{'prompt_token_ids': tokens} for tokens in ids]
        def infer(mode, inputs):
            if not llm.reset_prefix_cache():
                raise RuntimeError('Prefix cache reset failed')
            return (llm.beam_search(inputs, params) if mode == 'native'
                    else compact(llm, inputs, params))
        if start == 0:
            for mode in times:
                infer(mode, prompts[:1])
        outputs = {}
        batch_counts = {}
        order = ['native', 'compact'] if start // a.batch_size % 2 == 0 else ['compact', 'native']
        for mode in order:
            for key in counter:
                counter[key] = 0
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = infer(mode, prompts)
            observed = []
            for prompt, output in zip(ids, result):
                row = []
                for sequence in output.sequences:
                    suffix = sequence.tokens[len(prompt):]
                    row.append((suffix, float(sequence.cum_logprob),
                                tokenizer.decode(suffix, skip_special_tokens=True),
                                len(sequence.tokens) - len(prompt)))
                observed.append(row)
            torch.cuda.synchronize()
            times[mode].append(time.perf_counter() - begin)
            batch_counts[mode] = dict(counter)
            for key, value in counter.items():
                counts[mode][key] += value
            if len(result) != len(batch):
                raise RuntimeError('Incomplete model batch')
            outputs[mode] = observed
        if outputs['native'] != outputs['compact']:
            (out / 'parity_failure.json').write_text(json.dumps(outputs))
            raise RuntimeError('Candidate tokens/order/scores differ; cannot claim equivalent optimization')
        if batch_counts['native'] != batch_counts['compact']:
            raise RuntimeError('Inference budget differs despite requested identical settings')
        for index, (sample_id, sample) in enumerate(batch):
            values = outputs['native'][index]
            metrics = quality([value[2] for value in values], sample, mapping, a.beam_width)
            records.append({'sample_id': sample_id, 'uid': int(sample['metadata']['uid']),
                            'input_tokens': len(ids[index]),
                            'signature_sha256': hashlib.sha256(json.dumps(values).encode()).hexdigest(),
                            'tokens': [value[0] for value in values],
                            'logprobs': [value[1] for value in values], 'metrics': metrics})
        if len(records) % 64 == 0 or len(records) == len(items):
            (out / 'samples.json').write_text(json.dumps(records))
            (out / 'batch_times.json').write_text(json.dumps(times))
        print(f'Exact parity verified {len(records)}/{len(items)} rows', flush=True)
    native, optimized = np.array(times['native']), np.array(times['compact'])
    rng = np.random.default_rng(a.seed)
    ratios = []
    for _ in range(2000):
        indexes = rng.integers(0, len(native), len(native))
        ratios.append(1 - optimized[indexes].sum() / native[indexes].sum())
    report = {'protocol': protocol, 'samples': len(records), 'exact_candidate_parity': True,
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'torch_cuda_build': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(0),
                          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
              'input_length_tokens': {'min': min(row['input_tokens'] for row in records),
                                      'median': float(np.median([row['input_tokens'] for row in records])),
                                      'max': max(row['input_tokens'] for row in records)},
              'exact_cumulative_score_parity': True,
              'same_quality_for_both_arms': {name: float(np.mean([row['metrics'][name] for row in records]))
                                            for name in records[0]['metrics']},
              'inference': {mode: {'total_seconds': sum(values), 'batch_p50_seconds': float(np.median(values)),
                                   'batch_p95_seconds': float(np.quantile(values, .95)),
                                   'users_per_second': len(records) / sum(values), **counts[mode]}
                            for mode, values in times.items()},
              'latency_reduction': float(1 - optimized.sum() / native.sum()),
              'latency_reduction_paired_batch_ci95': np.quantile(ratios, [.025, .975]).tolist(),
              'scope': 'one GPU, paired batches, same seed; not online latency or training speedup'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
