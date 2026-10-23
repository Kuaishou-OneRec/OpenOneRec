"""Optional PID ranking that retains uncertainty from SID-to-PID collisions.

The mapping frequencies are an empirical approximation, not guaranteed
user-conditional probabilities. Weight one applies log p(SID|history) plus
log empirical_share(PID|SID). Larger weights are calibration hyperparameters.
The highest downsampled-count PID is retained; ties use the smallest PID.
"""
from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

SID_PATTERN = re.compile(r'(?:<\|sid_begin\|>)?<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>(?:<\|sid_end\|>)?')


@dataclass(frozen=True)
class RankedPID:
    index: int
    pid: int
    raw_logprob: float
    ranking_score: float
    pid_share: float


def resolve_pid(infos: Sequence[Mapping]) -> tuple[int, float]:
    """Return deterministic canonical PID and its downsampled-count share."""
    if not infos:
        return 0, 0.
    unique = {}
    for info in infos:
        pid = int(info.get('pid', info.get('iid', 0)))
        if pid in unique and unique[pid]['count_after_downsample'] != info['count_after_downsample']:
            raise ValueError('Conflicting counts for a repeated PID')
        unique[pid] = info
    infos = list(unique.values())
    counts = [float(info['count_after_downsample']) for info in infos]
    if any(not math.isfinite(count) or count < 0 for count in counts):
        raise ValueError('Mapping counts must be finite and nonnegative')
    pids = [int(info.get('pid', info.get('iid', 0))) for info in infos]
    if any(pid <= 0 for pid in pids):
        raise ValueError('Each mapped PID must be positive')
    maximum = max(counts)
    pid = min(pid for pid, count in zip(pids, counts) if count == maximum)
    total = math.fsum(counts)
    if total == 0:
        raise ValueError('PID share is undefined for an all-zero count mapping')
    return pid, maximum/total


def rank_pid_candidates(generations, logprobs, catalog, weight=1.):
    """Return valid candidates sorted by a separate score; leave inputs intact.

    Catalog keys may be encoded SID integers or their JSON string equivalents.
    Malformed and unmapped generations return PID zero at the end. Equal scores
    retain the original candidate order. This function does not use labels,
    add candidates, expand collisions, or update model parameters.
    """
    if len(generations) != len(logprobs):
        raise ValueError('One raw logprob per generation is required')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Weight must be finite and nonnegative')
    results = []
    for index, (generation, raw) in enumerate(zip(generations, logprobs)):
        raw = float(raw)
        if not math.isfinite(raw):
            raise ValueError('Raw model logprobs must be finite')
        match = SID_PATTERN.fullmatch(generation.strip())
        infos = []
        if match:
            codes = tuple(map(int, match.groups()))
            if all(0 <= code < 8192 for code in codes):
                encoded = codes[0]*8192**2+codes[1]*8192+codes[2]
                infos = catalog.get(encoded, catalog.get(str(encoded), []))
        pid, share = resolve_pid(infos)
        score = raw+weight*math.log(share) if pid else -math.inf
        results.append(RankedPID(index, pid, raw, score, share))
    return sorted(results, key=lambda candidate: -candidate.ranking_score)
