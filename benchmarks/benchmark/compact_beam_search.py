# SPDX-License-Identifier: Apache-2.0
# State-transition semantics follow vLLM v0.7.3 LLM.beam_search.
"""Suffix-only state for the vLLM 0.7.3 beam search used by this benchmark.

The model, logprob budget, ranking, EOS handling and stable tie order are kept.
Long prompt lists are built only for active model requests and final outputs,
instead of being copied into every temporary candidate. Full text is decoded
lazily: the benchmark consumes continuation tokens and cumulative logprobs only.
"""
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class _Beam:
    suffix: tuple
    cum_logprob: float
    logprobs: tuple


class _Sequence:
    finish_reason = None
    stop_reason = None
    multi_modal_data = None
    mm_processor_kwargs = None

    def __init__(self, prompt, beam, tokenizer):
        self._prompt = prompt
        self._suffix = beam.suffix
        self.cum_logprob = beam.cum_logprob
        self.logprobs = list(beam.logprobs)
        self._tokenizer = tokenizer
        self._text = None

    @property
    def tokens(self):
        return list(self._prompt) + list(self._suffix)

    @property
    def text(self):
        if self._text is None:
            self._text = self._tokenizer.decode(self.tokens)
        return self._text


def compact_beam_search(llm, prompts, params):
    """Return native-compatible sequence fields without eager prompt copies.

    Text and token prompts are supported; multimodal payloads are explicitly
    rejected. Sampling uses exactly the native 2*beam_width logprob budget.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    width = params.beam_width
    if width < 1 or params.max_tokens < 0:
        raise ValueError('Beam width must be positive and max_tokens nonnegative')
    roots = []
    for prompt in prompts:
        if 'multi_modal_data' in prompt or 'mm_processor_kwargs' in prompt:
            raise ValueError('Compact benchmark beam search supports text inputs only')
        tokens = (prompt['prompt_token_ids'] if 'prompt_token_ids' in prompt
                  else tokenizer.encode(prompt['prompt']))
        if not tokens:
            raise ValueError('A beam-search prompt must contain at least one token')
        roots.append(tuple(tokens))
    active = [[_Beam((), 0.0, ())] for _ in roots]
    completed = [[] for _ in roots]
    sampling = SamplingParams(logprobs=2 * width, max_tokens=1, temperature=params.temperature)

    def ranking(beam, root):
        length = len(root) + len(beam.suffix)
        last = beam.suffix[-1] if beam.suffix else root[-1]
        if last == tokenizer.eos_token_id:
            length -= 1
        return beam.cum_logprob / (length ** params.length_penalty)

    for _ in range(params.max_tokens):
        requests, owners = [], []
        for owner, beams in enumerate(active):
            for beam in beams:
                requests.append({'prompt_token_ids': list(roots[owner]) + list(beam.suffix)})
                owners.append((owner, beam))
        if not requests:
            break
        outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError('Incomplete model response during compact beam search')
        expanded = [[] for _ in roots]
        for (owner, parent), output in zip(owners, outputs):
            # Native v0.7.3 skips a completed/aborted row with logprobs=None.
            if output.outputs[0].logprobs is None:
                continue
            position = output.outputs[0].logprobs[0]
            for token, value in position.items():
                beam = _Beam(parent.suffix + (token,), parent.cum_logprob + value.logprob,
                             parent.logprobs + (position,))
                if token == tokenizer.eos_token_id and not params.ignore_eos:
                    completed[owner].append(beam)
                else:
                    expanded[owner].append(beam)
        # Keep Python's stable sort and native insertion order for boundary ties.
        active = [sorted(beams, key=lambda beam: ranking(beam, roots[owner]), reverse=True)[:width]
                  for owner, beams in enumerate(expanded)]
    results = []
    for owner, root in enumerate(roots):
        candidates = completed[owner] + active[owner]
        candidates = sorted(candidates, key=lambda beam: ranking(beam, root), reverse=True)[:width]
        results.append(SimpleNamespace(sequences=[_Sequence(root, beam, tokenizer) for beam in candidates]))
    return results
