import importlib.util
import itertools
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'benchmark/compact_beam_search.py'
spec = importlib.util.spec_from_file_location('compact_beam_under_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams
except ImportError:
    LLM = BeamSearchParams = None


class Tokenizer:
    def __init__(self, eos=99):
        self.eos_token_id = eos
        self.decode_calls = 0
    def encode(self, text):
        return list(map(int, text.split()))
    def decode(self, tokens, **kwargs):
        self.decode_calls += 1
        return '|'.join(map(str, tokens))


class FakeLLM:
    def __init__(self, eos=99, abort=False, vocab=8, track_requests=False):
        self.vocab = vocab
        self.tokenizer = Tokenizer(eos)
        self.abort = abort
        self.budgets = []
        self.track_requests = track_requests
        self.request_batches = []
    def get_tokenizer(self):
        return self.tokenizer
    def generate(self, prompts, sampling_params, use_tqdm):
        self.budgets.append(sampling_params.logprobs)
        if self.track_requests:
            self.request_batches.append([tuple(prompt['prompt_token_ids']) for prompt in prompts])
        results = []
        for prompt in prompts:
            tokens = prompt['prompt_token_ids']
            if self.abort and len(tokens) > 3 and tokens[-1] == 5:
                results.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=None)]))
                continue
            weights = [(token, -float((token + tokens[-1]) % 4)) for token in range(self.vocab)]
            normalizer = math.log(sum(math.exp(value) for _, value in weights))
            ordered = sorted(weights, key=lambda item: item[1], reverse=True)[:sampling_params.logprobs]
            values = {token: SimpleNamespace(logprob=value - normalizer) for token, value in ordered}
            results.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[values])]))
        return results


def signature(outputs):
    return [[(sequence.tokens, sequence.cum_logprob,
              [[(key, value.logprob) for key, value in position.items()] for position in sequence.logprobs],
              sequence.text) for sequence in output.sequences] for output in outputs]


class CompactBeamTests(unittest.TestCase):
    def parameters(self, **overrides):
        return SimpleNamespace(**{'beam_width': 2, 'max_tokens': 3, 'temperature': 0.,
                                 'ignore_eos': False, 'length_penalty': 1., **overrides})

    def run_compact(self, llm, prompts, parameters):
        class Params:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        with patch.dict(sys.modules, {'vllm': SimpleNamespace(SamplingParams=Params)}):
            return module.compact_beam_search(llm, prompts, parameters)

    def test_full_text_is_lazy_but_tokens_and_scores_are_available(self):
        llm = FakeLLM()
        outputs = self.run_compact(llm, [{'prompt_token_ids': [10] * 500}], self.parameters())
        self.assertEqual(llm.tokenizer.decode_calls, 0)
        self.assertEqual(len(outputs[0].sequences[0].tokens), 503)
        text = outputs[0].sequences[0].text
        self.assertEqual(llm.tokenizer.decode_calls, 1)
        self.assertEqual(outputs[0].sequences[0].text, text)
        self.assertEqual(llm.tokenizer.decode_calls, 1)
        self.assertEqual(llm.budgets, [4, 4, 4])

    def test_text_and_token_prompts_match(self):
        a = self.run_compact(FakeLLM(), [{'prompt': '10 11'}], self.parameters())
        b = self.run_compact(FakeLLM(), [{'prompt_token_ids': [10, 11]}], self.parameters())
        self.assertEqual(signature(a), signature(b))

    def test_multimodal_and_empty_prompts_rejected(self):
        for prompt in [{'prompt_token_ids': []}, {'prompt_token_ids': [1], 'multi_modal_data': {}}]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                self.run_compact(FakeLLM(), [prompt], self.parameters())

    def test_empty_input(self):
        self.assertEqual(self.run_compact(FakeLLM(), [], self.parameters()), [])

    @unittest.skipIf(LLM is None, 'requires installed vLLM; does not construct a real model or use CUDA')
    def test_matches_actual_vllm_073_for_eos_ties_lengths_and_aborted_rows(self):
        cases = 0
        for width, depth, penalty, ignore, eos, abort in itertools.product(
                [1, 2, 4], [0, 1, 3], [0., 1., 2.], [False, True], [0, 5, 99], [False, True]):
            params = BeamSearchParams(beam_width=width, max_tokens=depth,
                                      length_penalty=penalty, ignore_eos=ignore)
            prompts = [{'prompt_token_ids': [10, 11]}, {'prompt': '12 13 14 15'}]
            native_llm = FakeLLM(eos, abort, track_requests=True)
            compact_llm = FakeLLM(eos, abort, track_requests=True)
            native = LLM.beam_search(native_llm, prompts, params)
            compact = module.compact_beam_search(compact_llm, prompts, params)
            self.assertEqual(signature(native), signature(compact),
                             (width, depth, penalty, ignore, eos, abort))
            self.assertEqual(native_llm.budgets, compact_llm.budgets)
            self.assertEqual(native_llm.request_batches, compact_llm.request_batches)
            cases += 1
        self.assertEqual(cases, 324)


if __name__ == '__main__':
    unittest.main()
