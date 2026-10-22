"""Exercise the real worker entry point with model/Ray dependencies replaced."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/ray-vllm'))



class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BeamParams(Params):
    pass


def load_worker():
    package = ModuleType('benchmark')
    package.__path__ = []
    modules = {
        'ray': ModuleType('ray'),
        'vllm': SimpleNamespace(LLM=object, SamplingParams=Params),
        'vllm.sampling_params': SimpleNamespace(BeamSearchParams=BeamParams),
        'benchmark': package,
        'benchmark.base_generator': SimpleNamespace(
            Generator=type('Generator', (), {}), RayMixin=type('RayMixin', (), {}),
            VllmMixin=type('VllmMixin', (), {}), DISABLE_OPTIMIZATIONS_FOR_TASKS=[]),
        'benchmark.checkpoint_utils': SimpleNamespace(export_pt_to_safetensor=None),
        'benchmark.compact_beam_search': SimpleNamespace(compact_beam_search=lambda *args: None),
        'benchmark.console': ModuleType('benchmark.console'),
    }
    path = Path(__file__).resolve().parents[1] / 'scripts/ray-vllm/utils/generator.py'
    spec = importlib.util.spec_from_file_location('catalog_worker_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class CompactWorkerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_worker()
        self.worker = self.module.VllmWorker.__new__(self.module.VllmWorker)
        self.worker.worker_id = 0
        self.worker.tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: [99],
            decode=lambda tokens, **kwargs: ','.join(map(str, tokens)),
        )
        self.worker.llm = SimpleNamespace(
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_logprobs=40)),
            beam_search=lambda prompts, params: [SimpleNamespace(sequences=[
                SimpleNamespace(tokens=[99, 7, 8, 9], cum_logprob=-.9)])],
        )

    def test_default_native_beam_path_is_unchanged(self):
        result = self.worker.generate_batch({'a': 'prompt'},
            {'use_beam_search': True, 'beam_width': 1, 'max_tokens': 3})
        self.assertEqual(result[0], {'a': ['7,8,9']})
        self.assertEqual(result[1], {'a': [-.9]})

    def test_opt_in_compact_is_routed_and_decoded(self):
        output = [SimpleNamespace(sequences=[SimpleNamespace(tokens=[99, 1, 2, 3], cum_logprob=-.3)])]
        with patch.object(self.module, 'compact_beam_search', return_value=output) as search:
            result = self.worker.generate_batch({'a': 'prompt'},
                {'use_beam_search': True, 'beam_width': 1, 'max_tokens': 3, 'use_compact_beam_search': True})
        self.assertEqual(result[0], {'a': ['1,2,3']})
        self.assertEqual(result[1], {'a': [-.3]})
        self.assertEqual(search.call_args.args[1], [{'prompt': 'prompt'}])

    def test_sampling_mode_rejects_compact_flag(self):
        with self.assertRaises(ValueError):
            self.worker.generate_batch({'a': 'prompt'},
                {'use_beam_search': False, 'use_compact_beam_search': True})

    def test_compact_engine_failure_does_not_become_empty_success(self):
        with patch.object(self.module, 'compact_beam_search', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                self.worker.generate_batch({'a': 'prompt'},
                    {'use_beam_search': True, 'beam_width': 1, 'max_tokens': 3, 'use_compact_beam_search': True})


if __name__ == '__main__':
    unittest.main()
