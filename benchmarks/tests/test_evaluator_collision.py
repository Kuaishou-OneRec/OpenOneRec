"""Run with the isolated prospective benchmark snapshot on PYTHONPATH."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.tasks.v1_0.recommendation.evaluator import RecommendationEvaluator


class CollisionEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.gens = ['<s_a_1><s_b_2><s_c_3>', '<s_a_1><s_b_2><s_c_4>']
        k = 8192**2+2*8192
        mapping = {str(k+3): [{'pid': 4, 'count_after_downsample': 1}, {'pid': 2, 'count_after_downsample': 1}],
                   str(k+4): [{'pid': 8, 'count_after_downsample': 100}]}
        Path(self.tmp.name, 'sid2pid.json').write_text(json.dumps(mapping))
        self.samples = {'user': {'generations': self.gens, 'logprobs': [-1., -1.1],
                                 'ground_truth': '<|sid_begin|>'+self.gens[1]+'<|sid_end|>', 'metadata': {'answer_pid': [8]}}}
        self.config = {'name': 'ad', 'evaluation_config': {'k_values': [1], 'evaluation_mode': 'both',
                       'select_k': 'pid_probability', 'pid_probability_weight': 1.,
                       'sid_to_pid_strategy': 'most_popular_after_downsampling_deterministic'}}

    def evaluator(self, config=None, cached=None):
        return RecommendationEvaluator(self.samples, task_config=config or self.config,
                                       data_dir=self.tmp.name, cached_metrics=cached)

    def test_same_list_in_sid_and_pid_modes_preserves_raw_scores(self):
        untouched = deepcopy(self.samples)
        metrics, per_user = self.evaluator().evaluate()
        self.assertEqual(metrics['pid_recall@1'], 1.)
        self.assertEqual(metrics['recall@1'], 1.)
        self.assertEqual(per_user['user']['pid_generations'][0], 8)
        self.assertEqual(self.samples, untouched)
        self.assertEqual(metrics['pid_probability_weight'], 1.)

    def test_zero_weight_and_default_selection_keep_original_ranking(self):
        config = deepcopy(self.config)
        config['evaluation_config']['pid_probability_weight'] = 0.
        metrics, _ = self.evaluator(config).evaluate()
        self.assertEqual(metrics['pid_recall@1'], 0.)
        config['evaluation_config']['select_k'] = 'first_k'
        metrics, _ = self.evaluator(config).evaluate()
        self.assertEqual(metrics['pid_recall@1'], 0.)

    def test_incompatible_resolver_or_sid_only_rejected(self):
        for field, value in [('sid_to_pid_strategy', 'random'), ('evaluation_mode', 'sid')]:
            config = deepcopy(self.config)
            config['evaluation_config'][field] = value
            with self.assertRaises(ValueError):
                self.evaluator(config).evaluate()

    def test_cache_invalidates_on_ranking_change(self):
        original, _ = self.evaluator().evaluate()
        self.assertTrue(self.evaluator(cached=original)._has_all_required_metrics())
        for field, value in [('pid_probability_weight', 2.), ('select_k', 'first_k'),
                             ('sid_to_pid_strategy', 'random')]:
            config = deepcopy(self.config)
            config['evaluation_config'][field] = value
            self.assertFalse(self.evaluator(config, original)._has_all_required_metrics())
        legacy = {k: v for k,v in original.items() if k != 'select_k_strategy'}
        self.assertFalse(self.evaluator(cached=legacy)._has_all_required_metrics())

    def test_alignment_errors_fail_instead_of_using_wrong_scores(self):
        self.samples['user']['logprobs'] = [-1.]
        with self.assertRaises(ValueError):
            self.evaluator().evaluate()


if __name__ == '__main__':
    unittest.main()
