from copy import deepcopy
import unittest

from benchmark.tasks.v1_0.recommendation.collision_ranking import rank_pid_candidates, resolve_pid


class CollisionRankingTests(unittest.TestCase):
    def setUp(self):
        self.generations = ['<s_a_1><s_b_2><s_c_3>', '<s_a_1><s_b_2><s_c_4>']
        self.key = 1*8192**2+2*8192
        self.catalog = {self.key+3: [{'pid': 4, 'count_after_downsample': 1},
                                   {'pid': 2, 'count_after_downsample': 1}],
                        self.key+4: [{'pid': 8, 'count_after_downsample': 100}]}

    def test_group_mass_is_not_single_pid_mass(self):
        before = deepcopy(self.catalog)
        ranked = rank_pid_candidates(self.generations, [-1., -1.1], self.catalog)
        self.assertEqual([r.pid for r in ranked], [8, 2])
        self.assertEqual([r.raw_logprob for r in ranked], [-1.1, -1.])
        self.assertEqual(self.catalog, before)
        self.assertEqual(ranked[1].pid_share, .5)

    def test_zero_weight_is_valid_only_raw_order(self):
        result = rank_pid_candidates(self.generations, [-1., -1.1], self.catalog, weight=0)
        self.assertEqual([r.index for r in result], [0, 1])
        self.assertEqual([r.ranking_score for r in result], [-1., -1.1])

    def test_count_scale_and_mapping_order_invariance(self):
        infos = self.catalog[self.key+3]
        self.assertEqual(resolve_pid(infos), resolve_pid(infos[::-1]))
        scaled = [{'pid': i['pid'], 'count_after_downsample': i['count_after_downsample']*100} for i in infos]
        self.assertEqual(resolve_pid(infos), resolve_pid(scaled))

    def test_unmapped_multisid_and_out_of_range_go_last(self):
        inputs = self.generations+['', self.generations[0]*2, '<s_a_8192><s_b_2><s_c_3>']
        result = rank_pid_candidates(inputs, [-1., -2., 0., 0., 0.], self.catalog)
        self.assertEqual([r.pid for r in result][-3:], [0, 0, 0])

    def test_json_catalog_keys_and_wrappers(self):
        result = rank_pid_candidates(['<|sid_begin|>'+self.generations[0]+'<|sid_end|>'], [-1.],
                                     {str(k): v for k, v in self.catalog.items()})
        self.assertEqual(result[0].pid, 2)

    def test_invalid_counts_weight_and_alignment(self):
        for count in [-1., float('nan'), float('inf'), 0.]:
            with self.assertRaises(ValueError):
                resolve_pid([{'pid': 1, 'count_after_downsample': count}])
        with self.assertRaises(ValueError):
            rank_pid_candidates(self.generations, [-1.], self.catalog)
        for weight in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                rank_pid_candidates([], [], self.catalog, weight)
        self.assertEqual(resolve_pid([{'pid': 1, 'count_after_downsample': 1}]*2), (1, 1.))
        with self.assertRaises(ValueError):
            resolve_pid([{'pid': 1, 'count_after_downsample': 1}, {'pid': 1, 'count_after_downsample': 2}])

    def test_unambiguous_catalog_and_equal_scores_preserve_order(self):
        for infos in self.catalog.values():
            del infos[1:]
        result = rank_pid_candidates(self.generations, [-1., -1.], self.catalog)
        self.assertEqual([r.index for r in result], [0, 1])
        self.assertEqual([r.ranking_score for r in result], [-1., -1.])


if __name__ == '__main__':
    unittest.main()
