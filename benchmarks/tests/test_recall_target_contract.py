"""Exercise the real ordered-parser -> Recall contract without model dependencies."""
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1]/'benchmark/tasks/v1_0/recommendation'


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecallTargetContractTests(unittest.TestCase):
    def test_pid_ordered_parser_output_reaches_recall(self):
        module = load('utils_by_pid')
        answer = [42, 7, 42, 9]
        target = module.extract_ids_from_answer(answer)
        self.assertEqual(target, [42, 7, 9])
        self.assertEqual(module.compute_recall_at_k([7, 7], target, 2), 1/3)
        self.assertEqual(module.extract_first_id_from_answer(answer), 42)
        self.assertEqual(target, [42, 7, 9])

    def test_sid_ordered_parser_output_reaches_recall(self):
        module = load('utils')
        answer = ''.join('<|sid_begin|>'+sid+'<|sid_end|>' for sid in ['z', 'a', 'z', 'b'])
        target = module.extract_ids_from_answer(answer)
        self.assertEqual(target, ['z', 'a', 'b'])
        self.assertEqual(module.compute_recall_at_k(['a', 'a'], target, 2), 1/3)
        self.assertEqual(module.extract_first_id_from_answer(answer), 'z')
        self.assertEqual(target, ['z', 'a', 'b'])

    def test_existing_set_callers_empty_and_duplicate_targets(self):
        for name in ['utils', 'utils_by_pid']:
            module = load(name)
            for target in [{1, 2}, [1, 2], (1, 2), [1, 1, 2]]:
                self.assertEqual(module.compute_recall_at_k([1, 1, 9], target, 2), .5)
            self.assertEqual(module.compute_recall_at_k([], [1], 1), 0.)
            self.assertEqual(module.compute_recall_at_k([1], [], 1), 0.)
            self.assertEqual(module.compute_recall_at_k([1], [1], 0), 0.)


if __name__ == '__main__':
    unittest.main()
