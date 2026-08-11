from pathlib import Path

import tempfile
import unittest
from pathlib import Path


class DynACOGridRunnerTest(unittest.TestCase):
    def test_default_grid_contains_old_default_and_faco_test_only(self):
        from analysis import dynaco_grid as run_dynaco_grid

        configs = run_dynaco_grid.default_grid()

        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0].name, 'old_default')
        command = run_dynaco_grid.build_command(configs[0], smoke=False)
        self.assertIn('--val_infer_mode', command)
        self.assertIn('faco_test', command)
        self.assertIn('--select_metric_mode', command)
        self.assertIn('faco_test', command)
        self.assertNotIn('both', command)

    def test_parse_report_extracts_epoch0_final_and_best(self):
        from analysis import dynaco_grid as run_dynaco_grid

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.md'
            report.write_text(
                '| epoch | train_best_cost | train_mean_cost | faco_test_best_T |\n'
                '| --- | --- | --- | --- |\n'
                '| 0 | nan | nan | 5.900000 |\n'
                '| 1 | 12.0 | 13.0 | 5.800000 |\n'
                '| 5 | 11.0 | 12.0 | 5.850000 |\n',
                encoding='utf-8',
            )

            parsed = run_dynaco_grid.parse_report(report)

        self.assertEqual(parsed['epoch0'], 5.9)
        self.assertEqual(parsed['final_epoch'], 5)
        self.assertEqual(parsed['final'], 5.85)
        self.assertEqual(parsed['best_epoch'], 1)
        self.assertEqual(parsed['best'], 5.8)
        self.assertAlmostEqual(parsed['improvement'], 0.05)


if __name__ == '__main__':
    unittest.main()
