"""Regression for the actual baseline timing schema, using generated cohorts."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_evr0015_repair_fixtures', HERE/'test_diagnostic.py')
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)
with patch.dict(sys.modules, {'run_diagnostic': fixtures.runner}):
    repaired = fixtures.load('_evr0015_repaired_evaluator', HERE/'evaluate_saved_repair.py')

class RepairTests(fixtures.DiagnosticTests):
    def setUp(self):
        super().setUp()
        for _, original in self.originals:
            original['timing_seconds'] = {'estimator_call': original.pop('call_seconds')}
        self.saved = fixtures.evaluator
        fixtures.evaluator = repaired
        self.addCleanup(setattr, fixtures, 'evaluator', self.saved)

    def test_production_diff_is_only_report_metadata(self):
        source = (HERE/'evaluate_diagnostic.py').read_text()
        expected = source.replace('"""Evaluate all frozen 0015 predictions using only already-sealed 0013 labels."""',
            '"""Saved-output-only metadata repair of evaluate_diagnostic; predictions unchanged."""')
        expected = expected.replace("originals[i][1]['call_seconds']", "originals[i][1]['timing_seconds']['estimator_call']")
        expected = expected.replace("'schema': 'evr0015.evaluation.v1', 'source_sha256': seal['source_sha256'],",
            "'schema': 'evr0015.evaluation.v1', 'source_sha256': seal['source_sha256'],\n        'evaluation_source_sha256': runner.sha(__file__),\n        'repair': 'Read baseline timing_seconds.estimator_call; original scorer failed on absent top-level call_seconds. No prediction/metric/gate change.',")
        self.assertEqual((HERE/'evaluate_saved_repair.py').read_text(), expected)

if __name__ == '__main__':
    unittest.main()
