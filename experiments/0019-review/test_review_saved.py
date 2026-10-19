import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('audit0019',Path(__file__).with_name('review_saved.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class Tests(unittest.TestCase):
    def test_prefinal_discrete_and_numeric(self):
        p=r.parity({'iterations':4,'score':.5},{'iterations':4,'score':.5})
        self.assertEqual(p['bitwise_equal_float_components'],1)
        for changed in ({'iterations':4.,'score':.5},{'iterations':4,'score':.50000000001}):
            with self.assertRaises(ValueError):r.parity(changed,{'iterations':4,'score':.5})

    def test_signed_translation_and_auc_denominator(self):
        identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        e=r.u.errors({'R':identity,'t':[-2.,0.,0.]},(identity,[1.,0.,0.]))
        self.assertEqual(e['max_degrees'],180.)
        self.assertEqual(e['max_signfolded_degrees'],0.)
        self.assertAlmostEqual(r.u.auc([0.,180.],5.),.5)
        self.assertAlmostEqual(r.u.auc([5.,180.],5.),0.)

    def test_material_boundaries(self):
        self.assertTrue(r.harm(4.999,5.))
        self.assertFalse(r.harm(5.,5.1))
        self.assertTrue(r.harm(7.,8.))


if __name__=='__main__':unittest.main()
