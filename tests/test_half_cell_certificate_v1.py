"""Artificial-only tests. Does not open any saved real matrix or repository file."""
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import math
import struct
import unittest

spec=importlib.util.spec_from_file_location('halfcell',Path(__file__).resolve().parents[1] / 'scripts' / 'half_cell_certificate_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def matrix_source(matrix,order='little'):
    return {'R_hex':struct.pack(('<' if order=='little' else '>')+'9f',*(x for row in matrix for x in row)).hex(),'byte_order':order}

def rational(value):
    return F(int(value['numerator']),int(value['denominator']))

class ArtificialCertificateTests(unittest.TestCase):
    def test_all_four_branches(self):
        matrices=[('w',[[1,0,0],[0,1,0],[0,0,1]]),
                  ('x',[[1,0,0],[0,-1,0],[0,0,-1]]),
                  ('y',[[-1,0,0],[0,1,0],[0,0,-1]]),
                  ('z',[[-1,0,0],[0,-1,0],[0,0,1]])]
        for branch,r in matrices:
            with self.subTest(branch=branch):
                result=m.certify(matrix_source(r))
                self.assertEqual(result['status'],'COMPATIBILITY_CERTIFIED')
                self.assertEqual(result['proposal']['branch'],branch)
                self.assertEqual(result['cells_satisfied'],9)
                self.assertEqual(rational(result['exact_SO3']['determinant']),1)
                self.assertEqual(result['proposals_attempted'],1)
    def test_ties_and_direction(self):
        for r in [[[0,-1,0],[1,0,0],[0,0,1]],[[0,0,1],[1,0,0],[0,1,0]]]:
            result=m.certify(matrix_source(r))
            self.assertEqual(result['status'],'COMPATIBILITY_CERTIFIED')
            self.assertEqual(result['proposal']['branch'],'w')
            self.assertEqual([[rational(x) for x in row] for row in result['proposal']['hypothetical_rotation']],r)
    def test_rounded_synthetic_rotation(self):
        # Artificial 45-degree z rotation, quantized once to half for the test.
        c=struct.unpack('<e',struct.pack('<e',math.sqrt(.5)))[0]
        result=m.certify(matrix_source([[c,-c,0],[c,c,0],[0,0,1]]))
        self.assertEqual(result['status'],'COMPATIBILITY_CERTIFIED')
        self.assertEqual(result['cells_satisfied'],9)
    def test_failed_proposal_is_only_inconclusive(self):
        for r in [[[1,0,0],[0,1,0],[0,0,-1]],[[1.125,0,0],[0,1.125,0],[0,0,1.125]],[[0,0,0],[0,0,0],[0,0,0]]]:
            result=m.certify(matrix_source(r))
            self.assertEqual(result['status'],'INCONCLUSIVE')
            self.assertTrue(result['exact_SO3']['verified'])
            self.assertLess(result['cells_satisfied'],9)
            self.assertEqual(result['proposals_attempted'],1)
    def test_half_grid_decoder_all_finite_patterns(self):
        for bits in range(0x7c00):
            expected=F.from_float(struct.unpack('<e',bits.to_bytes(2,'little'))[0])
            self.assertEqual(m.positive_half_value(bits),expected)
            self.assertEqual(m.positive_half_bits(expected),bits)
    def test_asymmetric_power_two_cell_and_ties(self):
        cell=m.half_cell(F(1))
        self.assertEqual(cell['lower'],1-F(1,4096))
        self.assertEqual(cell['upper'],1+F(1,2048))
        self.assertTrue(m.inside(cell,cell['lower']))
        self.assertTrue(m.inside(cell,cell['upper']))
        odd=m.half_cell(1+F(1,1024))
        self.assertFalse(m.inside(odd,odd['lower']))
        self.assertFalse(m.inside(odd,odd['upper']))
        neg=m.half_cell(F(-1))
        self.assertEqual((neg['lower'],neg['upper']),(-cell['upper'],-cell['lower']))
    def test_subnormal_normal_and_overflow_cells(self):
        cell=m.half_cell(F(1,2**24))
        self.assertEqual((cell['lower'],cell['upper']),(F(1,2**25),F(3,2**25)))
        self.assertFalse(cell['lower_closed']);self.assertFalse(cell['upper_closed'])
        normal=m.half_cell(F(1,2**14))
        self.assertTrue(normal['lower_closed'] and normal['upper_closed'])
        top=m.half_cell(F(65504))
        self.assertEqual((top['lower'],top['upper']),(65488,65520))
        self.assertFalse(top['lower_closed'] or top['upper_closed'])
        self.assertIsNone(m.half_cell(F(65505)))
        self.assertIsNone(m.half_cell(F(1,2**25)))
    def test_signed_zero_is_explicit_not_silently_equal(self):
        positive,negative=m.half_cell(F(0)),m.half_cell(F(0),True)
        self.assertTrue(m.inside(positive,F(0)))
        self.assertFalse(m.inside(negative,F(0)))
        self.assertTrue(m.inside(negative,F(-1,2**25)))
        r=[[1.,-0.,0.],[0.,1.,0.],[0.,0.,1.]]
        result=m.certify(matrix_source(r))
        self.assertEqual(result['status'],'INCONCLUSIVE')
        self.assertTrue(result['cells'][1]['negative_zero'])
        self.assertEqual(result['cells'][1]['original_f32_hex'],'00000080')
    def test_endian_replay_input_unchanged(self):
        r=[[0,-1,0],[1,0,0],[0,0,1]]
        a,b=m.certify(matrix_source(r)),m.certify(matrix_source(r,'big'))
        self.assertEqual(a['proposal'],b['proposal'])
        self.assertEqual(a['status'],b['status'])
        self.assertNotEqual(a['input']['R_hex'],b['input']['R_hex'])
    def test_invalid_and_non_half_inputs(self):
        with self.assertRaises(ValueError):m.certify({'R_hex':'0000'})
        r=[[math.inf,0,0],[0,1,0],[0,0,1]]
        with self.assertRaises(ValueError):m.certify(matrix_source(r))
        r=[[.1,0,0],[0,1,0],[0,0,1]]
        self.assertEqual(m.certify(matrix_source(r))['status'],'INCONCLUSIVE')

if __name__=='__main__':
    unittest.main(verbosity=2)
