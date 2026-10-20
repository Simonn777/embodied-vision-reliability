"""One manufactured first-call check of the existing pinned 3-point API."""
import hashlib
import importlib
import os
from pathlib import Path
import unittest
import numpy as np
from controls import solve_points_native


class NativeApiTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('EVR0027_NATIVE_RUNTIME'), 'local pinned native ABI requires explicit runtime')
    def test_exact_affine_depth_pose_is_among_native_solutions(self):
        runtime=Path(os.environ['EVR0027_NATIVE_RUNTIME']).resolve()
        library=runtime/'poselib/_core.cpython-39-darwin.so'
        expected='3228d488ca11f966378b6b7657ee6ebc5716da474c7ae88fccff7ab433a6981d'
        self.assertEqual(hashlib.sha256(library.read_bytes()).hexdigest(),expected)
        native=importlib.import_module('poselib')
        module=importlib.import_module('poselib._core')
        self.assertEqual(Path(module.__file__).resolve(),library)
        X=np.array([[-.5,1./3.,3.],[.8,-.4,5.],[.3,.9,4.]])
        R=np.array([[2.,-2.,1.],[2.,1.,-2.],[1.,2.,2.]])/3.
        t=np.array([1./6.,2./3.,1./3.])
        Y=X@R.T+t
        rays1=X/X[:,2,None];rays2=Y/Y[:,2,None]
        d1=X[:,2]-.2;d2=(Y[:,2]+.1)/1.3
        result=solve_points_native(rays1,rays2,d1,d2,native)
        self.assertEqual(result['status'],'completed')
        self.assertGreater(result['native_returned'],0)
        matches=[]
        for c in result['candidates']:
            if not c['pose_usable']:continue
            matches.append(max(np.max(np.abs(np.asarray(c['R'])-R)),
                np.max(np.abs(np.asarray(c['t'])-t)),abs(c['b']-.2),abs(c['a']-1.3),abs(c['c']+.1)))
        self.assertTrue(matches)
        self.assertLess(min(matches),1e-7)
        self.assertEqual(hashlib.sha256(library.read_bytes()).hexdigest(),expected)


if __name__=='__main__':
    unittest.main()
