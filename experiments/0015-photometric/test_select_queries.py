"""Artificial checks for geometric query allocation and initialization."""
import unittest
import numpy as np
from select_queries import select_and_initialize

class SelectionTests(unittest.TestCase):
    def test_affine_recovery_and_unique_cells(self):
        grid=np.array([(x,y) for x in range(20,140,4) for y in range(20,140,4)],float)
        A=np.array([[1.1,.15],[-.08,.9]])
        target=np.einsum('ni,ji->nj',grid,A,optimize=False)+[4,-2]
        rows=select_and_initialize(grid,target,np.ones(len(grid),bool),160,160)
        self.assertLessEqual(len(rows),256)
        self.assertEqual(len({r['cell'] for r in rows}),len(rows))
        self.assertTrue(all(r['status']=='initialized' for r in rows))
        for r in rows: np.testing.assert_allclose(r['A0'],A,atol=1e-12,rtol=0)
    def test_collinear_geometry_retains_failures(self):
        grid=np.column_stack((np.arange(30,100,2),np.full(35,50.)))
        rows=select_and_initialize(grid,grid,np.ones(len(grid),bool),160,160)
        self.assertTrue(rows)
        self.assertTrue(all(r['status']=='initialization_failed' for r in rows))
    def test_mask_and_image_extent(self):
        x=np.array([[40.,40.],[80.,80.]])
        self.assertEqual(select_and_initialize(x,x,np.zeros(2,bool),160,160),[])
        with self.assertRaises(ValueError): select_and_initialize(x,x,np.ones(2,int),160,160)
        with self.assertRaises(ValueError): select_and_initialize(x+200,x,np.ones(2,bool),160,160)

if __name__=='__main__': unittest.main()
