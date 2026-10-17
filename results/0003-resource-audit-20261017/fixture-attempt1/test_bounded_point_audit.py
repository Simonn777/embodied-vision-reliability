import io
import struct
import unittest
from evr.bounded_point_audit import ReadBudget,npy_header,prefix,scalar,first_frame_points

def array(shape,values,descr='<f4',fortran=False):
    h=repr({'descr':descr,'fortran_order':fortran,'shape':shape}).encode()+b'\n'
    fmt={'<f4':'<f','<f2':'<e','|u1':'<B'}.get(descr,'<f')
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(h))+h+b''.join(struct.pack(fmt,x)for x in values)

def parsed(data):
    stream=io.BytesIO(data);budget=ReadBudget();h=npy_header(stream,budget);raw,_=prefix(stream,h,budget);return raw,h,budget

class BoundedPointAuditTests(unittest.TestCase):
    def test_three_frame_prefix_does_not_read_full_array(self):
        data=array((1000000,2),range(6));raw,h,b=parsed(data);self.assertEqual(len(raw),24);self.assertEqual(scalar(raw,h,2,1),5);self.assertEqual(b.returned_bytes,h['header_bytes']+24)
    def test_f16_and_visibility_uint8(self):
        raw,h,_=parsed(array((3,1),[1,2,3],'<f2'));self.assertEqual(scalar(raw,h,2,0),3)
        raw,h,_=parsed(array((3,1),[1,0,2],'|u1'));self.assertEqual(scalar(raw,h,1,0),0)
    def test_objects_structured_fortran_and_zero_dimensions_rejected(self):
        for descr,fortran,shape in [('|O',False,(3,1)),([('x','<f4')],False,(3,1)),('<f4',True,(3,1)),('<f4',False,(3,0)),('<f4',False,())]:
            with self.assertRaises(ValueError):npy_header(io.BytesIO(array(shape,[],descr,fortran)),ReadBudget())
    def test_header_cap_and_truncation(self):
        with self.assertRaises(ValueError):npy_header(io.BytesIO(b'\x93NUMPY\x02\x00'+struct.pack('<I',65537)),ReadBudget())
        with self.assertRaises(ValueError):parsed(array((3,2),[1]))
    def test_budget_prechecks_including_discarded_reads(self):
        b=ReadBudget(20);stream=io.BytesIO(b'a'*30);self.assertEqual(b.read(stream,12),b'a'*12)
        with self.assertRaises(ValueError):b.read(stream,9)
        self.assertEqual(stream.tell(),12)
    def test_stream_internal_buffer_reserve_counted(self):
        b=ReadBudget(65540);b.open_stream();self.assertEqual(b.charged_bytes,65536);b.read(io.BytesIO(b'abcd'),4)
        with self.assertRaises(ValueError):b.open_stream()
    def test_unsupported_version_and_bad_dtype(self):
        with self.assertRaises(ValueError):npy_header(io.BytesIO(b'\x93NUMPY\x09\x00'),ReadBudget())
        with self.assertRaises(ValueError):npy_header(io.BytesIO(array((3,1),[],'<c8')),ReadBudget())
    def test_frame0_selection_never_uses_future_visibility_or_error(self):
        xy,xh,_=parsed(array((3,3,2),[1]*18));xyz,gh,_=parsed(array((3,3,3),[1]*27));vis,vh,_=parsed(array((3,3),[0,1,1,1,0,0,1,0,0],'|u1'))
        self.assertEqual(first_frame_points(xy,xh,xyz,gh,vis,vh),[1,2])
    def test_nonfinite_first_frame_no_future_replacement(self):
        xy,xh,_=parsed(array((3,2,2),[float('nan'),0,1,1,0,0,0,0,0,0,0,0]));xyz,gh,_=parsed(array((3,2,3),[1]*18));vis,vh,_=parsed(array((3,2),[1]*6,'|u1'))
        self.assertEqual(first_frame_points(xy,xh,xyz,gh,vis,vh),[1])
    def test_prefix_limit_is_fixed(self):
        s=io.BytesIO(array((4,1),[1,2,3,4]));b=ReadBudget();h=npy_header(s,b)
        with self.assertRaises(ValueError):prefix(s,h,b,4)
if __name__=='__main__':unittest.main()
