"""Synthetic hand-computed C/F, endian, access-budget and geometry fixtures only."""
import io
import itertools
import math
import struct
import unittest
import zipfile
from evr.bounded_point_audit import ReadBudget
from evr.point_geometry_v2 import (header,offset,read_prefix,inspect_zip,load_geometry,
    selected_points,project_v2,check_geometry,Prefix,metadata_json,info_npz_headers,
    audit_budget,zip_envelope)


def npy(shape,descr='<f4',order='C',fn=None):
    text=repr({'descr':descr,'fortran_order':order=='F','shape':shape}).encode()
    raw=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(text))+text
    if descr.endswith('O'):return raw+b'FORBIDDEN PICKLE PAYLOAD'
    fmt={'f4':'f','f8':'d','b1':'B','u1':'B'}[descr[-2:]]
    axes=list(range(len(shape))) if order=='C' else list(reversed(range(len(shape))))
    payload=[]
    for physical in itertools.product(*(range(shape[i]) for i in axes)):
        ix=[0]*len(shape)
        for axis,v in zip(axes,physical):ix[axis]=v
        value=(fn or (lambda x:100*x[0]+10*x[1]+(x[2] if len(x)>2 else 0)))(ix)
        payload.append(struct.pack(('>' if descr.startswith('>') else '<')+fmt,value))
    return raw+b''.join(payload)


def stream(raw,budget):
    f=io.BytesIO(raw);h=header(f,budget);return f,h


def packed(entries):
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in entries.items():z.writestr(name,raw)
    f.seek(0);return f


def fixture():
    entries={
      'trajs_2d.npy':npy((4,2,2),fn=lambda x:x[1]*2+x[2]),
      'trajs_3d.npy':npy((4,2,3),fn=lambda x:(x[1]*2 if x[2]==0 else x[1]*2+1 if x[2]==1 else 1)),
      'visibs.npy':npy((4,2),'|b1','F',lambda x:1),
      'valids.npy':npy((4,2),'|b1','F',lambda x:1),
      'intrinsics.npy':npy((4,3,3),fn=lambda x:int(x[1]==x[2])),
      'extrinsics.npy':npy((4,4,4),fn=lambda x:int(x[1]==x[2]))}
    return entries


class StreamingTests(unittest.TestCase):
    def test_asymmetric_c_f_endian(self):
        for order in ['C','F']:
            for dtype in ['<f4','>f4','<f8','>f8']:
                b=ReadBudget();f,h=stream(npy((4,2,2),dtype,order),b)
                a,log=read_prefix(f,h,b)
                self.assertEqual(a.shape,[3,2,2])
                for t,p,c in itertools.product(range(3),range(2),range(2)):
                    self.assertEqual(a.at(t,p,c),100*t+10*p+c)
                self.assertEqual(log['numeric_stream_span_bytes'],(15 if order=='F' else 12)*h['itemsize'])

    def test_f_bool_exact_and_one_byte_short(self):
        raw=npy((4,2),'|b1','F',lambda x:x[0]%2)
        b=ReadBudget();f,h=stream(raw,b);before=b.returned_bytes
        a,log=read_prefix(f,h,b)
        self.assertEqual(offset(h,(2,1)),6);self.assertEqual(log['numeric_stream_span_bytes'],7)
        self.assertEqual(log['discarded_bytes'],1)
        self.assertEqual(a.values,[0,0,1,1,0,0])
        for limit,ok in [(before+7,True),(before+6,False)]:
            b=ReadBudget(limit);f,h=stream(raw,b)
            if ok:read_prefix(f,h,b);self.assertEqual(b.charged_bytes,limit)
            else:
                with self.assertRaisesRegex(ValueError,'physical span'):read_prefix(f,h,b)
                self.assertEqual(b.returned_bytes,before)

    def test_fortran_large_span_is_rejected_before_read(self):
        b=ReadBudget(65536)
        h={'shape':[849,11721,3],'itemsize':4,'order':'F','format':'<f'}
        f=io.BytesIO(b'not enough')
        with self.assertRaisesRegex(ValueError,'physical span'):read_prefix(f,h,b)
        self.assertEqual(f.tell(),0);self.assertEqual(b.returned_bytes,0)

    def test_carried_budget_and_stream_open(self):
        b=audit_budget();self.assertEqual(b.charged_bytes,590976)
        b.open_stream();self.assertEqual(b.charged_bytes,656512)
        b=ReadBudget(65536);b.open_stream()
        with self.assertRaises(ValueError):b.read(io.BytesIO(b'x'),1)

    def test_truncated_numeric_and_indices(self):
        raw=npy((4,2,2));b=ReadBudget();f,h=stream(raw[:-40],b)
        with self.assertRaisesRegex(ValueError,'truncated'):read_prefix(f,h,b)
        for ix in [(True,0,0),(-1,0,0),(4,0,0),(0,0)]:
            with self.assertRaises(ValueError):offset(h,ix)

    def test_bad_header_shape_dtype_and_order(self):
        for data in [b'not numpy',npy((4,2,2)).replace(b"'fortran_order': False",b"'fortran_order': 1    "),npy((4,2),'|O')]:
            with self.assertRaises((ValueError,struct.error)):header(io.BytesIO(data),ReadBudget())
        raw=npy((4,2,2));raw=raw.replace(b'(4, 2, 2)',b'(0, 2, 2)')
        with self.assertRaises(ValueError):header(io.BytesIO(raw),ReadBudget())

    def test_npz_schema_roundtrip(self):
        b=ReadBudget();f=packed(fixture());zip_envelope(f,b)
        with zipfile.ZipFile(f)as z:hs,data,ledger=load_geometry(z,b,[4,2])
        self.assertEqual(selected_points(data),[0,1]);self.assertEqual(ledger['visibs']['numeric_stream_span_bytes'],7)
        data['visibs'].values[5]=2  # exact projection, source-occluded: not visible GT support
        result={};check_geometry(hs,data,result)
        self.assertEqual(len(result['point_checks']),6);self.assertEqual(result['max_error_original_px'],0)
        self.assertEqual(sum(r['projection_consistent'] for r in result['point_checks']),6)
        self.assertEqual(sum(r['valid_visible_GT_support'] for r in result['point_checks']),5)

    def test_schema_fail_before_numeric(self):
        b=ReadBudget();progress={}
        with zipfile.ZipFile(packed(fixture()))as z:
            with self.assertRaisesRegex(ValueError,'T/N'):load_geometry(z,b,[849,11721],progress)
        self.assertEqual(progress,{})

    def test_zip_safety_size_and_envelope(self):
        for name in ['../a.npy','a//b.npy','a\x00.npy','a/b.npy']:
            if '\x00' in name:continue  # ZipFile itself truncates NUL names; direct safe_member tested elsewhere.
            with zipfile.ZipFile(packed({name:npy((4,2,2))})) as z:
                with self.assertRaises(ValueError):inspect_zip(z,ReadBudget())
        f=packed({'a.npy':npy((4,2,2))});b=ReadBudget();r=zip_envelope(f,b)
        self.assertEqual(b.returned_bytes,22)
        self.assertEqual(b.stream_reserve_bytes,r['central_directory_bytes']+22)
        self.assertEqual(b.charged_bytes,44+r['central_directory_bytes'])
        f.seek(0)
        with self.assertRaises(ValueError):zip_envelope(f,ReadBudget(b.charged_bytes-1))
        f=packed({'a.npy':npy((4,2,2))+b'extra'})
        with zipfile.ZipFile(f)as z:
            with self.assertRaises(ValueError):inspect_zip(z,ReadBudget())
        for raw in [b'',b'not zip'*10]:
            with self.assertRaises(ValueError):zip_envelope(io.BytesIO(raw),ReadBudget())
        f=packed({'a.npy':npy((4,2,2))})
        with zipfile.ZipFile(f,'a')as z:z.comment=b'forbidden'
        with self.assertRaises(ValueError):zip_envelope(f,ReadBudget())

    def test_zero_one_queries_and_future_not_selection(self):
        data={'trajs_2d':Prefix([3,3,2],[0.]*18),'trajs_3d':Prefix([3,3,3],[1.]*27),'visibs':Prefix([3,3],[0]*9)}
        self.assertEqual(selected_points(data),[])
        data['visibs'].values[1]=1
        self.assertEqual(selected_points(data),[1])
        data['visibs'].values[3:]=[1]*6
        self.assertEqual(selected_points(data),[1])
        data['visibs'].values[0]=1;data['trajs_2d'].values[0]=math.nan
        self.assertEqual(selected_points(data),[1])


class GeometryTests(unittest.TestCase):
    K=[[2.,0.,5.],[0.,3.,7.],[0.,0.,1.]]
    RT=[[1.,0.,0.,1.],[0.,1.,0.,2.],[0.,0.,1.,3.],[0.,0.,0.,1.]]
    def test_unique_convention_translation(self):
        xy,gate=project_v2([1.,2.,1.],self.K,self.RT,'f4')
        self.assertEqual(xy,[6.,10.]);self.assertEqual(gate['rotation_tolerance'],9.5367431640625e-7)
        self.assertEqual(project_v2([1.,2.,1.],self.K,self.RT,'f8')[1]['rotation_tolerance'],1e-8)

    def test_float32_rounded_rotation_vs_old_threshold(self):
        c=struct.unpack('<f',struct.pack('<f',math.sqrt(.5)))[0]
        rt=[[c,-c,0,0],[c,c,0,0],[0,0,1,0],[0,0,0,1]]
        xy,log=project_v2([0.,0.,1.],self.K,rt,'f4')
        self.assertGreater(log['RtR_max'],1e-8);self.assertLess(log['RtR_max'],9.5367431640625e-7)
        self.assertAlmostEqual(log['RtR_max'],3.422854177870249e-8)
        with self.assertRaises(ValueError):project_v2([0.,0.,1.],self.K,rt,'f8')

    def test_shear_scale_reflection_and_overflow(self):
        for rt in [[[1.0001,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                   [[1,.0001,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                   [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]:
            with self.assertRaises(ValueError):project_v2([0.,0.,1.],self.K,rt,'f4')
        k=[[1e308,0,0],[0,1,0],[0,0,1]]
        with self.assertRaises(ValueError):project_v2([1e308,0,1],k,self.RT,'f4')

    def test_exact_half_pixel_is_pass(self):
        with zipfile.ZipFile(packed(fixture()))as z:hs,data,_=load_geometry(z,ReadBudget())
        data['trajs_2d'].values[0]=.5;result={};check_geometry(hs,data,result)
        self.assertEqual(result['max_error_original_px'],.5)

    def test_exact_bottom_rows_depth_focal_and_no_replacement(self):
        for world,k,rt in [([1,2,-4],self.K,self.RT),([1,2,1],[[0,0,0],[0,1,0],[0,0,1]],self.RT),
                            ([1,2,1],self.K,self.RT[:3]+[[0,0,1e-12,1]])]:
            with self.assertRaises(ValueError):project_v2(world,k,rt,'f4')
        with zipfile.ZipFile(packed(fixture()))as z:hs,data,_=load_geometry(z,ReadBudget())
        data['trajs_2d'].values[0]=.500001;result={}
        with self.assertRaises(ValueError):check_geometry(hs,data,result)
        self.assertEqual(result['selected_original_point_indices'],[0,1]);self.assertEqual(len(result['point_checks']),1)
        self.assertEqual(result['point_checks'][0]['status'],'failed')
        self.assertFalse(result['point_checks'][0]['valid_visible_GT_support'])
        self.assertGreater(result['max_error_original_px'],.5)
        self.assertEqual(result['planned_point_checks'],6)
        self.assertEqual(result['attempted_point_checks'],1)
        self.assertEqual(result['unattempted_point_checks'],5)
        self.assertEqual(result['rms_error_original_px'],result['max_error_original_px'])
        with zipfile.ZipFile(packed(fixture()))as z:hs,data,_=load_geometry(z,ReadBudget())
        data['valids'].values[0]=0;result={}
        with self.assertRaises(ValueError):check_geometry(hs,data,result)
        self.assertEqual(result['selected_original_point_indices'],[0,1])
        self.assertEqual(result['attempted_point_checks'],1)
        self.assertEqual(result['unattempted_point_checks'],5)
        self.assertIsNone(result['max_error_original_px'])
        self.assertIsNone(result['point_checks'][0]['projection_consistent'])
        self.assertFalse(result['point_checks'][0]['valid_visible_GT_support'])


class MetadataTests(unittest.TestCase):
    def test_metadata_unknown_and_nonfinite(self):
        r=metadata_json(b'{"assets":["a"],"camera":"x"}')
        self.assertEqual(r['coverage']['asset']['matching_key_count'],1)
        self.assertIn('UNKNOWN',r['coverage']['asset']['relationship'])
        for raw in [b'{"x":NaN}',b'{"x":1e999}',b'{"a":1,"a":2}',b'[]',b'x'*16385]:
            with self.assertRaises(ValueError):metadata_json(raw)

    def test_object_scalar_info_header(self):
        raw=packed({'info.npy':npy((),'|O')}).getvalue()
        result=info_npz_headers(raw,ReadBudget())
        self.assertEqual(result[0]['header']['shape'],[])
        self.assertEqual(result[0]['header']['object_payload'],'FORBIDDEN_NOT_DECODED')

    def test_object_info_header_never_pickle(self):
        raw=packed({'info.npy':npy((1,),'|O')}).getvalue();b=ReadBudget()
        r=info_npz_headers(raw,b)
        self.assertEqual(r[0]['header']['object_payload'],'FORBIDDEN_NOT_DECODED')
        self.assertFalse(r[0]['numeric_payload_interpreted'])
        self.assertFalse(r[0]['object_payload_decoded'])

class AtomicCheckpointTests(unittest.TestCase):
    def test_sigterm_during_synthetic_numeric_read_preserves_intent(self):
        import json
        import os
        from pathlib import Path
        import select
        import signal
        import subprocess
        import sys
        import tempfile
        code = r"""
import json,signal,sys,zipfile
from pathlib import Path
from evr.bounded_point_audit import ReadBudget
from evr.point_geometry_v2 import load_geometry,atomic_json
from test_point_geometry_v2 import packed,fixture
ledger={};state={'audit_complete':False,'numeric_GT_access':'NUMERIC_ACCESS_STARTING','interrupted_counter_semantics':'INCOMPLETE_LOWER_BOUNDS_OR_UNKNOWN','numeric_access':ledger}
class InterruptedBudget(ReadBudget):
    block=False
    def read(self,f,n):
        value=super().read(f,n)
        if self.block:
            print('NUMERIC_READ_RETURNED_IN_MEMORY',flush=True)
            signal.pause()  # parent kills after real synthetic bytes were returned
        return value
b=InterruptedBudget()
def checkpoint():
    atomic_json(Path(sys.argv[1]),state)
    b.block=any(v['status']=='NUMERIC_ACCESS_STARTING' for v in ledger.values())
atomic_json(Path(sys.argv[1]),state)
with zipfile.ZipFile(packed(fixture()))as z:load_geometry(z,b,[4,2],ledger,checkpoint)
"""
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'audit.json';env={**os.environ,'PYTHONPATH':'src:tests'}
            p=subprocess.Popen([sys.executable,'-c',code,str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                ready,_,_=select.select([p.stdout],[],[],10)
                self.assertTrue(ready,'synthetic child must reach payload read')
                self.assertEqual(p.stdout.readline().strip(),'NUMERIC_READ_RETURNED_IN_MEMORY')
                p.terminate();p.wait(timeout=10)
                self.assertEqual(p.returncode,-signal.SIGTERM)
                saved=json.loads(path.read_text())
                self.assertEqual(saved['numeric_GT_access'],'NUMERIC_ACCESS_STARTING')
                self.assertFalse(saved['audit_complete'])
                self.assertEqual(saved['interrupted_counter_semantics'],'INCOMPLETE_LOWER_BOUNDS_OR_UNKNOWN')
                row=saved['numeric_access']['trajs_2d']
                self.assertFalse(row['counts_complete'])
                self.assertEqual(row['status'],'NUMERIC_ACCESS_STARTING')
                print(json.dumps({'synthetic_termination_fixture':True,'observed_child_returncode':p.returncode,'saved_status':saved['numeric_GT_access'],'audit_complete':saved['audit_complete'],'counter_semantics':saved['interrupted_counter_semantics']},sort_keys=True))
            finally:
                if p.poll() is None:p.kill();p.wait(timeout=10)
                p.stdout.close();p.stderr.close()


class ReceiptBoundaryTests(unittest.TestCase):
    def test_missing_receipt_blocks_before_cache(self):
        import importlib.util
        from pathlib import Path
        import os
        import tempfile
        script=Path(__file__).resolve().parents[1]/'scripts/audit_0003_fortran_v2.py'
        spec=importlib.util.spec_from_file_location('fortran_candidate',script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                with self.assertRaises(FileNotFoundError):mod.verify_published()
                self.assertEqual(list(Path('.').iterdir()),[])
                Path('protocols').mkdir()
                Path(mod.RECEIPT).write_text('{"schema":2,"scope":"DEVELOPMENT_GEOMETRY_AUDIT_ONLY","input_commit":"'+('a'*40)+'","inputs_sha256":{}}')
                with self.assertRaises(ValueError):mod.verify_published()
            finally:os.chdir(cwd)

if __name__=='__main__':unittest.main()
