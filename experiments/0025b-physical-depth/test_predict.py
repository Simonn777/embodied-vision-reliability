"""Artificial-only prediction contracts; no models, datasets, Torch or network."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test0025b_predict', HERE/'predict.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def artificial_encoded(color=2):
    chunk=b'IHDR'+struct.pack('>IIBBBBB',640,480,8,color,0,0,0)
    return b'\x89PNG\r\n\x1a\n'+(13).to_bytes(4,'big')+chunk+(zlib.crc32(chunk)&0xffffffff).to_bytes(4,'big')


def write_arrays(path, arrays, numpy):
    with path.open('xb') as stream:
        numpy.savez(stream, **arrays)
    return dict(filename=path.name, bytes=path.stat().st_size, sha256=p.c.sha(path),
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                                raw_c_order_sha256=p.raw_sha(v)) for k,v in arrays.items()})


class Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class FakeBackend:
    def __init__(self, fail_at=None):
        self.np, self.calls, self.fail_at = np, [], fail_at
        self.torch = NS(inference_mode=contextlib.nullcontext)
        self.moge = NS(infer=self.infer)
        self.Image = NS(open=lambda _: ImageFixture())
    def tensor(self, image): return image
    def infer(self, image, **options):
        self.calls.append(dict(shape=image.shape, dtype=str(image.dtype), options=options))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('manufactured third-call failure')
        k = np.array([[481.2/448,0,.5],[0,481.2/336,.5],[0,0,1]],dtype=np.float32)
        return {name: Tensor(value) for name,value in dict(depth=np.full((336,448),2,np.float32),
            mask=np.ones((336,448),np.bool_), points=np.zeros((336,448,3),np.float32), intrinsics=k).items()}
    def features(self, *_): raise AssertionError('forbidden features')
    def match(self, *_): raise AssertionError('forbidden matches')


class ImageFixture:
    format, size, mode = 'PNG', (640,480), 'RGB'
    def __enter__(self): return self
    def __exit__(self,*_): return False
    def getexif(self): return {}
    def __array__(self, dtype=None, copy=None):
        return np.full((480,640,3),137,dtype=np.uint8)


def fake_core():
    views = ('gs','rs_x_positive','rs_x_negative','rs_y_positive','rs_y_negative')
    def geometry(_):
        xy = np.empty((336,448,2),np.float64)
        xy[...,0],xy[...,1] = 30.25,40.5
        return dict(source_xy=xy,h=np.ones((336,448)),row_angle_radians=np.zeros((336,448)),
                    full_rgb_four_neighbor_inbounds=True)
    return NS(VIEWS=views,WIDTH=448,HEIGHT=336,READOUT_SECONDS=.04,TOTAL_ANGLE_RADIANS=np.pi/15,
        OUTPUT_K=np.array([[481.2,0,223.5],[0,481.2,167.5],[0,0,1]]),
        SOURCE_K=np.array([[481.2,0,319.5],[0,480,239.5],[0,0,1]]),make_view_geometry=geometry)


def fake_frontend(backend=None):
    def validate(raw, shape, numpy):
        if raw['depth'].shape != shape or raw['mask'].shape != shape:
            raise ValueError('prediction shape')
    q = NS(MOGE_OPTIONS=dict(num_tokens=1200,use_fp16=False,apply_mask=False,force_projection=True),
           save_arrays=write_arrays,capture_warnings=lambda _: contextlib.nullcontext())
    return NS(q=q,validate_depth=validate,Backend=mock.Mock(return_value=backend))


def base_record(n=1):
    return dict(frames=[{} for _ in range(n)],stage='started',depth_forwards_attempted=0,
                depth_forwards_completed=0,feature_forwards=0,matcher_forwards=0,
                native_calls=0,gt_values_decoded=False,offline_attempts=[])


class PredictionTests(unittest.TestCase):
    def test_float_bilinear_preserves_nonquantized_channel_value(self):
        raw = np.zeros((480,640,3),np.uint8)
        raw[41,31] = [255,128,64]
        xy = np.broadcast_to([30.25,40.5],(336,448,2)).copy()
        output = p.remap_rgb(raw,xy,np)
        expected = np.asarray([255,128,64],np.float64)/255*.125
        np.testing.assert_allclose(output[0,0],expected,rtol=2e-7,atol=0)
        self.assertEqual(output.dtype,np.float32)
        self.assertNotEqual(float(output[0,0,0])*255,round(float(output[0,0,0])*255))

    def test_rgb_neighbor_boundary_is_strict_even_when_cast_rounds(self):
        raw = np.zeros((480,640,3),np.uint8)
        xy = np.broadcast_to([639.,40.5],(336,448,2)).copy()
        with self.assertRaisesRegex(ValueError,'unobserved'):
            p.remap_rgb(raw,xy,np)
        xy[...,0] = np.nextafter(639.,-np.inf)
        self.assertEqual(p.remap_rgb(raw,xy,np).shape,(336,448,3))

    def test_one_frame_exact_five_calls_and_full_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); path = output/'rgb.png'; path.write_bytes(artificial_encoded())
            encoded = dict(path=str(path),sha256=p.c.sha(path),bytes=path.stat().st_size,member='rgb/7.png')
            backend,record = FakeBackend(),base_record()
            result = p.process_frame(dict(index=0,scene='fake'),encoded,output,backend,fake_core(),
                fake_frontend(),NS(check_intrinsics=lambda *_: None),record,lambda:None)
            self.assertEqual(len(backend.calls),5)
            self.assertEqual(record['depth_forwards_completed'],5)
            self.assertEqual(record['feature_forwards'],0)
            self.assertTrue(all(c['options']['fov_x'] > 0 and c['options']['num_tokens']==1200 for c in backend.calls))
            with np.load(output/'frame-00.npz',allow_pickle=False) as saved:
                self.assertEqual(len(saved.files),15)
                self.assertEqual(saved['view_4_depth'].shape,(336,448))
            self.assertFalse(result['gt_values_decoded'])

    def test_partial_depth_arrays_and_failed_time_survive_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); path=output/'rgb.png'; path.write_bytes(artificial_encoded())
            encoded=dict(path=str(path),sha256=p.c.sha(path),bytes=path.stat().st_size,member='rgb/3.png')
            record=base_record()
            with self.assertRaisesRegex(RuntimeError,'third-call'):
                p.process_frame(dict(index=0),encoded,output,FakeBackend(3),fake_core(),
                    fake_frontend(),NS(check_intrinsics=lambda *_:None),record,lambda:None)
            saved=json.loads((output/'frame-00.json').read_text())
            self.assertEqual((record['depth_forwards_attempted'],record['depth_forwards_completed']),(3,2))
            self.assertEqual(saved['status'],'failed')
            self.assertGreaterEqual(saved['views'][-1]['inference_seconds'],0)
            with np.load(output/'frame-00-partial.npz',allow_pickle=False) as arrays:
                self.assertEqual(len(arrays.files),6)
            self.assertFalse((output/'seal.json').exists())

    def test_encoded_hash_tamper_blocks_decode_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); path=output/'rgb.png'; path.write_bytes(b'changed')
            backend=FakeBackend(); backend.Image.open=mock.Mock(side_effect=AssertionError('must not decode'))
            encoded=dict(path=str(path),sha256='0'*64,bytes=7,member='rgb/0.png')
            with self.assertRaisesRegex(ValueError,'RGB bytes'):
                p.process_frame(dict(index=0),encoded,output,backend,fake_core(),fake_frontend(),
                                NS(check_intrinsics=lambda *_:None),base_record(),lambda:None)
            backend.Image.open.assert_not_called(); self.assertEqual(backend.calls,[])

    def test_metadata_selection_uses_numeric_intersection_and_endpoints(self):
        indices={}
        for name in p.c.ASSETS:
            members=[dict(name='%s/%s.png'%(kind,i),type='file',size=100)
                     for kind in ('rgb','depth') for i in (0,1,2,3,4,5,6,7,8,9,10)]
            members += [dict(name='rgb/999.png',type='file',size=100)]
            indices[name]=dict(members=list(reversed(members)),archive_sha256_tofu='a'*64,
                member_count=len(members),member_content_bytes=sum(m['size'] for m in members),
                expanded_bytes=1024+sum(512+((m['size']+511)//512)*512 for m in members))
        rows,populations=p.c.select_frames(indices)
        self.assertEqual(len(rows),16)
        self.assertEqual([r['numeric_stem'] for r in rows[:8]],[0,1,3,4,6,7,9,10])
        self.assertTrue(all(x['paired_count']==11 for x in populations))

    def make_seal_fixture(self, output):
        selected=[dict(index=i,scene='invented') for i in range(16)]
        record=base_record(16)
        record.update(expected_commit='a'*40,depth_forwards_attempted=80,depth_forwards_completed=80,
                      all_source_sha256={},acquisition_bindings={},frontend_bindings={})
        p.c.write_new(output/'selection.json',dict(selected=selected))
        p.c.write_new(output/'rgb-inputs.json',dict(images=[]))
        for row in selected:
            binary=output/('frame-%02d.npz'%row['index']); binary.write_bytes(b'artificial cache')
            saved=dict(row,status='completed',views=[dict(status='completed',name=name,index=i)
                       for i,name in enumerate(p.VIEW_NAMES)],
                       cache=dict(filename=binary.name,bytes=binary.stat().st_size,sha256=p.c.sha(binary)))
            p.c.write_new(output/('frame-%02d.json'%row['index']),saved)
            record['frames'][row['index']]=saved
        return selected,record

    def test_seal_full_coverage_then_cache_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); selected,record=self.make_seal_fixture(output)
            (output/'frame-08.npz').write_bytes(b'corrupted cache')
            with self.assertRaisesRegex(ValueError,'cache'):
                p.seal_outputs(output,selected,record)
            self.assertFalse((output/'seal.json').exists())
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); selected,record=self.make_seal_fixture(output)
            seal=p.seal_outputs(output,selected,record)
            self.assertEqual((seal['frame_count'],seal['depth_forward_count'],len(seal['outputs'])),(16,80,34))

    def test_failed_coverage_or_feature_call_never_seals(self):
        for change in ({'depth_forwards_completed':79},{'feature_forwards':1}):
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp); selected,record=self.make_seal_fixture(output); record.update(change)
                with self.assertRaises(ValueError): p.seal_outputs(output,selected,record)
                self.assertFalse((output/'seal.json').exists())

    def test_source_failure_retains_receipt_before_acquisition_or_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args=NS(output_dir=root/'out',expected_commit='a'*40)
            with mock.patch.object(p.c,'ROOT',root/'repository'), \
                 mock.patch.object(p,'prepare_loader',side_effect=ValueError('manufactured source mismatch')), \
                 mock.patch.object(p.c,'verify_acquisition') as acquisition:
                with self.assertRaisesRegex(ValueError,'source mismatch'): p.run(args)
            acquisition.assert_not_called()
            receipt=json.loads((args.output_dir/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            self.assertFalse(receipt['eligible']); self.assertEqual(receipt['depth_forwards_attempted'],0)
            self.assertFalse((args.output_dir/'seal.json').exists())

    def test_raw_negative_and_nonfinite_not_mask_filtered(self):
        backend=FakeBackend()
        raw={key:value.numpy() for key,value in backend.infer(np.zeros((336,448,3))).items()}
        raw['depth'][0,:3]=[-3.,0.,np.inf]
        self.assertTrue(raw['mask'][0,:3].all())
        p.validate_prediction(raw,(336,448),np)
        self.assertEqual(float(raw['depth'][0,0]),-3.)
        self.assertTrue(np.isinf(raw['depth'][0,2]))

    def test_opaque_rgba_and_nonopaque_rejection(self):
        class RGBA(ImageFixture):
            mode='RGBA'
            alpha=255
            def __array__(self,dtype=None,copy=None):
                array=np.full((480,640,4),100,np.uint8); array[...,3]=self.alpha
                return array
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgba.png'; path.write_bytes(artificial_encoded(6))
            image=RGBA(); source=NS(open=lambda _:image)
            raw,header=p.decode_rgb(path,source,np)
            self.assertEqual(raw.shape,(480,640,3)); self.assertEqual(header['color_type'],6)
            image.alpha=254
            with self.assertRaisesRegex(ValueError,'nonopaque'): p.decode_rgb(path,source,np)


if __name__=='__main__':
    unittest.main()
