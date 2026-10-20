"""Artificial evaluator/barrier tests. No models, archive payloads or real GT."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import warnings
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_test0025b_eval',HERE/'evaluate.py')
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
ARMS=('rs_row','gs_transport','rs_mid','rs_scale_only')
VIEWS=('gs','rs_x_positive','rs_x_negative','rs_y_positive','rs_y_negative')


def png_header(bits=16,color=0):
    chunk=b'IHDR'+struct.pack('>IIBBBBB',640,480,bits,color,0,0,0)
    return b'\x89PNG\r\n\x1a\n'+(13).to_bytes(4,'big')+chunk+(zlib.crc32(chunk)&0xffffffff).to_bytes(4,'big')


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


def fake_core():
    geom=dict(source_xy=np.zeros((1,1,2)),h=np.ones((1,1)),row_angle_radians=np.zeros((1,1)))
    return NS(VIEWS=VIEWS,SOURCE_K=np.array([[481.2,0,319.5],[0,480,239.5],[0,0,1]]),
        OUTPUT_K=np.array([[481.2,0,223.5],[0,481.2,167.5],[0,0,1]]),
        TOTAL_ANGLE_RADIANS=math.radians(12),make_view_geometry=lambda _:geom)


def cases():
    return [dict(frame_index=i,view_index=v,scene='living_room' if i<8 else 'office',
        support=dict(status='usable'),arms={arm:dict(evaluation=dict(status='usable',mean_absrel=.05))
        for arm in ARMS}) for i in range(16) for v in range(1,5)]


class Fixture:
    def __init__(self,root):
        self.root=root; self.folder=root/'pred'; self.folder.mkdir(); (self.folder/'rgb').mkdir()
        self.original=root/'original.json';self.original.write_text('manufactured locator')
        self.outerpath=root/'outer.json';self.outerpath.write_text('manufactured outer')
        self.args=NS(prediction_dir=self.folder,prediction_supervision=self.outerpath,
                     acquisition_dir=root/'acq',acquisition_supervision=root/'acq-outer',expected_commit='a'*40)
        self.core=fake_core(); self.acquired={'archive':'manufactured'}; self.populations=[{'invented':True}]
        self.sources={'f.py':'b'*64}
        legacy=NS(INPUT_FREEZE='c'*40,MODEL_FREEZE='d'*40)
        self.loader=dict(sources=self.sources,all_sources=self.sources,legacy=legacy,
                         bindings={'model':'bound'},qualified={'qualification':'bound'})
        self.predictor=NS(PROBE_FREEZE='e'*40,prepare_loader=mock.Mock(return_value=self.loader))
        self.outer=dict(source_sha256=self.sources,stdout_sha256='1'*64,stderr_sha256='2'*64,
            command=['$PYTHON','$REPOSITORY/experiments/0025b-physical-depth/predict.py',
                '--expected-commit',self.args.expected_commit,'--acquisition-dir',str(self.args.acquisition_dir),
                '--acquisition-supervision',str(self.args.acquisition_supervision),
                '--original-frontend-supervision',str(self.original),'--output-dir',str(self.folder)])
        self.selected=[]; self.rows=[]; rgb_rows=[]
        k=np.array([[481.2/448,0,.5],[0,481.2/336,.5],[0,0,1]],dtype=np.float32)
        depth=np.ones((336,448),np.float32); mask=np.ones((336,448),np.bool_)
        self.arrays={f'view_{i}_{name}':value for i in range(5)
                     for name,value in dict(depth=depth,mask=mask,intrinsics=k).items()}
        fov=math.degrees(2*math.atan(448/(2*481.2))); geom=self.core.make_view_geometry('gs')
        for i in range(16):
            row=dict(index=i,scene='living_room' if i<8 else 'office',numeric_stem=i,
                     rgb_member=dict(name=f'rgb/{i}.png',size=3),depth_member=dict(name=f'depth/{i}.png',size=33))
            self.selected.append(row)
            rgb=self.folder/'rgb'/f'frame-{i:02d}.png';rgb.write_bytes(b'RGB')
            binary=self.folder/f'frame-{i:02d}.npz';binary.write_bytes(b'fake raw')
            inp=dict(filename='rgb/'+rgb.name,member=row['rgb_member']['name'],encoded_bytes=3,encoded_sha256=e.c.sha(rgb))
            rgb_rows.append(dict(index=i,filename=inp['filename'],member=inp['member'],bytes=3,sha256=inp['encoded_sha256']))
            saved=dict(row,status='completed',gt_values_decoded=False,input=inp,
                camera=dict(source_K=self.core.SOURCE_K.tolist(),canonical_K=self.core.OUTPUT_K.tolist(),
                    size=[448,336],readout_seconds=.04,total_angle_radians=self.core.TOTAL_ANGLE_RADIANS,fov_x_degrees=fov),
                cache=dict(filename=binary.name,bytes=binary.stat().st_size,sha256=e.c.sha(binary)),
                views=[dict(index=j,name=name,status='completed',full_rgb_four_neighbor_inbounds=True,
                    remap_sha256=e.raw_sha(geom['source_xy']),h_sha256=e.raw_sha(geom['h']),
                    row_angle_sha256=e.raw_sha(geom['row_angle_radians']),model_input_shape=[336,448,3],
                    model_input_dtype='float32',fov_x_degrees=fov) for j,name in enumerate(VIEWS)])
            self.rows.append(saved);write(self.folder/f'frame-{i:02d}.json',saved)
        write(self.folder/'selection.json',dict(schema='evr0025b.selection.v1',selected=self.selected,
            populations=self.populations,acquisition_bindings=self.acquired,selection_uses_pixels=False))
        write(self.folder/'rgb-inputs.json',dict(schema='evr0025b.rgb-inputs.v1',images=rgb_rows,
            selection_sha256=e.c.sha(self.folder/'selection.json'),all_hashes_recorded_before_pixel_decode=True))
        self.receipt=dict(schema='evr0025b.prediction.v1',expected_commit=self.args.expected_commit,
            status='completed',eligible=True,gt_values_decoded=False,depth_forwards_attempted=80,depth_forwards_completed=80,
            feature_forwards=0,matcher_forwards=0,native_calls=0,offline_attempts=[],source_sha256=self.sources,
            all_source_sha256=self.sources,probe_source_freeze=self.predictor.PROBE_FREEZE,
            input_source_freeze=legacy.INPUT_FREEZE,model_freeze=legacy.MODEL_FREEZE,
            original_frontend_supervision_sha256=e.c.sha(self.original),
            frontend_bindings=dict(self.loader['bindings'],**self.loader['qualified']),
            acquisition_bindings=self.acquired,frames=self.rows)
        self.reseal()

    def reseal(self):
        self.receipt['selection_sha256']=e.c.sha(self.folder/'selection.json')
        self.receipt['rgb_inputs_sha256']=e.c.sha(self.folder/'rgb-inputs.json')
        paths=[self.folder/x for x in ('selection.json','rgb-inputs.json')]
        paths += [self.folder/f'frame-{i:02d}.{ext}' for i in range(16) for ext in ('json','npz')]
        seal=dict(schema='evr0025b.prediction-seal.v1',expected_commit=self.args.expected_commit,
            frame_count=16,depth_forward_count=80,gt_values_decoded=False,source_sha256=self.sources,
            acquisition_bindings=self.acquired,frontend_bindings=self.receipt['frontend_bindings'],
            selection_sha256=self.receipt['selection_sha256'],rgb_inputs_sha256=self.receipt['rgb_inputs_sha256'],
            outputs=[dict(filename=x.name,bytes=x.stat().st_size,sha256=e.c.sha(x)) for x in paths])
        write(self.folder/'seal.json',seal);self.receipt['seal_sha256']=e.c.sha(self.folder/'seal.json')
        write(self.folder/'receipt.json',self.receipt)

    def verify(self):
        with mock.patch.object(e.c,'verify_outer',return_value=self.outer), \
             mock.patch.object(e.c,'verify_source_bytes'), mock.patch.object(e,'module',return_value=self.predictor), \
             mock.patch.object(e,'load_arrays',return_value=self.arrays):
            return e.verify_predictions(self.args,self.selected,self.populations,self.acquired,np,self.core)


class EvaluationTests(unittest.TestCase):
    def test_34_outputs_and_15_raw_arrays_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); predictions,binding=f.verify()
            self.assertEqual(len(predictions),16);self.assertEqual(len(predictions[0]),15)
            self.assertEqual(binding['supervision_log_sha256']['stdout'],'1'*64)
            f.arrays.pop('view_4_mask')
            with self.assertRaisesRegex(ValueError,'array/view'): f.verify()

    def test_tampered_sealed_output_and_unsafe_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp));(f.folder/'frame-15.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'sealed output'): f.verify()
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); seal=json.loads((f.folder/'seal.json').read_text())
            seal['outputs'][0]['filename']='../selection.json';write(f.folder/'seal.json',seal)
            f.receipt['seal_sha256']=e.c.sha(f.folder/'seal.json');write(f.folder/'receipt.json',f.receipt)
            with self.assertRaisesRegex(ValueError,'34 outputs'): f.verify()

    def test_caller_and_full_source_binding_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp));f.outer['command'] += ['--ignored','true']
            with self.assertRaisesRegex(ValueError,'caller flags'): f.verify()
            f.outer['command']=f.outer['command'][:-2]
            f.receipt['all_source_sha256']={'other.py':'f'*64};write(f.folder/'receipt.json',f.receipt)
            with self.assertRaisesRegex(ValueError,'source/model'): f.verify()

    def test_rgb_manifest_crosscheck_survives_consistent_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); values=json.loads((f.folder/'rgb-inputs.json').read_text())
            values['images'][7]['member']='rgb/999.png';write(f.folder/'rgb-inputs.json',values);f.reseal()
            with self.assertRaisesRegex(ValueError,'inventory/frame'): f.verify()

    def test_K_zero_exact_and_negative_raw_mask_true_is_not_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); f.arrays['view_0_depth'][0,:2]=[-4.,np.inf]
            self.assertTrue(f.arrays['view_0_mask'][0,:2].all()); f.verify()
            f.arrays['view_0_intrinsics'][0,1]=1e-12
            with self.assertRaisesRegex(ValueError,'structure/K'): f.verify()

    def test_array_metadata_and_bytes_verified_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.npz';binding=e.save_arrays(path,{'depth':np.array([-1.,np.inf],np.float32)})
            saved=e.load_arrays(path,binding,np);self.assertTrue(np.isinf(saved['depth'][1]))
            binding['arrays']['depth']['raw_c_order_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'array bytes'):e.load_arrays(path,binding,np)

    def test_all_GT_headers_preflight_before_any_numeric_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'gt-png').mkdir();encoded={};selected=[]
            for i in range(16):
                path=output/'gt-png'/f'frame-{i:02d}.png';path.write_bytes(png_header(bits=8 if i==15 else 16))
                encoded[i]=dict(path=str(path),bytes=33,sha256=e.c.sha(path),member=f'depth/{i}.png')
                selected.append(dict(index=i,scene='fake',numeric_stem=i,depth_member=dict(name=f'depth/{i}.png',size=33)))
            with mock.patch.object(e,'decode_truth',side_effect=AssertionError('no decode')) as decode:
                with self.assertRaisesRegex(ValueError,'16-bit'):e.preflight_truth(encoded,selected,output)
                decode.assert_not_called()
            self.assertFalse((output/'GT-inputs.json').exists())
            Path(encoded[15]['path']).write_bytes(png_header());encoded[15]['sha256']=e.c.sha(encoded[15]['path'])
            result=e.preflight_truth(encoded,selected,output)
            self.assertEqual(len(result['images']),16);self.assertFalse(result['gt_values_decoded'])

    def test_strict16bit_axial5000_and_zero_invalid(self):
        class Image:
            format='PNG';size=(640,480);mode='I;16'
            def __enter__(self):return self
            def __exit__(self,*_):return False
            def getexif(self):return {}
            def __array__(self,dtype=None,copy=None):
                a=np.full((480,640),10000,np.uint16);a[0,0]=0;return a
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'d.png';path.write_bytes(png_header())
            encoded=dict(path=str(path),bytes=33,sha256=e.c.sha(path),member='depth/0.png')
            truth,meta=e.decode_truth(encoded,np,NS(open=lambda _:Image()))
            self.assertEqual(float(truth[200,600]),2.);self.assertEqual(float(truth[0,0]),0.)
            self.assertEqual(meta['zero_invalid_count'],1)

    def test_paired_median_not_difference_of_medians(self):
        rows=cases()
        # 32 per scene: median(row)-median(gs)=-.08, paired median=0.
        a=[0.]*22+[.09]*10;b=[0.]*11+[.08]*21
        for offset in (0,32):
            for j in range(32):
                rows[offset+j]['arms']['rs_row']['evaluation']['mean_absrel']=a[j]
                rows[offset+j]['arms']['gs_transport']['evaluation']['mean_absrel']=b[j]
        result=e.aggregate(rows,np)
        expected=float(np.median(np.asarray(a)-np.asarray(b)))
        self.assertNotEqual(expected,float(np.median(a)-np.median(b)))
        self.assertEqual(result['scenes'][0]['primary_metrics']['paired_rs_minus_gs']['median'],expected)

    def test_required_unavailable_nulls_all_primary_optional_does_not(self):
        rows=cases();rows[0]['arms']['rs_scale_only']['evaluation']=dict(status='unavailable',mean_absrel=None)
        self.assertTrue(e.aggregate(rows,np)['gate_passed'])
        rows[1]['arms']['gs_transport']['evaluation']=dict(status='unavailable',mean_absrel=None)
        result=e.aggregate(rows,np)
        self.assertFalse(result['gate_passed']);self.assertFalse(result['primary_complete'])
        self.assertTrue(all(r['primary_metrics'] is None for r in result['scenes']))
        with self.assertRaisesRegex(ValueError,'64-case'):e.aggregate(rows[:-1],np)

    def test_four_oracle_arms_attempted_with_false_masks(self):
        support=np.ones(4,dtype=bool)
        lattice=dict(u=np.arange(4),v=np.zeros(4,dtype=int),train=np.array([1,0,1,0],bool),eval=np.array([0,1,0,1],bool))
        truth=dict(lattice=lattice,support=support,h=np.ones(4),source_xy=np.zeros((4,2)),
            canonical_gs_xy=np.zeros((4,2)),source_mid_z=np.full(4,2.),row_z=np.full(4,2.),
            relative_depth_spread=np.zeros(4),status='usable',reason=None,counts=dict(total=4096,support=4))
        calls=[]
        def fit(design,t):
            self.assertIs(t,truth);calls.append(design);return dict(status='usable',coefficients=[1.,0.])
        core=NS(VIEWS=VIEWS,ARMS=ARMS,make_view_geometry=lambda _:None,sample_truth=lambda *a,**k:truth,
            build_designs=lambda *a:{arm:np.ones((4,2)) for arm in ARMS},fit_on_training=fit,
            evaluate_fixed_holdout=lambda *a:dict(status='usable',absolute_relative_error=np.array([.1,.2])))
        prediction={f'view_{i}_{key}':value for i in (0,1) for key,value in
                    dict(depth=np.full((336,448),2,np.float32),mask=np.zeros((336,448),bool)).items()}
        with tempfile.TemporaryDirectory() as tmp:
            result=e.case_diagnostic(dict(index=0,scene='fake',numeric_stem=3),1,np.ones((480,640)),
                                     prediction,core,np,Path(tmp))
        self.assertEqual(len(calls),4);self.assertEqual(set(result['arms']),set(ARMS))
        self.assertTrue(all(r['evaluation']['status']=='usable' for r in result['arms'].values()))
        self.assertFalse(result['raw_model_anomalies']['rs']['mask_used_to_filter'])
        self.assertEqual(result['raw_model_anomalies']['rs']['full_image_model_mask_true'],0)

    def test_barrier_failure_prevents_GT_extraction_or_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=NS(output_dir=root/'out',expected_commit='a'*40,
                acquisition_dir=root/'acq',acquisition_supervision=root/'acq-outer')
            with mock.patch.object(e.c,'ROOT',root/'repository'),mock.patch.object(e,'module',return_value=fake_core()), \
                 mock.patch.object(e.c,'verify_sources',return_value={'f':'0'*64}), \
                 mock.patch.object(e.c,'verify_acquisition',return_value=({},{})), \
                 mock.patch.object(e.c,'select_frames',return_value=([],[])), \
                 mock.patch.object(e,'verify_predictions',side_effect=ValueError('manufactured failed outer')), \
                 mock.patch.object(e.c,'extract_selected') as extract,mock.patch.object(e,'decode_truth') as decode:
                with self.assertRaisesRegex(ValueError,'failed outer'):e.run(args)
                extract.assert_not_called();decode.assert_not_called()
            record=json.loads((args.output_dir/'receipt.json').read_text())
            self.assertFalse(record['gt_values_decoded']);self.assertEqual(record['status'],'failed')

    def test_attempt_counter_and_warning_preserved_if_decode_raises(self):
        def failing_decode(*_):
            warnings.warn('manufactured decode warning',RuntimeWarning)
            raise ValueError('manufactured numeric decode failure')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=NS(output_dir=root/'out',expected_commit='a'*40,
                acquisition_dir=root/'acq',acquisition_supervision=root/'acq-outer')
            selected=[dict(index=i) for i in range(16)]
            def preflight(_encoded,_selected,output):write(output/'GT-inputs.json',dict(artificial=True))
            with mock.patch.object(e.c,'ROOT',root/'repository'),mock.patch.object(e,'module',return_value=fake_core()), \
                 mock.patch.object(e.c,'verify_sources',return_value={'f':'0'*64}), \
                 mock.patch.object(e.c,'verify_acquisition',return_value=({},{})), \
                 mock.patch.object(e.c,'select_frames',return_value=(selected,[])), \
                 mock.patch.object(e,'verify_predictions',return_value=([{}]*16,{})), \
                 mock.patch.object(e.c,'extract_selected',return_value={0:{}}), \
                 mock.patch.object(e,'preflight_truth',side_effect=preflight), \
                 mock.patch.object(e,'decode_truth',side_effect=failing_decode):
                with self.assertRaisesRegex(ValueError,'numeric decode failure'):e.run(args)
            record=json.loads((args.output_dir/'receipt.json').read_text())
            self.assertEqual((record['gt_decode_attempted'],record['gt_decode_completed']),(1,0))
            self.assertTrue(record['gt_value_access_attempted']);self.assertFalse(record['gt_values_decoded'])
            self.assertEqual(record['warnings'][0]['message'],'manufactured decode warning')


if __name__=='__main__':unittest.main()
