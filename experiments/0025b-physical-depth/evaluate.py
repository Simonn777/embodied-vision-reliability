"""Post-seal physical-depth diagnostic; oracle calibration, never a pose method.

Only after the complete prediction cohort AND its clean outer receipt verify
may this module extract/decode selected GT PNG samples. Source poses are unused.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time
import warnings
import zlib
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


c = module('_evr0025b_evaluation_common', HERE / 'common.py')


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def save_arrays(path, arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)
    return dict(filename=Path(path).name, sha256=c.sha(path), bytes=Path(path).stat().st_size,
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                         raw_c_order_sha256=raw_sha(value)) for key, value in arrays.items()})


def load_arrays(path, binding, np):
    c.require(Path(path).name == binding['filename'] and Path(path).stat().st_size == binding['bytes']
              and c.sha(path) == binding['sha256'], 'prediction cache file changed')
    with np.load(path, allow_pickle=False) as archive:
        c.require(set(archive.files) == set(binding['arrays']), 'prediction array names changed')
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        c.require(binding['arrays'][key] == dict(shape=list(value.shape), dtype=str(value.dtype),
                  raw_c_order_sha256=raw_sha(value)), 'prediction array bytes/metadata changed')
    return arrays


def verify_predictions(args, selected, populations, acquired, np, core):
    """Verify every output before ANY selected ground-truth value is decoded."""
    folder = Path(args.prediction_dir)
    outer = c.verify_outer(args.prediction_supervision, 'predict', args.expected_commit,
                           folder, '--output-dir')
    flags = c.command_flags(outer['command'])
    c.require(set(flags) == {'--expected-commit','--acquisition-dir','--acquisition-supervision',
              '--original-frontend-supervision','--output-dir'}, 'prediction caller flags')
    c.require(c.expand(flags['--acquisition-dir']) == Path(args.acquisition_dir).resolve()
              and c.expand(flags['--acquisition-supervision']) == Path(args.acquisition_supervision).resolve(),
              'prediction outer acquisition inputs differ')
    predictor = module('_evr0025b_eval_prediction_bindings', HERE/'predict.py')
    loader = predictor.prepare_loader(SimpleNamespace(expected_commit=args.expected_commit,
        original_frontend_supervision=c.expand(flags['--original-frontend-supervision'])))
    receipt = c.read_json(folder / 'receipt.json')
    c.require(receipt.get('schema') == 'evr0025b.prediction.v1' and
              receipt.get('expected_commit') == args.expected_commit and
              receipt.get('status') == 'completed' and receipt.get('eligible') is True and
              receipt.get('gt_values_decoded') is False and
              receipt.get('depth_forwards_attempted') == receipt.get('depth_forwards_completed') == 80 and
              all(receipt.get(key) == 0 for key in ('feature_forwards','matcher_forwards','native_calls')) and
              receipt.get('offline_attempts') == [], 'incomplete/invalid prediction cohort')
    c.require(receipt['source_sha256'] == outer['source_sha256'] == loader['sources'] and
              receipt['all_source_sha256'] == loader['all_sources'] and
              receipt['probe_source_freeze'] == predictor.PROBE_FREEZE and
              receipt['input_source_freeze'] == loader['legacy'].INPUT_FREEZE and
              receipt['model_freeze'] == loader['legacy'].MODEL_FREEZE and
              receipt['original_frontend_supervision_sha256'] ==
              c.sha(c.expand(flags['--original-frontend-supervision'])) and
              receipt['frontend_bindings'] == dict(loader['bindings'],**loader['qualified']),
              'prediction source/model/qualification chain differs')
    c.verify_source_bytes(receipt['all_source_sha256'])
    seal = c.read_json(folder / 'seal.json')
    c.require(receipt['seal_sha256'] == c.sha(folder/'seal.json') and
              seal.get('schema') == 'evr0025b.prediction-seal.v1' and
              seal.get('expected_commit') == args.expected_commit and
              seal.get('frame_count') == 16 and seal.get('depth_forward_count') == 80 and
              seal.get('gt_values_decoded') is False and
              seal.get('source_sha256') == receipt['all_source_sha256'] and
              seal.get('acquisition_bindings') == receipt['acquisition_bindings'] == acquired and
              seal.get('frontend_bindings') == receipt['frontend_bindings'], 'prediction seal differs')
    expected_names = {'selection.json', 'rgb-inputs.json'} | {
        'frame-%02d.%s' % (index, extension) for index in range(16) for extension in ('json','npz')}
    c.require(len(seal['outputs']) == 34 and {v['filename'] for v in seal['outputs']} == expected_names,
              'seal must bind all 34 outputs')
    for entry in seal['outputs']:
        path = folder / entry['filename']
        c.require(path.stat().st_size == entry['bytes'] and c.sha(path) == entry['sha256'],
                  'sealed output changed')
    selection = c.read_json(folder/'selection.json')
    c.require(selection.get('schema') == 'evr0025b.selection.v1' and
              selection.get('selected') == selected and selection.get('populations') == populations and
              selection.get('acquisition_bindings') == acquired and
              selection.get('selection_uses_pixels') is False and
              receipt['selection_sha256'] == seal['selection_sha256'] == c.sha(folder/'selection.json'),
              'prediction cohort no longer matches structural selection')
    c.require(len(receipt['frames']) == 16, 'prediction frame count')
    rgb_inputs = c.read_json(folder/'rgb-inputs.json')
    c.require(rgb_inputs.get('schema') == 'evr0025b.rgb-inputs.v1' and
              rgb_inputs.get('selection_sha256') == receipt['selection_sha256'] and
              rgb_inputs.get('all_hashes_recorded_before_pixel_decode') is True and
              len(rgb_inputs.get('images',[])) == 16 and
              [v['index'] for v in rgb_inputs['images']] == list(range(16)) and
              receipt['rgb_inputs_sha256'] == seal['rgb_inputs_sha256'] == c.sha(folder/'rgb-inputs.json'),
              'pre-decode RGB identity inventory differs')
    predictions = []
    expected_keys = {'view_%d_%s' % (i, key) for i in range(5) for key in ('depth','mask','intrinsics')}
    expected_k = np.array([[481.2/448, 0., .5], [0., 481.2/336, .5], [0.,0.,1.]])
    geometries = {name: core.make_view_geometry(name) for name in core.VIEWS}
    for row in selected:
        index = row['index']
        saved = c.read_json(folder/('frame-%02d.json' % index))
        c.require(saved == receipt['frames'][index] and saved['status'] == 'completed' and
                  saved['gt_values_decoded'] is False and all(saved[k] == v for k,v in row.items()),
                  'frame identity/status changed')
        c.require(saved['camera']['source_K'] == core.SOURCE_K.tolist() and
                  saved['camera']['canonical_K'] == core.OUTPUT_K.tolist() and
                  saved['camera']['size'] == [448,336] and saved['camera']['readout_seconds'] == .04 and
                  saved['camera']['total_angle_radians'] == core.TOTAL_ANGLE_RADIANS and
                  saved['camera']['fov_x_degrees'] == math.degrees(2*math.atan(448/(2*481.2))),
                  'frame camera/motion changed')
        rgb = folder/'rgb'/('frame-%02d.png' % index)
        c.require(rgb_inputs['images'][index] == dict(index=index,filename='rgb/'+rgb.name,
                  sha256=saved['input']['encoded_sha256'],bytes=saved['input']['encoded_bytes'],
                  member=row['rgb_member']['name']), 'RGB inventory/frame mismatch')
        c.require(saved['input']['filename'] == 'rgb/'+rgb.name and
                  saved['input']['member'] == row['rgb_member']['name'] and
                  rgb.stat().st_size == saved['input']['encoded_bytes'] == row['rgb_member']['size'] and
                  c.sha(rgb) == saved['input']['encoded_sha256'], 'selected RGB binding changed')
        arrays = load_arrays(folder/('frame-%02d.npz' % index), saved['cache'], np)
        c.require(set(arrays) == expected_keys and len(saved['views']) == 5, 'frame array/view count')
        for i,name in enumerate(core.VIEWS):
            view = saved['views'][i]
            geometry = geometries[name]
            c.require(view['index'] == i and view['name'] == name and view['status'] == 'completed' and
                      view['full_rgb_four_neighbor_inbounds'] is True and
                      view['remap_sha256'] == raw_sha(geometry['source_xy']) and
                      view['h_sha256'] == raw_sha(geometry['h']) and
                      view['row_angle_sha256'] == raw_sha(geometry['row_angle_radians']) and
                      view['fov_x_degrees'] == saved['camera']['fov_x_degrees'] and
                      view['model_input_shape'] == [336,448,3] and view['model_input_dtype'] == 'float32',
                      'view generation contract changed')
            depth, mask, intrinsics = [arrays['view_%d_%s' % (i,key)] for key in ('depth','mask','intrinsics')]
            c.require(depth.shape == mask.shape == (336,448) and depth.dtype == np.float32 and
                      mask.dtype == np.bool_ and intrinsics.shape == (3,3) and np.isfinite(intrinsics).all() and
                      np.all(np.abs(intrinsics-expected_k) <= 1e-5*np.abs(expected_k)), 'raw cache structure/K')
        predictions.append(arrays)
    return predictions, dict(receipt_sha256=c.sha(folder/'receipt.json'),
        seal_sha256=c.sha(folder/'seal.json'), supervision_sha256=c.sha(args.prediction_supervision),
        supervision_log_sha256={key:outer[key+'_sha256'] for key in ('stdout','stderr')})


def truth_header(encoded):
    """Hash and IHDR only; no image decoder or numeric ground-truth samples."""
    path = Path(encoded['path'])
    c.require(path.stat().st_size == encoded['bytes'] and c.sha(path) == encoded['sha256'],
              'selected GT PNG bytes changed')
    with path.open('rb') as stream:
        header = stream.read(33)
    c.require(len(header) == 33 and header[:8] == b'\x89PNG\r\n\x1a\n' and
              header[8:16] == b'\x00\x00\x00\rIHDR' and
              zlib.crc32(header[12:29]) & 0xffffffff == struct.unpack('>I', header[29:33])[0],
              'GT PNG signature/IHDR CRC')
    width,height,bits,color,compression,filter_method,interlace = struct.unpack('>IIBBBBB',header[16:29])
    c.require((width,height,bits,color,compression,filter_method) == (640,480,16,0,0,0)
              and interlace in (0,1), 'GT must be 640x480 16-bit grayscale PNG')
    return dict(width=width,height=height,bit_depth=bits,color_type=color,
                interlace=interlace,ihdr_sha256=hashlib.sha256(header).hexdigest())


def preflight_truth(encoded, selected, output):
    c.require(set(encoded) == set(range(16)) and [r['index'] for r in selected] == list(range(16)),
              'exact sixteen GT input identities')
    rows=[]
    for row in selected:
        item=encoded[row['index']]
        path=Path(item['path'])
        c.require(path.resolve() == (output/'gt-png'/('frame-%02d.png'%row['index'])).resolve()
                  and not path.is_symlink() and item['member']==row['depth_member']['name']
                  and item['bytes']==row['depth_member']['size'], 'selected GT member identity')
        rows.append(dict(index=row['index'],scene=row['scene'],numeric_stem=row['numeric_stem'],
            filename='gt-png/'+path.name,member=item['member'],bytes=item['bytes'],sha256=item['sha256'],
            png=truth_header(item)))
    artifact=dict(schema='evr0025b.GT-inputs.v1',images=rows,
                  all_hashes_and_IHDRs_recorded_before_numeric_decode=True,gt_values_decoded=False)
    c.write_new(output/'GT-inputs.json',artifact)
    return artifact


def decode_truth(encoded, np, Image):
    """Strict 16-bit grayscale TUM PNG axial z; no native-range conversion."""
    truth_header(encoded)  # Recheck after the complete sixteen-file preflight.
    path=Path(encoded['path'])
    with Image.open(path) as image:
        c.require(image.format == 'PNG' and image.size == (640,480) and
                  image.mode in ('I','I;16','I;16B','I;16L') and image.getexif().get(274,1) == 1,
                  'GT PNG decoded mode/orientation')
        stored = np.asarray(image).copy()
    c.require(stored.shape == (480,640) and stored.dtype.kind in 'iu' and
              (stored >= 0).all() and (stored <= 65535).all(), 'GT integer values/shape')
    stored = stored.astype(np.uint16)
    return stored.astype(np.float64)/5000., dict(encoded_sha256=encoded['sha256'],
        encoded_bytes=encoded['bytes'], member=encoded['member'], decoded_uint16_sha256=raw_sha(stored),
        zero_invalid_count=int((stored == 0).sum()), positive_count=int((stored > 0).sum()),
        units='axial metres; stored uint16 / 5000; zero invalid', source_poses_read=False)


def summary(values, np):
    values = np.asarray(values,dtype=np.float64)
    c.require(values.ndim == 1 and np.isfinite(values).all(), 'summary requires finite 1D values')
    if values.size == 0:
        return dict(count=0, mean=None, median=None, quantiles=None)
    return dict(count=int(values.size), mean=float(np.mean(values)), median=float(np.median(values)),
                quantiles={str(q):float(np.quantile(values,q)) for q in (0.,.25,.5,.75,.9,.95,1.)})


def case_diagnostic(row, view_index, truth_z, prediction, core, np, output):
    name = core.VIEWS[view_index]
    truth = core.sample_truth(truth_z, core.make_view_geometry(name), max_relative_depth_spread=.02)
    designs = core.build_designs(prediction['view_%d_depth' % view_index], prediction['view_0_depth'], truth)
    arrays = dict(lattice_u=truth['lattice']['u'], lattice_v=truth['lattice']['v'],
        lattice_train=truth['lattice']['train'], lattice_eval=truth['lattice']['eval'], support=truth['support'],
        h=truth['h'], source_xy=truth['source_xy'], canonical_gs_xy=truth['canonical_gs_xy'],
        source_mid_z=truth['source_mid_z'], row_z=truth['row_z'], relative_depth_spread=truth['relative_depth_spread'])
    case = dict(schema='evr0025b.case-diagnostic.v1', frame_index=row['index'], scene=row['scene'],
        numeric_stem=row['numeric_stem'], view_index=view_index, view=name, oracle_gt_assisted=True,
        support=dict(status=truth['status'], reason=truth['reason'], counts=truth['counts'],
                     max_relative_depth_spread=.02, support_uses_model_values=False), arms={})
    use = truth['support']
    case['physical_mid_row_separation'] = summary(np.abs(truth['source_mid_z'][use]-truth['row_z'][use])/
                                                truth['row_z'][use],np)
    for arm in core.ARMS:
        design = designs[arm]
        fit = core.fit_on_training(design,truth)
        evaluation = core.evaluate_fixed_holdout(fit,design,truth)
        arrays[arm+'_design'] = design
        clean = {}
        for key,value in evaluation.items():
            if isinstance(value,np.ndarray):
                arrays[arm+'_'+key] = value
            else:
                clean[key] = value
        clean['errors'] = summary(evaluation['absolute_relative_error'],np) if evaluation['status']=='usable' else None
        clean['mean_absrel'] = clean['errors']['mean'] if clean['errors'] else None
        case['arms'][arm] = dict(fit=fit,evaluation=clean)
    case['raw_model_anomalies'] = {}
    for index,label in ((0,'gs'),(view_index,'rs')):
        depth,mask = [prediction['view_%d_%s' % (index,k)] for k in ('depth','mask')]
        lattice_depth = depth[truth['lattice']['v'],truth['lattice']['u']]
        case['raw_model_anomalies'][label] = dict(full_image_count=depth.size,
            full_image_nonfinite=int((~np.isfinite(depth)).sum()),
            full_image_nonpositive_finite=int((np.isfinite(depth)&(depth<=0)).sum()),
            full_image_model_mask_true=int(mask.sum()),
            own_lattice_outside_common_support=int((~use).sum()),
            own_lattice_outside_support_nonfinite=int((~use&~np.isfinite(lattice_depth)).sum()),
            own_lattice_outside_support_nonpositive_finite=int((~use&np.isfinite(lattice_depth)&(lattice_depth<=0)).sum()),
            mask_used_to_filter=False)
    filename = 'frame-%02d-%s' % (row['index'],name)
    case['arrays'] = save_arrays(output/(filename+'.npz'),arrays)
    c.write_new(output/(filename+'.json'),case)
    return case


def aggregate(cases, np):
    """All 64 primary cases or unavailable; no successful-subset primary score."""
    c.require(len(cases) == 64 and {(r['frame_index'],r['view_index']) for r in cases} ==
              {(i,v) for i in range(16) for v in range(1,5)}, 'exact 64-case coverage required')
    c.require(all(row['scene'] == ('living_room' if row['frame_index']<8 else 'office') for row in cases),
              'fixed scene membership')
    complete = all(row['support']['status']=='usable' and
        all(row['arms'][arm]['evaluation']['status']=='usable' for arm in ('rs_row','gs_transport')) for row in cases)
    result = dict(schema='evr0025b.aggregate.v1', case_count=64, primary_complete=complete,
        primary_status='available' if complete else 'unavailable', gate_passed=False,
        oracle_diagnostic_only=True, pose_gain_established=False, manuscript_admitted=False,
        arm_availability={arm:sum(row['arms'][arm]['evaluation']['status']=='usable' for row in cases)
                          for arm in ('rs_row','gs_transport','rs_mid','rs_scale_only')}, scenes=[])
    for name in ('living_room','office'):
        rows = [row for row in cases if row['scene']==name]
        c.require(len(rows)==32,'fixed 32-case scene count')
        record = dict(scene=name,case_count=32,primary_metrics=None,
                      absolute_error_gate=False,paired_excess_gate=False,descriptive_arms={})
        for arm in result['arm_availability']:
            values = [row['arms'][arm]['evaluation']['mean_absrel'] for row in rows
                      if row['arms'][arm]['evaluation']['status']=='usable']
            record['descriptive_arms'][arm] = dict(coverage=len(values), fixed_denominator=32,
                label='all-case descriptive' if len(values)==32 else 'partial-coverage descriptive; not primary',
                values=values, distribution=summary(values,np))
        if complete:
            errors = [row['arms']['rs_row']['evaluation']['mean_absrel'] for row in rows]
            deltas = [row['arms']['rs_row']['evaluation']['mean_absrel']-
                      row['arms']['gs_transport']['evaluation']['mean_absrel'] for row in rows]
            record['primary_metrics'] = dict(rs_row=summary(errors,np), paired_rs_minus_gs=summary(deltas,np))
            record['absolute_error_gate'] = record['primary_metrics']['rs_row']['median']<=.10
            record['paired_excess_gate'] = record['primary_metrics']['paired_rs_minus_gs']['median']<=.02
        result['scenes'].append(record)
    result['gate_passed'] = complete and all(r['absolute_error_gate'] and r['paired_excess_gate'] for r in result['scenes'])
    result['decision'] = 'PERMIT_SEPARATE_PROSPECTIVE_POSE_PROPOSAL' if result['gate_passed'] else 'CLOSE_TESTED_UNADAPTED_APPROXIMATION'
    return result


def run(args):
    output = c.private_output(args.output_dir)
    began = time.perf_counter()
    record = dict(schema='evr0025b.evaluation.v1',status='started',expected_commit=args.expected_commit,
        stage='verify_pre_GT_barrier',gt_values_decoded=False,source_poses_read=False,
        gt_decode_attempted=0,gt_decode_completed=0,gt_value_access_attempted=False,
        gt_values_decoded_meaning='At least one numeric decode completed successfully; false does not exclude a failed attempted decode.',
        frame_count=0,case_count=0,warnings=[],oracle_diagnostic_only=True,manuscript_admitted=False)
    def save():
        c.save_receipt(output/'receipt.json',record)
    save()
    try:
        import numpy as np
        from PIL import Image
        core = module('_evr0025b_eval_core',HERE/'geometry_core.py')
        bindings = c.verify_sources(args.expected_commit)
        indices,acquired = c.verify_acquisition(args.acquisition_dir,args.acquisition_supervision,args.expected_commit)
        selected,populations = c.select_frames(indices)
        predictions,predicted = verify_predictions(args,selected,populations,acquired,np,core)
        record.update(source_sha256=bindings, acquisition_bindings=acquired,prediction_bindings=predicted,
                      runtime=dict(python=sys.version,numpy=np.__version__,pillow=Image.__version__))
        c.write_new(output/'pre-GT-barrier.json',dict(schema='evr0025b.pre-GT-barrier.v1',
            acquisition_bindings=acquired,prediction_bindings=predicted,verified_frames=16,
            verified_forwards=80,gt_values_decoded=False,source_sha256=bindings))
        record['pre_GT_barrier_sha256'] = c.sha(output/'pre-GT-barrier.json')
        record['stage']='extract_selected_GT_PNG'; save()
        encoded = c.extract_selected(args.acquisition_dir,selected,'depth',output/'gt-png')
        record['stage']='preflight_all_GT_headers'; save()
        preflight_truth(encoded,selected,output)
        record['GT_inputs_sha256']=c.sha(output/'GT-inputs.json')
        save()
        cases=[]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                for row in selected:
                    record['stage']='decode_GT_frame-%02d' % row['index']
                    record['gt_decode_attempted']+=1
                    record['gt_value_access_attempted']=True
                    save()
                    truth,metadata = decode_truth(encoded[row['index']],np,Image)
                    record['gt_values_decoded']=True
                    record['gt_decode_completed']+=1
                    save()
                    c.write_new(output/('truth-%02d.json' % row['index']),dict(row,**metadata))
                    for view in range(1,5):
                        record['stage']='diagnostic_frame-%02d_view-%d' % (row['index'],view); save()
                        cases.append(case_diagnostic(row,view,truth,predictions[row['index']],core,np,output))
                        record['case_count']=len(cases); save()
                    record['frame_count']+=1
            finally:
                record['warnings']=[dict(category=w.category.__name__,message=str(w.message),
                                         filename=Path(w.filename).name,lineno=w.lineno) for w in caught]
        result=aggregate(cases,np)
        c.write_new(output/'aggregate.json',result)
        record['stage']='postcheck'; save()
        c.verify_source_bytes(bindings)
        _,end_acquired=c.verify_acquisition(args.acquisition_dir,args.acquisition_supervision,args.expected_commit)
        c.require(end_acquired==acquired,'acquisition changed during diagnostic')
        _,end_predicted=verify_predictions(args,selected,populations,acquired,np,core)
        c.require(end_predicted==predicted,'predictions changed during diagnostic')
        for item in encoded.values():
            c.require(c.sha(item['path'])==item['sha256'],'GT PNG bytes changed during diagnostic')
        c.require(c.sha(output/'GT-inputs.json')==record['GT_inputs_sha256'],'GT input inventory changed')
        outputs=[path for path in output.iterdir() if path.suffix in ('.json','.npz') and path.name!='receipt.json']
        c.write_new(output/'result-seal.json',dict(schema='evr0025b.result-seal.v1',expected_commit=args.expected_commit,
            outputs=[dict(filename=p.name,bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(outputs)]))
        record.update(status='completed',stage='complete',gate_passed=result['gate_passed'],
            decision=result['decision'],aggregate_sha256=c.sha(output/'aggregate.json'),
            result_seal_sha256=c.sha(output/'result-seal.json'),eligibility_requires_clean_outer_supervision=True)
    except BaseException as exc:
        record.update(status='failed',failed_stage=record['stage'],error=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        record['wall_seconds']=time.perf_counter()-began
        save()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit',required=True)
    for flag in ('acquisition-dir','acquisition-supervision','prediction-dir','prediction-supervision','output-dir'):
        parser.add_argument('--'+flag,type=Path,required=True)
    result=run(parser.parse_args())
    print(json.dumps({key:result[key] for key in ('status','case_count','gate_passed','decision','wall_seconds')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
