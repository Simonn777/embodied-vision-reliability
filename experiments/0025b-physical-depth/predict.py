"""Fixed RGB-only, 80-forward physical RS depth qualification prediction stage.

No scientific imports or execution on import. Ground-truth PNG values, poses,
calibration fits and scientific metrics are never read/computed by this module.
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
import zlib

HERE = Path(__file__).resolve().parent
PROBE_FREEZE = '5d282a4b8b50094f433c6914931b005e7e16e73a'
VIEW_NAMES = ('gs','rs_x_positive','rs_x_negative','rs_y_positive','rs_y_negative')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


c = module('_evr0025b_prediction_common', HERE / 'common.py')


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def decode_rgb(path, Image, np):
    with Path(path).open('rb') as stream:
        header = stream.read(33)
    c.require(len(header) == 33 and header[:8] == b'\x89PNG\r\n\x1a\n'
              and header[8:16] == b'\x00\x00\x00\rIHDR'
              and zlib.crc32(header[12:29]) & 0xffffffff == int.from_bytes(header[29:33],'big'),
              'RGB PNG IHDR/signature/CRC')
    w,h,bits,color,compression,filtering,interlace = struct.unpack('>IIBBBBB',header[16:29])
    c.require((w,h,bits) == (640,480,8) and color in (2,6) and compression == filtering == 0
              and interlace in (0,1), 'RGB PNG header shape/type')
    expected_mode = 'RGB' if color == 2 else 'RGBA'
    with Image.open(path) as image:
        c.require(image.format == 'PNG' and image.size == (640,480) and image.mode == expected_mode
                  and image.getexif().get(274,1) == 1, 'source RGB PNG decoded mode/orientation')
        raw = np.asarray(image,dtype=np.uint8).copy()
    c.require(raw.shape == (480,640,3 if color == 2 else 4), 'decoded RGB channel shape')
    if color == 6:
        c.require((raw[...,3] == 255).all(), 'nonopaque source RGBA')
        raw = raw[...,:3].copy()
    return raw, dict(bit_depth=bits,color_type=color,interlace=interlace,
                     alpha_policy='RGB unchanged; RGBA must be entirely opaque then alpha removed')


def validate_prediction(raw, shape, np):
    """Structural checks only; masks never admit/discard raw depth samples."""
    c.require({'depth','mask','points','intrinsics'} <= set(raw), 'missing MoGe outputs')
    c.require(raw['depth'].shape == raw['mask'].shape == shape and raw['depth'].dtype == np.float32
              and raw['mask'].dtype == np.bool_, 'MoGe depth/mask shape/dtype')
    c.require(raw['points'].shape == (*shape,3) and raw['intrinsics'].shape == (3,3)
              and np.isfinite(raw['intrinsics']).all(), 'MoGe geometry shape/intrinsics')
    if 'normal' in raw:
        c.require(raw['normal'].shape == (*shape,3), 'MoGe normal shape')


def prepare_loader(args):
    """All Git/source qualification occurs before Backend installs its guard.

    The old command locates qualified model/runtime assets only. No old plan,
    Map-free image, calibration or acquisition payload is opened/verified here.
    """
    sources = c.verify_sources(args.expected_commit)
    supervisor = c.module('_evr0025b_predict_supervisor', HERE / 'supervise.py')
    legacy_path = c.ROOT / 'experiments/0024-virtual-depth/run.py'
    legacy_sources = supervisor.verify_frozen(PROBE_FREEZE, [legacy_path])
    legacy = module('_evr0025b_frozen0024_locator', legacy_path)
    old = legacy.original_args(args.original_frontend_supervision)
    base = c.ROOT / 'experiments/0023-mapfree'
    old_paths = [base / name for name in ('frontend.py', 'acquire_operation.py',
                 'range_inputs.py', 'propose_plan.py', 'supervise.py')]
    old_paths.append(c.ROOT / 'protocols/0023-mapfree-directional-instability.md')
    old_sources = supervisor.verify_frozen(legacy.INPUT_FREEZE, old_paths)
    frontend = module('_evr0025b_frozen_frontend', base / 'frontend.py')
    qa = frontend.qualification_args(old)
    frozen = frontend.q.frozen_bindings(qa)
    files, _, wheels, bindings = frontend.q.verify_files(qa, frozen)
    qualified = frontend.verify_qualification(old, bindings, frozen)
    all_sources = {}
    for group in (sources, legacy_sources, old_sources, frozen):
        for path, digest in group.items():
            c.require(path not in all_sources or all_sources[path] == digest, 'conflicting source binding')
            all_sources[path] = digest
    return dict(frontend=frontend, legacy=legacy, old=old, qa=qa, files=files,
                wheels=wheels, frozen=frozen, bindings=bindings, qualified=qualified,
                sources=sources, all_sources=all_sources)


def remap_rgb(raw, xy, np):
    """Float64 channel interpolation; one float32 cast, no uint8 requantization."""
    c.require(raw.shape == (480, 640, 3) and raw.dtype == np.uint8, 'source RGB shape/dtype')
    c.require(xy.shape == (336, 448, 2) and np.isfinite(xy).all(), 'RGB map shape/finiteness')
    c.require((xy[..., 0] >= 0).all() and (xy[..., 0] < 639).all()
              and (xy[..., 1] >= 0).all() and (xy[..., 1] < 479).all(),
              'RGB map requires unobserved neighbors')
    lo = np.floor(xy).astype(np.int64)
    x, y = lo[..., 0], lo[..., 1]
    dx, dy = (xy[..., 0]-x)[..., None], (xy[..., 1]-y)[..., None]
    values = raw.astype(np.float64) / 255.0
    result = ((1-dx)*(1-dy)*values[y, x] + dx*(1-dy)*values[y, x+1]
              + (1-dx)*dy*values[y+1, x] + dx*dy*values[y+1, x+1])
    c.require(np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all(),
              'invalid remapped RGB')
    return np.ascontiguousarray(result, dtype=np.float32)


def process_frame(row, encoded, output, backend, core, frontend, legacy, record, save):
    np, Image, torch = backend.np, backend.Image, backend.torch
    began, arrays = time.perf_counter(), {}
    result = dict(schema='evr0025b.frame-prediction.v1', **row, status='started',
                  stage='decode_rgb', gt_values_decoded=False, views=[], timings={})
    record['frames'][row['index']] = result
    save()
    try:
        path = Path(encoded['path'])
        c.require(path.stat().st_size == encoded['bytes'] and c.sha(path) == encoded['sha256'],
                  'extracted RGB bytes changed')
        raw, png_header = decode_rgb(path,Image,np)
        result['input'] = dict(filename='rgb/'+path.name, encoded_sha256=encoded['sha256'],
                               encoded_bytes=encoded['bytes'], member=encoded['member'],
                               decoded_rgb_sha256=raw_sha(raw), shape=[480, 640, 3], png=png_header)
        result['timings']['decode_seconds'] = time.perf_counter()-began
        result['camera'] = dict(source_K=core.SOURCE_K.tolist(), canonical_K=core.OUTPUT_K.tolist(),
            size=[core.WIDTH, core.HEIGHT], readout_seconds=core.READOUT_SECONDS,
            total_angle_radians=core.TOTAL_ANGLE_RADIANS,
            coordinate_convention='zero-based array centers; positive-fy custom coordinates; no source poses')
        fov = math.degrees(2*math.atan(core.WIDTH/(2*float(core.OUTPUT_K[0, 0]))))
        result['camera']['fov_x_degrees'] = fov
        c.require(tuple(core.VIEWS) == VIEW_NAMES, 'fixed five-view ordering')
        for index, name in enumerate(core.VIEWS):
            result['stage'] = name
            record['stage'] = 'frame-%02d:%s' % (row['index'], name)
            view = dict(index=index, name=name, status='started')
            result['views'].append(view)
            save()
            started = time.perf_counter()
            geometry = core.make_view_geometry(name)
            image = remap_rgb(raw, geometry['source_xy'], np)
            view.update(model_input_sha256=raw_sha(image), model_input_shape=list(image.shape),
                model_input_dtype=str(image.dtype), remap_sha256=raw_sha(geometry['source_xy']),
                h_sha256=raw_sha(geometry['h']), row_angle_sha256=raw_sha(geometry['row_angle_radians']),
                full_rgb_four_neighbor_inbounds=geometry['full_rgb_four_neighbor_inbounds'],
                remap_seconds=time.perf_counter()-started, fov_x_degrees=fov)
            record['depth_forwards_attempted'] += 1
            save()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    prediction = backend.moge.infer(backend.tensor(image),
                        **dict(frontend.q.MOGE_OPTIONS, fov_x=fov))
            finally:
                view['inference_seconds'] = time.perf_counter()-started
            values = {key: value.detach().cpu().numpy() for key, value in prediction.items()}
            # Retain unfiltered raw arrays even if a subsequent check fails.
            for key in ('depth', 'mask', 'intrinsics'):
                if key in values:
                    arrays['view_%d_%s' % (index, key)] = values[key].copy()
            validate_prediction(values, image.shape[:2], np)
            legacy.check_intrinsics(values['intrinsics'], core.OUTPUT_K, (core.WIDTH, core.HEIGHT), np)
            view['depth_diagnostics'] = dict(model_mask_count=int(values['mask'].sum()),
                finite_count=int(np.isfinite(values['depth']).sum()),
                nonpositive_finite_count=int((np.isfinite(values['depth']) & (values['depth'] <= 0)).sum()),
                model_mask_used_for_support=False)
            view['status'] = 'completed'
            record['depth_forwards_completed'] += 1
            save()
            del prediction, values, image, geometry
        result['stage'] = 'cache'
        started = time.perf_counter()
        result['cache'] = frontend.q.save_arrays(output / ('frame-%02d.npz' % row['index']), arrays, np)
        result['timings']['cache_seconds'] = time.perf_counter()-started
        result.update(status='completed', stage='complete')
        return result
    except BaseException as exc:
        result.update(status='failed', failed_stage=result['stage'],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        if result['views'] and result['views'][-1]['status'] == 'started':
            result['views'][-1]['status'] = 'failed'
        if arrays and 'cache' not in result:
            try:
                result['partial_cache'] = frontend.q.save_arrays(
                    output / ('frame-%02d-partial.npz' % row['index']), arrays, np)
            except BaseException as saving:
                result['partial_cache_error'] = dict(type=type(saving).__name__, message=str(saving))
        raise
    finally:
        result['timings']['total_seconds'] = time.perf_counter()-began
        c.write_new(output / ('frame-%02d.json' % row['index']), result)
        save()


def seal_outputs(output, selected, record):
    c.require(len(selected) == len(record['frames']) == 16
              and [r['index'] for r in selected] == list(range(16)), 'prediction coverage')
    c.require(record['depth_forwards_attempted'] == record['depth_forwards_completed'] == 80,
              'exactly eighty depth forwards required')
    c.require(record['feature_forwards'] == record['matcher_forwards'] == record['native_calls'] == 0
              and record['gt_values_decoded'] is False and not record['offline_attempts'],
              'forbidden operation or offline attempt')
    files = [output / 'selection.json', output / 'rgb-inputs.json']
    for row in selected:
        path = output / ('frame-%02d.json' % row['index'])
        saved = c.read_json(path)
        c.require(saved == record['frames'][row['index']] and saved['status'] == 'completed'
                  and all(saved[k] == v for k, v in row.items())
                  and len(saved['views']) == 5
                  and [v['name'] for v in saved['views']] == list(VIEW_NAMES)
                  and [v['index'] for v in saved['views']] == list(range(5))
                  and all(v['status'] == 'completed' for v in saved['views']), 'incomplete/changed frame')
        cache = saved['cache']
        binary = output / ('frame-%02d.npz' % row['index'])
        c.require(cache['filename'] == binary.name and binary.stat().st_size == cache['bytes']
                  and c.sha(binary) == cache['sha256'], 'changed raw prediction cache')
        files.extend((path, binary))
    seal = dict(schema='evr0025b.prediction-seal.v1', expected_commit=record['expected_commit'],
        frame_count=16, depth_forward_count=80, gt_values_decoded=False,
        source_sha256=record['all_source_sha256'], acquisition_bindings=record['acquisition_bindings'],
        frontend_bindings=record['frontend_bindings'], selection_sha256=c.sha(output/'selection.json'),
        rgb_inputs_sha256=c.sha(output/'rgb-inputs.json'),
        outputs=[dict(filename=p.name, bytes=p.stat().st_size, sha256=c.sha(p)) for p in files])
    c.write_new(output / 'seal.json', seal)
    return seal


def run(args):
    output = c.private_output(args.output_dir)
    began = time.perf_counter()
    record = dict(schema='evr0025b.prediction.v1', status='started', eligible=False,
        expected_commit=args.expected_commit, stage='verification', timings={}, warnings=[],
        offline_attempts=[], compatibility_probes=[], local_hub_calls=[], frames=[],
        depth_forwards_attempted=0, depth_forwards_completed=0, feature_forwards=0,
        matcher_forwards=0, native_calls=0, gt_values_decoded=False)
    record['warnings_scope'] = 'Explicit loading/inference contexts; import warnings remain in captured outer stderr.'
    def save():
        c.save_receipt(output / 'receipt.json', record)
    save()
    try:
        loader = prepare_loader(args)
        frontend, legacy = loader['frontend'], loader['legacy']
        record.update(source_sha256=loader['sources'], all_source_sha256=loader['all_sources'],
            probe_source_freeze=PROBE_FREEZE, input_source_freeze=legacy.INPUT_FREEZE,
            model_freeze=legacy.MODEL_FREEZE,
            original_frontend_supervision_sha256=c.sha(args.original_frontend_supervision),
            frontend_bindings=dict(loader['bindings'], **loader['qualified']))
        indices, acquired = c.verify_acquisition(args.acquisition_dir, args.acquisition_supervision, args.expected_commit)
        record['acquisition_bindings'] = acquired
        selected, populations = c.select_frames(indices)
        c.require(len(selected) == 16, 'fixed sixteen frames')
        selection = dict(schema='evr0025b.selection.v1', selected=selected, populations=populations,
                         acquisition_bindings=acquired, selection_uses_pixels=False)
        c.write_new(output / 'selection.json', selection)
        record['selection_sha256'] = c.sha(output / 'selection.json')
        record['frames'] = [dict(row, status='not_attempted') for row in selected]
        record['stage'] = 'extract_rgb'; save()
        started = time.perf_counter()
        encoded = c.extract_selected(args.acquisition_dir, selected, 'rgb', output / 'rgb')
        c.require(set(encoded) == set(range(16)), 'exact extracted RGB identities')
        rgb_inputs = [dict(index=i,filename='rgb/'+Path(encoded[i]['path']).name,
                          sha256=encoded[i]['sha256'],bytes=encoded[i]['bytes'],member=encoded[i]['member'])
                      for i in range(16)]
        c.write_new(output/'rgb-inputs.json',dict(schema='evr0025b.rgb-inputs.v1',images=rgb_inputs,
                    selection_sha256=record['selection_sha256'],all_hashes_recorded_before_pixel_decode=True))
        record['rgb_inputs_sha256'] = c.sha(output/'rgb-inputs.json')
        record['timings']['rgb_extract_seconds'] = time.perf_counter()-started
        record['timings']['verification_seconds'] = time.perf_counter()-began
        record['settings'] = dict(moge=dict(frontend.q.MOGE_OPTIONS, fov_x='fixed canonical K'),
            depth_forwards=80, cpu_threads=4, source_rgb=[640,480], output_rgb=[448,336],
            rgb_interpolation='float64 RGB/255 ordinary bilinear; once float32; no uint8 requantization',
            protocol_sha256=c.sha(c.PROTOCOL),
            loader='unchanged qualified0023 Backend; MoGe, SuperPoint and LightGlue loaded; latter two never inferred')
        record['stage'] = 'model_loading'; save()
        started = time.perf_counter()
        try:
            backend = frontend.Backend(loader['old'], loader['files'], loader['wheels'], record)
        finally:
            record['timings']['backend_total_seconds'] = time.perf_counter()-started
        record['offline_guard_scope'] = 'Python network/process/input audit events; not an OS native-syscall sandbox'
        core = module('_evr0025b_geometry_core', HERE / 'geometry_core.py')
        with frontend.q.capture_warnings(record):
            for row in selected:
                process_frame(row, encoded[row['index']], output, backend, core, frontend, legacy, record, save)
        record['stage'] = 'postcheck'; save()
        started = time.perf_counter()
        # Everything after Backend is file hashing/JSON validation, never Git.
        c.verify_source_bytes(loader['all_sources'])
        _, end_acquired = c.verify_acquisition(args.acquisition_dir, args.acquisition_supervision, args.expected_commit)
        c.require(end_acquired == acquired, 'acquisition changed during prediction')
        for item in encoded.values():
            c.require(Path(item['path']).stat().st_size == item['bytes']
                      and c.sha(item['path']) == item['sha256'], 'RGB changed during prediction')
        _, _, _, end_bindings = frontend.q.verify_files(loader['qa'], loader['frozen'])
        c.require(end_bindings == loader['bindings'] and
                  frontend.verify_qualification(loader['old'], end_bindings, loader['frozen']) == loader['qualified'],
                  'model/runtime/qualification changed')
        c.require(c.sha(args.original_frontend_supervision) == record['original_frontend_supervision_sha256'],
                  'original locator receipt changed')
        legacy.original_args(args.original_frontend_supervision)  # JSON/log hashes only.
        c.require(c.sha(output/'selection.json') == record['selection_sha256'], 'selection changed')
        c.require(c.sha(output/'rgb-inputs.json') == record['rgb_inputs_sha256'], 'RGB input manifest changed')
        seal_outputs(output, selected, record)
        record['seal_sha256'] = c.sha(output / 'seal.json')
        record['timings']['postcheck_seconds'] = time.perf_counter()-started
        record.update(status='completed', eligible=True, stage='complete',
                      eligibility_requires_clean_outer_supervision=True)
    except BaseException as exc:
        record.update(status='failed', eligible=False, failed_stage=record['stage'],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        record['wall_seconds'] = time.perf_counter()-began
        save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    for flag in ('acquisition-dir', 'acquisition-supervision', 'original-frontend-supervision', 'output-dir'):
        parser.add_argument('--'+flag, type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ('status','eligible','wall_seconds','depth_forwards_completed')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
