"""Bounded virtual-rotation model-assumption probe; no GT or pose estimation.

All image/model execution is behind main/run. The installed 0022 loader and
0023 acquisition chain are reused without modifying their source or settings.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / 'protocols/0024-virtual-rotation-falsifier.md'
INPUT_FREEZE = 'acf502b831ad9e3485ab56549217c8a603d56bf7'
MODEL_FREEZE = 'fa454585e67a8f0395ad5f67a03d5a28cc33a66a'
PLAN_SHA = 'cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5'
INPUT_SHA = 'bf6cd9baafa4b8c72fc3a33021becc3cf37f3bf3a6ecec406df77c4b25114905'
FRONTEND_OUTER_SHA = '4439b3d24550e425d976a18f114aeb5734c526e3ec08bba73dfb4e206c3bba5f'
PATH_FLAGS = ('protocol', 'plan', 'input_receipt', 'input_operation', 'input_supervision',
              'inputs', 'output_dir', 'installation', 'installation_supervision',
              'acquisition_receipt', 'assets_dir', 'qualification_receipt',
              'qualification_supervision', 'frontend_protocol', 'asset_manifest',
              'wheel_manifest', 'wheel_assets_manifest')
TEXT_FLAGS = ('expected_commit', 'plan_sha256', 'input_receipt_sha256', 'frontend_freeze')
VIEW_NAMES = ('identity', 'yaw_positive', 'yaw_negative', 'pitch_positive', 'pitch_negative')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def original_args(outer_path):
    """Only the exact completed original command can identify reusable inputs."""
    require(sha(outer_path) == FRONTEND_OUTER_SHA, 'original frontend supervision hash')
    outer = json.loads(Path(outer_path).read_text())
    require(outer['status'] == 'completed' and outer['exit_code'] == 0
            and outer['expected_commit'] == INPUT_FREEZE
            and outer['remaining_descendant_pids'] == [] and outer['cleanup_errors'] == [],
            'original frontend not cleanly completed')
    command = outer['command']
    require(command[:2] == ['$PYTHON', '$REPOSITORY/experiments/0023-mapfree/frontend.py']
            and len(command[2:]) % 2 == 0, 'original command shape')
    pairs = list(zip(command[2::2], command[3::2]))
    require(len(set(k for k, _ in pairs)) == len(pairs), 'duplicate original flags')
    flags = {k.removeprefix('--').replace('-', '_'): v for k, v in pairs}
    require(set(flags) == set(PATH_FLAGS + TEXT_FLAGS), 'original command flags')
    for name in PATH_FLAGS:
        p = Path(flags[name].replace('$REPOSITORY', str(ROOT)).replace('$PRIVATE', str(ROOT.parent)))
        flags[name] = p if p.is_absolute() else ROOT / p
    args = SimpleNamespace(**flags)
    require(args.expected_commit == INPUT_FREEZE and args.frontend_freeze == MODEL_FREEZE
            and args.plan_sha256 == PLAN_SHA and args.input_receipt_sha256 == INPUT_SHA,
            'original plan/source identity')
    for name in ('stdout', 'stderr'):
        require(sha(Path(outer_path).parent / (name + '.log')) == outer[name + '_sha256'],
                'original frontend log hash')
    return args


def selected_images(plan):
    """Metadata only; use each anchor and eight queries, never references/labels."""
    rows = []
    for scene in plan['scenes']:
        names = [scene['anchor']] + scene['query_images']
        require(len(names) == 9 and len(set(names)) == 9, 'nine image names per place')
        require([e['target'] for e in scene['query_edges']] == names[1:]
                and all(e['source'] == names[0] for e in scene['query_edges']), 'query image identity')
        for name in names:
            rows.append(dict(scene=scene['scene'], image_id=name,
                             intrinsics_member=scene['intrinsics_member']['filename']))
    require(len(rows) == 27 and len({r['image_id'] for r in rows}) == 27
            and len({r['scene'] for r in rows}) == 3, 'fixed 27 image population')
    return rows


def verified_sources(frontend, expected_commit):
    current = frontend.acquisition.supervisor.verify_frozen(expected_commit,
        [Path(__file__), HERE / 'math_core.py', HERE / 'supervise.py', PROTOCOL])
    old_paths = frontend.acquisition.source_paths(ROOT / 'protocols/0023-mapfree-directional-instability.md')
    old_paths += [Path(frontend.__file__)]
    old = frontend.acquisition.supervisor.verify_frozen(INPUT_FREEZE, old_paths)
    return current, old


def check_intrinsics(actual, Kc, size, np):
    w, h = size
    expected = np.array([[Kc[0, 0]/w, 0., .5], [0., Kc[1, 1]/h, .5], [0., 0., 1.]])
    require(actual.shape == (3, 3) and np.isfinite(actual).all()
            and np.all(np.abs(actual-expected) <= 1e-5*np.abs(expected)),
            'known-FOV returned intrinsics differ from canonical camera')


def remap_rgb(raw, xy, cv2, np):
    h, w = raw.shape[:2]
    require(xy.shape[-1] == 2 and np.isfinite(xy).all()
            and (xy[..., 0] >= 2).all() and (xy[..., 0] <= w-3).all()
            and (xy[..., 1] >= 2).all() and (xy[..., 1] <= h-3).all(),
            'virtual view would consume unobserved/border pixels')
    coordinates = xy.astype(np.float32)
    rounded_coordinates = coordinates.astype(np.float64)
    require((rounded_coordinates[..., 0] >= 2).all() and (rounded_coordinates[..., 0] <= w-3).all()
            and (rounded_coordinates[..., 1] >= 2).all() and (rounded_coordinates[..., 1] <= h-3).all(),
            'float32 remap coordinates exceed observed rectangle')
    rgb = cv2.remap(raw, coordinates[..., 0], coordinates[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255)), rgb


def process_image(row, calibration, old, output, backend, core, frontend, record, save):
    np, Image, torch = backend.np, backend.Image, backend.torch
    import cv2
    started = time.perf_counter()
    result = dict(row, status='started', phase='decode', views=[], timings={})
    arrays = {}
    path = frontend.safe_input(old.inputs, row['image_id'])
    try:
        with Image.open(path) as im:
            require(im.format in ('JPEG', 'PNG') and im.size == (calibration['width'], calibration['height'])
                    and im.getexif().get(274, 1) == 1, 'source image dimensions/orientation')
            raw = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
        result['original_rgb_sha256'] = frontend.raw_sha(raw)
        result['original_encoded_sha256'] = sha(path)
        result['timings']['decode_seconds'] = time.perf_counter()-started
        K = np.array([[calibration['fx'], 0, calibration['cx']],
                      [0, calibration['fy'], calibration['cy']], [0, 0, 1]], dtype=np.float64)
        geometry = core.geometry(K, (calibration['width'], calibration['height']))
        Kc, size, rotations = geometry['Kc'], geometry['size'], geometry['rotations']
        result['geometry'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in geometry.items()}
        result['geometry']['K_original'] = K.tolist()
        fov = math.degrees(2*math.atan(size[0]/(2*float(Kc[0, 0]))))
        result['fov_x_degrees'] = fov
        depths, masks, arrays = [], [], {}
        for index, (name, rotation) in enumerate(zip(VIEW_NAMES, rotations)):
            result['phase'] = name
            record['stage'] = row['image_id'] + ':' + name
            save()
            t = time.perf_counter()
            xy = core.remap_coordinates(K, Kc, rotation, size)
            image, rgb = remap_rgb(raw, xy, cv2, np)
            view = dict(name=name, index=index, rgb_sha256=frontend.raw_sha(rgb),
                        model_input_sha256=frontend.raw_sha(image),
                        remap_float64_sha256=frontend.raw_sha(xy),
                        remap_seconds=time.perf_counter()-t, fov_x_degrees=fov)
            result['views'].append(view)
            t = time.perf_counter()
            record['depth_forwards_attempted'] += 1
            save()
            with torch.inference_mode():
                prediction = backend.moge.infer(backend.tensor(image),
                    **dict(frontend.q.MOGE_OPTIONS, fov_x=fov))
            view['inference_seconds'] = time.perf_counter()-t
            raw_prediction = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
            for key in ('depth', 'mask', 'intrinsics'):
                if key in raw_prediction:
                    arrays['view_%d_%s' % (index, key)] = raw_prediction[key].copy()
            frontend.validate_depth(raw_prediction, image.shape[:2], np)
            check_intrinsics(raw_prediction['intrinsics'], Kc, size, np)
            record['depth_forwards_completed'] += 1
            depths.append(arrays['view_%d_depth' % index]); masks.append(arrays['view_%d_mask' % index])
            del prediction, raw_prediction, xy, rgb, image
        result['phase'] = 'cache'
        result['arrays'] = frontend.q.save_arrays(output / ('image-%03d.npz' % row['index']), arrays, np)
        result['phase'] = 'cross_axis_analysis'
        t = time.perf_counter()
        result['analysis'] = core.analyze_image(np.stack(depths), np.stack(masks), Kc, rotations)
        result['timings']['analysis_seconds'] = time.perf_counter()-t
        result.update(status='completed', phase='complete')
        return result
    except BaseException as exc:
        result.update(status='failed', failed_phase=result['phase'],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        if arrays and 'arrays' not in result:
            result['partial_arrays'] = frontend.q.save_arrays(
                output / ('image-%03d-partial.npz' % row['index']), arrays, np)
        raise
    finally:
        result['timings']['total_seconds'] = time.perf_counter()-started
        write_new(output / ('image-%03d.json' % row['index']), result)


def seal_outputs(output, rows, source_hashes):
    items = []
    for row in rows:
        for suffix in ('.json', '.npz'):
            path = output / ('image-%03d' % row['index'] + suffix)
            items.append(dict(filename=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    seal = dict(schema='evr0024.probe-seal.v1', image_count=27, depth_forward_count=135,
                source_sha256=source_hashes, outputs=items)
    write_new(output / 'seal.json', seal)
    return seal


def read_sealed_analyses(output, seal):
    for item in seal['outputs']:
        p = output / item['filename']
        require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'sealed output changed')
    rows = [json.loads((output / item['filename']).read_text())
            for item in seal['outputs'] if item['filename'].endswith('.json')]
    require(len(rows) == 27 and all(r['status'] == 'completed' for r in rows), 'incomplete sealed analyses')
    return rows


def run(args):
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), 'fresh output required')
    require(not args.output_dir.resolve().is_relative_to(ROOT), 'private outputs outside public repository')
    args.output_dir.mkdir(parents=True)
    began = time.perf_counter()
    record = dict(schema='evr0024.probe.v1', status='started', eligible=False, stage='verification',
        expected_commit=args.expected_commit, input_freeze=INPUT_FREEZE, model_freeze=MODEL_FREEZE,
        original_frontend_supervision_sha256=FRONTEND_OUTER_SHA,
        timings={}, warnings=[], offline_attempts=[], compatibility_probes=[], local_hub_calls=[],
        depth_forwards_attempted=0, depth_forwards_completed=0, feature_forwards=0,
        matcher_forwards=0, native_calls=0, gt_read=False, images=[])
    def save():
        pending = args.output_dir / 'receipt.pending'
        pending.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
        os.replace(pending, args.output_dir / 'receipt.json')
    save()
    try:
        frontend = load_module('_evr0024_frozen_frontend', ROOT / 'experiments/0023-mapfree/frontend.py')
        sources, old_sources = verified_sources(frontend, args.expected_commit)
        record['source_sha256'], record['input_source_sha256'] = sources, old_sources
        old = original_args(args.original_frontend_supervision)
        q = frontend.q
        qa = frontend.qualification_args(old)
        frozen = q.frozen_bindings(qa)
        files, _, wheels, bindings = q.verify_files(qa, frozen)
        record['frontend_bindings'] = dict(bindings, **frontend.verify_qualification(old, bindings, frozen))
        record['input_acquisition_bindings'] = frontend.acquisition.verify_completed_chain(old)
        plan, input_hashes = frontend.verify_inputs(old)
        record['input_sha256'] = input_hashes
        selected = selected_images(plan)
        selected = [dict(row, index=i) for i, row in enumerate(selected)]
        record['images'] = [dict(row, status='not_attempted') for row in selected]
        calibrations = {}
        for scene in plan['scenes']:
            names = [r['image_id'] for r in selected if r['scene'] == scene['scene']]
            paths = ['/'.join(n.split('/')[-2:]) for n in names]
            kp = frontend.safe_input(old.inputs, scene['intrinsics_member']['filename'])
            values = frontend.parse_intrinsics(kp.read_text(), paths)
            calibrations.update({name: values[short] for name, short in zip(names, paths)})
        record['timings']['verification_seconds'] = time.perf_counter()-began
        record['settings'] = dict(moge=dict(q.MOGE_OPTIONS, fov_x='explicit canonical per-image camera'),
            views=list(VIEW_NAMES), tokens=1200, cpu_threads=4, protocol_sha256=sha(PROTOCOL))
        record['stage'] = 'model_loading'; save()
        backend = frontend.Backend(old, files, wheels, record)
        core = load_module('_evr0024_math_core', HERE / 'math_core.py')
        record['offline_guard_scope'] = 'Python network/process/input audit events; not an OS native-syscall sandbox'
        with q.capture_warnings(record):
            for row in selected:
                result = process_image(row, calibrations[row['image_id']], old, args.output_dir,
                                       backend, core, frontend, record, save)
                record['images'][row['index']] = dict(row, status=result['status'],
                    json_sha256=sha(args.output_dir / ('image-%03d.json' % row['index'])),
                    usable_offset=result['analysis']['usable_offset'])
                save()
        record['stage'] = 'postcheck'; save()
        require(record['depth_forwards_attempted'] == record['depth_forwards_completed'] == 135,
                'not exactly 135 depth forwards')
        require(not record['offline_attempts'], 'offline violation')
        _, final_hashes = frontend.verify_inputs(old)
        require(final_hashes == input_hashes, 'input bytes changed')
        require(frontend.acquisition.verify_completed_chain(old, verified_bindings=old_sources)
                == record['input_acquisition_bindings'], 'acquisition chain changed')
        _, _, _, end_bindings = q.verify_files(qa, frozen)
        require(end_bindings == bindings, 'model/runtime bytes changed')
        require(sha(args.original_frontend_supervision) == FRONTEND_OUTER_SHA, 'original supervision changed')
        require(all(sha(ROOT / path) == digest for path, digest in {**frozen, **old_sources, **sources}.items()),
                'frozen source changed')
        seal = seal_outputs(args.output_dir, selected, {**sources, **old_sources, **frozen})
        record['seal_sha256'] = sha(args.output_dir / 'seal.json')
        summary = core.summarize(read_sealed_analyses(args.output_dir, seal))
        diagnostic = dict(schema='evr0024.cross-axis-diagnostic.v1', prediction_seal_sha256=record['seal_sha256'],
                          summary=summary, interpretation='Necessary predictor-model transfer test only; no depth/pose accuracy result')
        write_new(args.output_dir / 'diagnostic.json', diagnostic)
        record['diagnostic_sha256'] = sha(args.output_dir / 'diagnostic.json')
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
    parser.add_argument('--original-frontend-supervision', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ('status', 'eligible', 'wall_seconds', 'depth_forwards_completed')}))
    return 0 if result['eligible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
