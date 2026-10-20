"""Fixed27-image corrections and24-query known-control opportunity study.

Consumes observations only. No evaluation/ground-truth module is imported.
Every unavailable method retains the query and all native intents are durable.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import common as c


def admit_frontend(job, job_path, commit, sources):
    root = Path(job['raw_frontend'])
    outer = c.verify_outer(job['raw_frontend_outer'],root/'seal.json',commit,'frontend.py',job_path)
    seal = c.read(root/'seal.json')
    c.require(seal['schema'] == 'evr0028.frontend-seal.v1' and seal['status'] == 'completed'
        and seal['expected_commit'] == commit and seal['source_sha256'] == sources
        and seal['job_sha256'] == c.sha(job_path) and seal['image_count'] == 27
        and seal['query_count'] == 24 and seal['raw_forward_count'] == 27
        and seal['ground_truth_access'] is False, 'incomplete/mismatched raw frontend')
    for name,digest in seal['files'].items():
        c.require(Path(name).name == name and not (root/name).is_symlink()
                  and c.sha(root/name) == digest, 'raw frontend file changed')
    c.require({p.name for p in root.iterdir()} == set(seal['files'])|{'seal.json'},
              'raw frontend inventory changed')
    receipt = c.read(root/'receipt.json')
    c.require(receipt['status'] == 'completed' and receipt['raw_forwards_attempted'] ==
              receipt['raw_forwards_completed'] == len(receipt['images']) == 27
              and c.sha(root/'receipt.json') == seal['receipt_sha256'], 'raw frontend receipt')
    return receipt,dict(outer=outer,seal_sha256=c.sha(root/'seal.json'))


def runtime(frontend, fa, record, old, job):
    """Load only the existing arithmetic dependencies; no model construction."""
    for k in c.THREAD_KEYS:
        os.environ[k] = '4'
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      XFORMERS_DISABLED='1',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(fa.installation/'runtime')]+[
        str(fa.installation/'sources'/k) for k in frontend.q.PINS]
    native_root = Path(job['native_runtime'])
    record['native_binding'] = old.native_binding(native_root,frontend.q)
    c.require(c.sha(job['native_library']) == job['native_library_sha256']
              and c.sha(job['native_receipt']) == job['native_receipt_sha256'], 'native TC binding')
    sys.path.insert(0,str(native_root))
    frontend.q.install_guard(record['offline_attempts'],record['compatibility_probes'])
    record['typing_compatibility'] = frontend.q.prepare_utils3d_typing(fa.installation)
    import torch
    import numpy as np
    import scipy
    import poselib
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    c.require(np.__version__ == '2.0.2' and scipy.__version__ == '1.13.1', 'qualified numeric runtime')
    c.require(Path(poselib.__file__).resolve().is_relative_to(native_root.resolve()), 'native import origin')
    bridge = c.module('_evr0028_native_bridge',c.ROOT/'experiments/0026-calibration-moments/native_bridge/bridge.py')
    refiner = bridge.NativeRefiner(job['native_library'],job['native_receipt'])
    record['runtime'] = dict(numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
                             poselib=poselib.__version__,tc_library_sha256=refiner.library_sha256)
    return np,torch,poselib,refiner


def normalized_map(depth, mask, np, median=None):
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.full(depth.shape,np.nan),None
    if median is None:
        median = float(np.median(depth[valid].astype(np.float64)))
    c.require(np.isfinite(median) and median > 0, 'invalid depth median')
    return depth.astype(np.float64)/median,median


def prepare_image(row, arrays, original, f, post, provider, offset, pose, np, torch):
    start = time.perf_counter()
    K,xy = arrays['K_resized'],original['xy']
    prepared = offset.prepare(arrays['C_depth'],arrays['C_normal'],arrays['C_mask'],K)
    meta = dict(image_index=row['image_index'],filename=row['filename'],
                calibration=offset.describe(prepared),corrections={},medians={},samples={})
    maps,masks = {},{}
    for mode in ('A','B','C'):
        maps[mode],meta['medians'][mode] = normalized_map(arrays[mode+'_depth'],arrays[mode+'_mask'],np,
            prepared['median_raw'] if mode == 'C' else None)
        masks[mode] = arrays[mode+'_mask']
    for method in ('LS','IRLS','ANG'):
        fit = offset.solve(prepared,method)
        meta['corrections'][method] = fit
        corrected = offset.corrected_depth(prepared,fit)
        maps[method] = np.full(maps['C'].shape,np.nan) if corrected is None else corrected
        masks[method] = masks['C']
    before = time.perf_counter()
    if prepared['median_raw'] is None:
        full = dict(status='numerical_failure',reason='no_original_positive_depth')
    else:
        full = post.refine_with_normal(maps['C'],arrays['C_normal'],K,provider=provider)
    meta['full'] = {k:v for k,v in full.items() if not isinstance(v,np.ndarray) and not torch.is_tensor(v)}
    meta['full']['seconds'] = time.perf_counter()-before
    if full['status'] == 'helper_error':
        # The caller persists this metadata before escalating the technical error.
        return None,meta
    full_ok = full['status'] == 'completed'
    maps['FULL'] = (full['depth'].detach().cpu().numpy().copy() if torch.is_tensor(full.get('depth'))
                    else np.asarray(full['depth']).copy()) if full_ok else np.full(maps['C'].shape,np.nan)
    masks['FULL'] = masks['C']
    sampled = dict(xy=xy.copy(),K=K.copy())
    for method in ('A','B','C','LS','IRLS','ANG','FULL'):
        got = f.sample_depth(maps[method],masks[method],xy,np)
        # Do NOT use the helper's second normalization: all corrections retain C units.
        sampled['depth_'+method] = got['depth_sample_raw']
        sampled['valid_'+method] = got['depth_sample_valid']
        meta['samples'][method] = dict(valid_count=int(got['depth_sample_valid'].sum()),count=len(xy))
    normal = pose.sample_normal(arrays['C_normal'],arrays['C_mask'],xy)
    c.require(normal['status'] == 'completed' and normal['query_count'] == len(xy)
              and len(normal['samples']) == len(xy), 'normal sampler schema')
    sampled['normal_valid'] = np.asarray([
        value['status'] == 'completed' for value in normal['samples']],dtype=bool)
    sampled['normal'] = np.asarray([
        value['normal'] if value['status'] == 'completed' else [np.nan]*3
        for value in normal['samples']],dtype=np.float64).reshape((-1,3))
    # Preserve each rejection reason, including interpolation cancellations.
    meta['normal_sampling'] = normal
    sampled['original_valid'] = sampled['valid_A'] & sampled['valid_B'] & sampled['valid_C']
    meta.update(status='completed',comparator_valid=full_ok,seconds=time.perf_counter()-start)
    return sampled,meta


def run(args):
    sources = c.verify_source(args.expected_commit)
    job = c.read(args.job)
    c.require(job['expected_commit'] == args.expected_commit,'job freeze')
    out = Path(args.output)
    c.require(not out.resolve().is_relative_to(c.ROOT),'private output required')
    out.mkdir(parents=True,exist_ok=False)
    record = dict(schema='evr0028.prediction.v1',status='started',stage='binding',
        expected_commit=args.expected_commit,source_sha256=sources,job_sha256=c.sha(args.job),
        ground_truth_access=False,images=[],queries=[],offline_attempts=[],compatibility_probes=[])
    started = time.perf_counter()
    def save():
        c.write(out/'receipt.json',record,replace=True)
    save()
    try:
        raw_receipt,raw_binding = admit_frontend(job,args.job,args.expected_commit,sources)
        old,f,fa,prior,old_outer,frozen,cache,verified = c.original_frontend(job)
        images,edges = c.selected_layout(verified)
        c.require([r['filename'] for r in raw_receipt['images']] == [r['filename'] for r in images],
                  'new/old image identity differs')
        record.update(raw_frontend_binding=raw_binding,old_frontend_binding=old_outer,
                      source_input_files=verified['file_sha256'])
        np,torch,backend,refiner = runtime(f,fa,record,old,job)
        import normal_offset as offset
        import point_postprocess as post
        import pose_controls as pose
        import study_core as study
        provider = post.with_window_compatibility(post.load_provider())
        cache_root = Path(job['cache']); raw_root = Path(job['raw_frontend'])
        sample_by_name,meta_by_name = {},{}
        for original_row,raw_row in zip(images,raw_receipt['images']):
            record['stage'] = 'image:'+original_row['filename']; save()
            fields = {'xy','K_original','K_resized','image_size_original','image_size_resized'}
            original = cache.read_arrays(cache_root,original_row,fields)
            arrays = c.load_arrays(raw_root,raw_row['processed'])
            c.require(np.array_equal(arrays['K_resized'],original['K_resized']), 'camera changed')
            sample,meta = prepare_image(original_row,arrays,original,f,post,provider,offset,pose,np,torch)
            record['images'].append(meta); save()
            c.require(sample is not None, 'normal helper technical failure')
            meta['samples_file'] = c.save_arrays(out/('samples-%03d.npz'%original_row['image_index']),sample)
            name = original_row['filename']
            sample_by_name[name] = sample
            meta_by_name[name] = dict(meta,width=int(original['image_size_resized'][0]),
                                      height=int(original['image_size_resized'][1]))
            c.write(out/('image-%03d.json'%original_row['image_index']),meta)
            save()
        comparator_valid = all(r['comparator_valid'] for r in record['images'])
        for index,row in enumerate(edges):
            record['stage'] = 'query:%02d'%index; save()
            edge = cache.read_arrays(cache_root,row,{'match_ids','match_scores','xy0','xy1'})
            s,t = row['source'],row['target']
            im1,im2 = sample_by_name[s],sample_by_name[t]
            m1,m2 = meta_by_name[s],meta_by_name[t]
            ids = edge['match_ids']
            c.require(ids.ndim == 2 and ids.shape[1] == 2 and ids.dtype.kind in 'iu'
                and (ids >= 0).all() and (ids[:,0] < len(im1['xy'])).all()
                and (ids[:,1] < len(im2['xy'])).all(), 'canonical match IDs')
            c.require(np.array_equal(edge['xy0'],im1['xy'][ids[:,0]])
                and np.array_equal(edge['xy1'],im2['xy'][ids[:,1]]), 'edge canonical pixels changed')
            selected = cache.select_matches(ids,edge['match_scores'],im1['xy'],m1['width'],m1['height'],256)
            ii,jj = ids[selected,0],ids[selected,1]
            pair = dict(xy1=im1['xy'][ii],xy2=im2['xy'][jj],K1=im1['K'],K2=im2['K'],
                width1=m1['width'],height1=m1['height'],width2=m2['width'],height2=m2['height'],
                depths1={k:im1['depth_'+k][ii] for k in ('A','B','C','LS','IRLS','ANG','FULL')},
                depths2={k:im2['depth_'+k][jj] for k in ('A','B','C','LS','IRLS','ANG','FULL')},
                original_valid1=im1['original_valid'][ii],original_valid2=im2['original_valid'][jj],
                normal1=im1['normal'][ii],normal2=im2['normal'][jj],
                normal_valid1=im1['normal_valid'][ii],normal_valid2=im2['normal_valid'][jj],
                correction1=m1['corrections'],correction2=m2['corrections'],
                comparator_valid=m1['comparator_valid'] and m2['comparator_valid'])
            flat = {k:v for k,v in pair.items() if isinstance(v,np.ndarray)}
            for key in ('depths1','depths2'):
                flat.update({key+'_'+k:v for k,v in pair[key].items()})
            input_meta = c.save_arrays(out/('pair-%02d-input.npz'%index),flat)
            identity = dict(scene=row['scene'],query='B'+str(index%8+1),anchor=s,target=t,
                            edge_index=row['edge_index'],original_match_indices=selected.tolist(),
                            selected_match_ids=ids[selected].tolist(),pair_index=index)
            def emit(snapshot):
                c.write(out/('pair-%02d.partial.json'%index),dict(identity=identity,progress=snapshot),replace=True)
            result = study.process_pair(pair,backend,refiner,emit=emit)
            result.update(identity,inputs=input_meta)
            c.require(set(result['arms']) == set(c.ARMS), 'incomplete method population')
            comparator_valid = comparator_valid and result['comparator_valid']
            name = 'pair-%02d.json'%index
            c.write(out/name,result)
            record['queries'].append(dict(identity,filename=name,sha256=c.sha(out/name)))
            save()
        c.require(len(record['images']) == 27 and len(record['queries']) == 24,'incomplete predictions')
        c.require(not record['offline_attempts'],'blocked offline attempt')
        c.require(c.source_hashes() == sources and c.sha(args.job) == record['job_sha256'],
                  'source/job changed during prediction')
        record.update(status='completed',stage='completed',wall_seconds=time.perf_counter()-started,
                      comparator_valid=comparator_valid)
        save()
        seal = dict(schema='evr0028.prediction-seal.v1',status='completed',expected_commit=args.expected_commit,
            source_sha256=sources,job_sha256=c.sha(args.job),receipt_sha256=c.sha(out/'receipt.json'),
            query_count=24,image_count=27,arms=list(c.ARMS),comparator_valid=comparator_valid,
            files={p.name:c.sha(p) for p in sorted(out.iterdir()) if p.is_file()},ground_truth_access=False)
        c.write(out/'seal.json',seal)
        print(dict(status='completed',queries=24,comparator_valid=comparator_valid,
                   wall_seconds=record['wall_seconds']))
        return 0
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                      wall_seconds=time.perf_counter()-started)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--output',required=True,type=Path)
    sys.exit(run(parser.parse_args()))
