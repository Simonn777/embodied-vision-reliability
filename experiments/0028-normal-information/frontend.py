"""One raw forward per fixed development image; no fitting or GT access.

Reuse original RGB/feature provenance. A/B/C share exactly one learned output.
The output is private, and the final seal requires all27 image records.
"""
import argparse
from pathlib import Path
import sys
import time
import common as c


def as_numpy(value, np):
    if hasattr(value,'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def calibration(row):
    k = row['input']['authored_K_original']
    h,w,_ = row['input']['original_shape_hwc']
    return dict(fx=k[0][0],fy=k[1][1],cx=k[0][2],cy=k[1][2],width=w,height=h)


def run(args):
    sources = c.verify_source(args.expected_commit)
    job = c.read(args.job)
    c.require(job['expected_commit'] == args.expected_commit, 'job freeze')
    out = Path(args.output)
    c.require(not out.resolve().is_relative_to(c.ROOT), 'private output required')
    out.mkdir(parents=True,exist_ok=False)
    record = dict(schema='evr0028.frontend.v1',status='started',stage='binding',
        expected_commit=args.expected_commit,source_sha256=sources,job_sha256=c.sha(args.job),
        ground_truth_access=False,raw_forwards_attempted=0,raw_forwards_completed=0,
        images=[],timings={},offline_attempts=[],compatibility_probes=[],warnings=[],local_hub_calls=[])
    start = time.perf_counter()
    def save():
        c.write(out/'receipt.json',record,replace=True)
    save()
    try:
        old,f,fa,prior,outer,frozen,cache,verified = c.original_frontend(job)
        rows,edges = c.selected_layout(verified)
        qa = f.qualification_args(fa)
        files,_,wheels,bindings = f.q.verify_files(qa,frozen)
        record.update(input_bindings=dict(frontend=outer,plan_sha256=c.PLAN_SHA,
            model=bindings,cache_file_sha256=verified['file_sha256']),
            selected_images=[r['filename'] for r in rows],
            selected_edges=[dict(scene=e['scene'],source=e['source'],target=e['target'],
                                 edge_index=e['edge_index']) for e in edges])
        record['stage'] = 'load_qualified_backend'; save()
        # Imports can emit the already observed local SSL compatibility warning.
        # Retain it in the same warning ledger used by the qualified loader.
        with f.q.capture_warnings(record):
            backend = f.Backend(fa,files,wheels,record)
        np,torch = backend.np,backend.torch
        import point_postprocess as post
        record['timings']['binding_and_load_seconds'] = time.perf_counter()-start
        for row in rows:
            image_start = time.perf_counter()
            r = dict(image_index=row['image_index'],filename=row['filename'],status='started',stage='decode')
            record['images'].append(r); save()
            pixels,geometry,meta = f.prepare_image(f.safe_input(fa.inputs,row['filename']),
                                                  calibration(row),np,backend.Image)
            for key in ('original_encoded_sha256','resized_rgb_uint8_sha256',
                        'resized_rgb_float32_sha256','authored_K_original','authored_K_resized'):
                c.require(meta[key] == row['input'][key], 'original RGB/preprocessing changed: '+key)
            r['input'] = meta
            r['stage'] = 'raw_forward'; record['raw_forwards_attempted'] += 1; save()
            began = time.perf_counter()
            with f.q.capture_warnings(record):
                with torch.inference_mode():
                    raw = backend.moge.forward(backend.tensor(pixels).unsqueeze(0),num_tokens=1200)
            r['forward_seconds'] = time.perf_counter()-began
            c.require(set(raw) == {'points','normal','mask','metric_scale'}, 'raw forward contract')
            raw_arrays = {'raw_'+k:as_numpy(v,np) for k,v in raw.items()}
            r['raw'] = c.save_arrays(out/('raw-%03d.npz'%row['image_index']),raw_arrays)
            record['raw_forwards_completed'] += 1
            r['stage'] = 'postprocess'; save()
            began = time.perf_counter()
            results = post.postprocess_all(raw,geometry['K_resized'])
            r['postprocess_seconds'] = time.perf_counter()-began
            arrays = dict(K_resized=geometry['K_resized'],K_original=geometry['K_original'])
            r['controls'] = {}
            for mode in ('A','B','C'):
                result = results[mode]
                metadata = {}
                for name,value in result.items():
                    if isinstance(value,np.ndarray) or torch.is_tensor(value):
                        arrays[mode+'_'+name] = as_numpy(value,np)
                    else:
                        metadata[name] = value
                r['controls'][mode] = metadata
                save()
                if result['status'] != 'completed':
                    r['partial_processed'] = c.save_arrays(out/('partial-processed-%03d.npz'%row['image_index']),arrays)
                    save()
                    raise ValueError('camera-control failure: '+mode)
                for name in ('depth','mask','normal'):
                    c.require(mode+'_'+name in arrays,'missing camera-control array')
            r['processed'] = c.save_arrays(out/('processed-%03d.npz'%row['image_index']),arrays)
            r.update(status='completed',stage='completed',total_seconds=time.perf_counter()-image_start)
            c.write(out/('image-%03d.json'%row['image_index']),r)
            save()
        c.require(record['raw_forwards_attempted'] == record['raw_forwards_completed'] == len(rows) == 27,
                  'incomplete raw forward population')
        c.require(not record['offline_attempts'],'blocked offline operation')
        c.require(c.source_hashes() == sources, 'source changed during frontend')
        c.require(c.sha(args.job) == record['job_sha256'], 'job changed during frontend')
        record.update(status='completed',stage='completed',wall_seconds=time.perf_counter()-start)
        save()
        seal = dict(schema='evr0028.frontend-seal.v1',status='completed',expected_commit=args.expected_commit,
            source_sha256=sources,job_sha256=c.sha(args.job),receipt_sha256=c.sha(out/'receipt.json'),
            image_count=27,query_count=24,raw_forward_count=27,
            files={p.name:c.sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            ground_truth_access=False)
        c.write(out/'seal.json',seal)
        print(c.serial(dict(status='completed',images=27,raw_forwards=27,
                            wall_seconds=record['wall_seconds'])))
        return 0
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                      wall_seconds=time.perf_counter()-start)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--output',required=True,type=Path)
    sys.exit(run(parser.parse_args()))
