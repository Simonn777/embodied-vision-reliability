"""One frozen all-inlier initializer-noise falsifier. No real-data or model access."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / 'protocols/0025c-rs-initializer-noise.md'
LEGACY = ROOT / 'experiments/0025-rs-identifiability/check_manufactured.py'
OWNED = ('core.py', 'fixtures.py', 'run.py', 'audit_saved.py', 'supervise.py')
ARMS = ('D4', 'A2', 'D8', 'N8', 'A8')
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(safe(value), f, indent=2, allow_nan=False)
        f.write('\n')


def angle(t, truth):
    if t is None:
        return None
    t = np.asarray(t, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    den = float(np.linalg.norm(t)*np.linalg.norm(truth))
    if not np.all(np.isfinite(t)) or not np.isfinite(den) or den <= 0:
        return None
    unit=t/np.linalg.norm(t); reference=truth/np.linalg.norm(truth)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(unit,reference))),float(unit@reference)))


def metrics(rows):
    errors = np.array([r['error_degrees'] for r in rows], dtype=np.float64)
    valid = np.array([r['status'] == 'valid' for r in rows])
    return dict(attempted=len(rows), available=int(valid.sum()),
                useful_count=int(np.sum(valid & (errors <= 5.))),
                useful_yield=float(np.mean(valid & (errors <= 5.))),
                median_error_degrees=float(np.median(errors)),
                p90_error_degrees=float(np.quantile(errors,.9)),
                maximum_error_degrees=float(np.max(errors)),
                empirical_cdf_integral_auc5_percent=float(100*np.mean(np.maximum(0.,1-errors/5))),
                recall={str(q):float(np.mean(valid & (errors <= q))) for q in (1,2,5,10,20,90,180)})


def summarize(rows, technical_pass):
    grouped = {}
    for r in rows:
        grouped.setdefault((r['case'],r['regime'],r['arm']),[]).append(r)
    groups = [{**dict(zip(('case','regime','arm'),key)),**metrics(values)} for key,values in sorted(grouped.items())]
    decisions = []
    for case in ('G','N','C'):
        for regime in ('P','S'):
            m = next((v for v in groups if (v['case'],v['regime'],v['arm'])==(case,regime,'D4')),None)
            decisions.append(dict(case=case,regime=regime,attempted=None if m is None else m['attempted'],
                                  useful_count=None if m is None else m['useful_count'],
                                  pass_gate=bool(m and m['attempted']==1120 and m['useful_count']>=144)))
    headroom=[]
    for case in ('G','N','C'):
        d4=next((v for v in groups if (v['case'],v['regime'],v['arm'])==(case,'S','D4')),None)
        a2=next((v for v in groups if (v['case'],v['regime'],v['arm'])==(case,'S','A2')),None)
        ok=bool(d4 and a2 and d4['attempted']==1120 and a2['attempted']==448 and d4['useful_count']>10*a2['useful_count'])
        headroom.append(dict(case=case,regime='S',D4_attempted=None if d4 is None else d4['attempted'],D4_useful_count=None if d4 is None else d4['useful_count'],A2_attempted=None if a2 is None else a2['attempted'],A2_useful_count=None if a2 is None else a2['useful_count'],pass_gate=ok))
    headroom_pass=any(d['pass_gate'] for d in headroom)
    return dict(groups=groups,decisions=decisions,technical_pass=technical_pass,
                optimistic_A2_headroom=headroom,optimistic_A2_headroom_pass=headroom_pass,
                permitted_next='separately_frozen_robust_stage_only' if technical_pass and all(d['pass_gate'] for d in decisions) and headroom_pass else 'do_not_advance_fixed_four_point_initializer',
                utility_gate_pass=bool(technical_pass and all(d['pass_gate'] for d in decisions) and headroom_pass),
                missing_solution_error_degrees=180.,subset_independence=False,
                auc_definition='Exact integral of empirical error CDF on [0,5] / 5; mean max(0,1-error/5). No trapezoid interpolation across empirical jumps.',
                no_real_pose_gain=True,no_manuscript=True)


def offline_guard(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn','os.posix_spawnp','os.fork','os.exec'):
        raise RuntimeError('offline/no-child operation: '+event)


def run(args):
    # Verify Git source identity before enabling the no-subprocess audit hook.
    import supervise
    paths = [HERE/name for name in OWNED] + [PROTOCOL,LEGACY]
    bindings = supervise.verify_frozen(args.expected_commit,paths)
    if any(os.environ.get(k) != '4' for k in THREAD_KEYS):
        raise ValueError('must use the reviewed four-thread supervisor')
    output = Path(args.output_dir).resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('require new private output directory')
    output.mkdir(parents=True)
    began = time.perf_counter()
    receipt = dict(schema='evr0025c.noise.v1',status='started',expected_commit=args.expected_commit,
                   source_sha256=bindings,python=platform.python_version(),numpy=np.__version__,
                   thread_environment={k:os.environ[k] for k in THREAD_KEYS},
                   case_attempted=0,case_completed=0,solver_attempted=0,solver_completed=0,
                   model_calls=0,real_data_reads=0,network_access_permitted=False)
    save(output/'started.json',receipt)
    rows=[]; records=[]; technical_pass=True
    try:
        import core
        import fixtures
        sys.addaudithook(offline_guard)
        cohort = fixtures.generate()
        if len(cohort)!=132:
            raise ValueError('frozen cohort count must be132')
        expected = {(c,'clean',-1) for c in ('G','N','C','E')} | {(c,r,i) for c in ('G','N','C','E') for r in ('P','S') for i in range(16)}
        keys = [(item['case'],item['regime'],item['realization']) for item in cohort]
        if len(set(keys))!=132 or set(keys)!=expected:
            raise ValueError('frozen cohort keys differ')
        for record_index,item in enumerate(cohort):
            receipt['case_attempted']+=1
            data=item['data']; truth=item['truth']; outputs=[]
            clean_check=None
            if item['regime']=='clean':
                full=core.build(data)
                beta=np.r_[truth['t'],truth['v1'],truth['v2'],truth['b1'],truth['a2'],truth['b2']]
                discrepancy=float(np.max(np.abs(full['A']@beta-full['y'])))
                clean_check=dict(affine_truth_max_abs_residual=discrepancy,pass_gate=discrepancy<=1e-10)
                if not clean_check['pass_gate']:
                    raise ValueError('clean physical affine equation failed')
            tasks=[('D4',list(s)) for s in itertools.combinations(range(8),4)] + [('A2',list(s)) for s in itertools.combinations(range(8),2)] + [(a,list(range(8))) for a in ('D8','N8','A8')]
            for arm,indices in tasks:
                receipt['solver_attempted']+=1
                call_began=time.perf_counter()
                result=(core.solve_depth_free(data,indices) if arm=='N8' else core.solve_affine(data,indices,angular_only=(arm in ('A2','A8'))))
                elapsed=time.perf_counter()-call_began
                if item['regime']=='clean' and result.get('task_null'):
                    true_t=np.asarray(truth['t'])
                    tn=float(np.linalg.norm(true_t)); direction=true_t/tn
                    projection=(np.eye(3)-np.outer(direction,direction))/tn
                    diagnostic={}
                    for kind in ('numerical','weak'):
                        basis=np.asarray(result['task_null'][kind]['basis_physical'])
                        task=projection@basis[:3]
                        diagnostic[kind]=dict(map=task,frobenius=float(np.linalg.norm(task)))
                    result['clean_truth_task_null_diagnostic']=diagnostic
                receipt['solver_completed']+=1
                raw_angle=angle((result.get('parameters') or {}).get('t'),truth['t'])
                if result['status']=='valid' and raw_angle is None:
                    raise ValueError('core declared valid undefined translation')
                error=raw_angle if result['status']=='valid' else 180.
                row=dict(record_index=record_index,case=item['case'],regime=item['regime'],realization=item['realization'],
                         arm=arm,indices='-'.join(map(str,indices)),status=result['status'],reason=result.get('reason'),
                         raw_angle_degrees=raw_angle,error_degrees=error,rank=result.get('rank'),weak_rank=result.get('weak_rank'),solve_wall_seconds=elapsed)
                rows.append(row)
                outputs.append(dict(**row,solver=result))
            name='case-%03d.json'%record_index
            save(output/name,dict(schema='evr0025c.case.v1',input=item,clean_check=clean_check,solves=outputs))
            records.append(dict(path=name,sha256=digest(output/name)))
            receipt['case_completed']+=1
        if receipt['solver_completed']!=13332:
            raise ValueError('frozen solver count must be13332')
        aggregate=summarize(rows,technical_pass)
        save(output/'aggregate.json',aggregate)
        with (output/'all-solves.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        if any(digest(ROOT/path)!=sha for path,sha in bindings.items()):
            raise ValueError('frozen source mutated')
        records.extend(dict(path=name,sha256=digest(output/name)) for name in ('aggregate.json','all-solves.csv','started.json'))
        save(output/'seal.json',dict(schema='evr0025c.seal.v1',expected_commit=args.expected_commit,records=records))
        receipt.update(status='completed',wall_seconds=time.perf_counter()-began,seal_sha256=digest(output/'seal.json'),
                       utility_gate_pass=aggregate['utility_gate_pass'],permitted_next=aggregate['permitted_next'])
        save(output/'receipt.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('status','case_completed','solver_completed','utility_gate_pass','permitted_next')}))
        return 0
    except BaseException as error:
        receipt.update(status='failed',wall_seconds=time.perf_counter()-began,error_type=type(error).__name__,error=str(error).replace(str(ROOT),'$REPOSITORY').replace(str(ROOT.parent),'$PRIVATE'))
        save(output/'receipt.json',receipt)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--output-dir',required=True)
    return run(parser.parse_args())

if __name__=='__main__':
    raise SystemExit(main())
