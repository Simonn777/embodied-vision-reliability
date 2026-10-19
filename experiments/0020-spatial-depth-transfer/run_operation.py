"""One exclusive, bounded outer receipt for a frozen 0020 operation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def snapshot(parent):
    sample=subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,rss='],capture_output=True,text=True,timeout=3,check=True)
    rows=[tuple(map(int,line.split())) for line in sample.stdout.splitlines() if len(line.split())==4]
    descendants={parent}
    while True:
        new=descendants|{pid for pid,ppid,_,_ in rows if ppid in descendants}
        if new==descendants:break
        descendants=new
    selected=[row for row in rows if row[0] in descendants]
    # The supervisor and isolated workers are session/group leaders. Never
    # signal an unrelated shared group containing the outer wrapper itself.
    groups={pgid for _,_,pgid,_ in selected if pgid in descendants}
    return selected,groups


def run(args):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head != args.expected_commit:
        raise ValueError('The working HEAD differs from the verified scientific freeze')
    subprocess.run(['git','diff','--quiet',args.expected_commit,'--'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--quiet',args.expected_commit,'--'],cwd=ROOT,check=True)
    relative = 'experiments/0020-spatial-depth-transfer/'
    command = [relative+('run_diagnostic.py' if args.mode == 'prediction' else 'evaluate_saved.py'),
               '--manifest','configs/0017-development-manifest.json',
               '--baseline-seal','results/0017-depth/predictions/seal.json']
    base = Path(args.output_dir)
    if not base.is_absolute():
        base = ROOT/base
    base.mkdir(parents=True,exist_ok=True)
    if args.mode == 'prediction':
        command += ['--data-dir',args.data_dir,'--output-dir',str(base/'predictions'),
                    '--dependency-runtime',args.dependency_runtime]
    else:
        command += ['--seal',str(base/'predictions/seal.json'),'--output',str(base/'evaluation.json')]
    env = dict(os.environ)
    env['PYTHONPATH'] = str((ROOT/args.dependency_runtime).resolve())
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    display = [part.replace(str(ROOT),'<repository>') for part in command]
    record = {'schema':'evr0020.outer-operation.v1','mode':args.mode,'status':'started',
              'command':['$PYTHON',*display],'python_version':platform.python_version(),'cwd':'repository root',
              'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_commit':head,
              'hard_timeout_seconds':600,'group_rss_limit_kib':8*1024*1024}
    receipt_path,log_path = (base/(args.mode+suffix) for suffix in ('-process.json','.log'))
    with receipt_path.open('x+') as receipt:
        def save():
            receipt.seek(0);json.dump(record,receipt,indent=2,allow_nan=False);receipt.write('\n')
            receipt.truncate();receipt.flush();os.fsync(receipt.fileno())
        save()
        began=time.perf_counter();child=None;code=None;termination=None;peak=0
        observed_groups=set();remaining_groups=[]
        with log_path.open('xb') as log:
            try:
                child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
                observed_groups={child.pid}
                while child.poll() is None:
                    rows,observed_groups=snapshot(child.pid)
                    peak=max(peak,sum(rss for _,_,_,rss in rows))
                    if peak>record['group_rss_limit_kib']:
                        termination='group_rss_limit';break
                    if time.perf_counter()-began>record['hard_timeout_seconds']:
                        termination='outer_timeout';break
                    time.sleep(.1)
            except BaseException as exc:
                termination=type(exc).__name__+': '+str(exc)
            finally:
                if child is not None:
                    if child.poll() is None or child.returncode != 0 or termination is not None:
                        # Stop new worker launches, refresh descendants, and kill
                        # their separate sessions as well as the supervisor.
                        try:os.killpg(child.pid,signal.SIGSTOP)
                        except ProcessLookupError:pass
                        try:
                            _,fresh_groups=snapshot(child.pid)
                            observed_groups |= fresh_groups
                        except (OSError,subprocess.SubprocessError):
                            pass
                        for group in observed_groups|{child.pid}:
                            try:os.killpg(group,signal.SIGKILL)
                            except ProcessLookupError:pass
                    code=child.wait(timeout=5)
                    if termination is not None:
                        # Grandchildren cannot be waitpid'ed here. Poll their
                        # process groups and report any non-reaped remainder.
                        until=time.monotonic()+5
                        while True:
                            remaining_groups=[]
                            for group in observed_groups-{child.pid}:
                                try:os.killpg(group,0);remaining_groups.append(group)
                                except ProcessLookupError:pass
                            if not remaining_groups or time.monotonic()>=until:break
                            time.sleep(.05)
                if time.perf_counter()-began>record['hard_timeout_seconds'] and termination is None:
                    termination='outer_timeout_at_exit'
                log.flush()
        record.update(status='completed' if code==0 and termination is None else 'failed',exit_code=code,
                      termination=termination,wall_seconds=time.perf_counter()-began,peak_group_rss_kib=peak,
                      remaining_process_groups_after_cleanup=remaining_groups,
                      log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest())
        save()
    print(json.dumps({k:record[k] for k in ('mode','status','exit_code','termination','wall_seconds','peak_group_rss_kib')}),flush=True)
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prediction','evaluation'),required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--data-dir',default='../0017-preparation/inputs')
    parser.add_argument('--dependency-runtime',default='../runtime')
    parser.add_argument('--output-dir',default='results/0020-spatial-depth-transfer')
    raise SystemExit(run(parser.parse_args()))
