"""Read-only environment/Git/history preservation. No case generation or seeds."""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0009-stage0';RUNTIME=ROOT/'bootstapir-cache/runtime-fixed'
MAIN='cd144b90d337a172a892b56d87cd794524605753';SOURCE='4b11f1de4d6fcc9bef45c8eee824ff8e7241abc0';FREEZE='17f417efcccf85047e0935569f87f94b27705e82'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def main():
    assert not any((HERE/n).exists() for n in ('collection','supervision','evaluation','review.json')),'existing operation output'
    assert not git('status','--porcelain')
    assert git('rev-parse','HEAD').decode().strip()==git('rev-parse','origin/main').decode().strip()==MAIN
    receipt='protocols/0009-geometric-stage0-freeze.json';f=json.loads((REPO/receipt).read_text());assert f['source_commit']==SOURCE and len(f['files'])==15
    assert git('show',FREEZE+':'+receipt)==git('show',':'+receipt)==(REPO/receipt).read_bytes()
    subprocess.run(['git','merge-base','--is-ancestor',FREEZE,MAIN],cwd=REPO,check=True)
    for n,h in f['files'].items():
        assert sha(REPO/n)==h
        for v in (':'+n,'HEAD:'+n,SOURCE+':'+n):assert hashlib.sha256(git('show',v)).hexdigest()==h,n
    assert sha(REPO/'scripts/check_geometric_stage0.py')=='0cd4e5bbf8528212cd4c051a51f7cf8dd46ca89c0c69e933cf762da61513ea55'
    import numpy as np
    assert np.__version__=='2.0.2' and Path(np.__file__).resolve().is_relative_to(RUNTIME.resolve())
    assert sys.version.split()[0]=='3.12.14' and sys.platform=='linux'
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip();assert memory==str(8*1024**3),'8GiB cgroup required'
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        names={Path(a.decode(errors='replace')).name for a in args if a};hit=names&{'supervise_0009.py','run_geometric_stage0.py','run_antithetic_training.py','run_antithetic_gradient_screen.py'}
        if hit:active.append({'pid':int(p.name),'scripts':sorted(hit)})
    assert not active,active
    tracked={n:sha(REPO/n) for n in git('ls-files','-z').decode().split('\0') if n}
    trees={}
    for name in ('bootstapir-cache','tapnextpp-cache','tapnextpp-recovery','conditional-residual-0006','antithetic-gradient-0007','antithetic-training-0008','scratch/0009-runtime-qualification','scratch/0009-hpatches-directory'):
        d=ROOT/name;assert d.is_dir(),name
        trees[name]={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and not p.is_symlink()}
    result={'status':'PASS','execution_main':MAIN,'source_commit':SOURCE,'freeze_commit':FREEZE,'canonical_sha256':sha(REPO/receipt),'frozen_files':f['files'],'tracked':tracked,'historical_trees':trees,'python':sys.version,'python_executable':sys.executable,'numpy_version':np.__version__,'numpy_module_path':np.__file__,'numpy_module_entry_sha256':sha(np.__file__),'platform':platform.platform(),'cgroup_memory_max':memory,'cgroup_memory_current':Path('/sys/fs/cgroup/memory.current').read_text().strip(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_model':next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},'no_active_project_science':True,'prior_marker_search':'Initial metadata-only search: 3034 entries, depth<=4 in project/scratch and named0009 directories; no prior scientific0009 marker found. Qualification directories are not science.','new_asset_download_bytes':0,'existing_asset_accounting_gap':'Original ceiling record7532690072 plus measured206 metadata163286; 302 body unknown. No verified complete updated ceiling.'}
    with (HERE/'pre-execution.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':'PASS','main':MAIN,'frozen_inputs':15,'numpy_version':np.__version__,'numpy_path':np.__file__,'cgroup_memory_max':memory,'tracked_files':len(tracked),'prior_result_protocol_files':sum(n.startswith(('results/','protocols/')) for n in tracked),'cache_counts':{k:len(v) for k,v in trees.items()},'baseline_sha256':sha(HERE/'pre-execution.json'),'scientific_markers_created':False}))
if __name__=='__main__':main()
