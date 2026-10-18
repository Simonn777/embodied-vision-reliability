"""Read-only published source/input and original evidence identity inventory.
Clean references are not opened, including for hashing, before scoring.
"""
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0008-training';CTL=Path(__file__).parent
MAIN='74f9330014aa8a13a8c55785992e115f045f2fa9';SOURCE='829688df750964d796417da569f0710453f62ca7';FREEZE='5680e6dae68ed102147ca89eeb115010cfe13687'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def main():
    assert not (ROOT/'antithetic-training-0008').exists(),'existing one-shot directory'
    assert not git('status','--porcelain'),'dirty checkout'
    assert git('rev-parse','HEAD').decode().strip()==git('rev-parse','origin/main').decode().strip()==MAIN
    rel='protocols/0008-training-freeze.json';f=json.loads((REPO/rel).read_text())
    assert sha(REPO/rel)=='f513c01f78e238ca0daef2427bb8ab756efafc7bd0965337ac073c8848dc83f5'
    assert git('show',FREEZE+':'+rel)==git('show',':'+rel)==(REPO/rel).read_bytes()
    assert f['source_commit']==SOURCE and len(f['files'])==13
    subprocess.run(['git','merge-base','--is-ancestor',FREEZE,MAIN],cwd=REPO,check=True)
    for name,h in f['files'].items():
        assert sha(REPO/name)==h
        for version in (SOURCE+':'+name,'HEAD:'+name,':'+name):assert hashlib.sha256(git('show',version)).hexdigest()==h,name
    assert sha(REPO/'scripts/check_antithetic_training.py')=='37b40d62dbfc849a1eb579af007d081a1d83de79b01309190c8064b1a9f2c91a'
    cfg=json.loads((REPO/'configs/0008-training.json').read_text());cache=ROOT/'conditional-residual-0006';inputs=[]
    for c in cfg['cases']:
        p=cache/c['input_relative'];assert p.is_file() and not p.is_symlink() and sha(p)==c['input_sha256']
        inputs.append({'relative':c['input_relative'],'sha256':c['input_sha256'],'bytes':p.stat().st_size})
    references={c['reference_relative']:c['reference_sha256'] for c in cfg['cases']};assert len(references)==4
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:a=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        names={Path(x.decode(errors='replace')).name for x in a if x}
        hit=names&{'run_antithetic_training.py','run_antithetic_gradient_screen.py','run_conditional_residual_stage0.py'}
        if hit:active.append({'pid':int(p.name),'scripts':sorted(hit)})
    assert not active,active
    tracked={n:sha(REPO/n) for n in git('ls-files','-z').decode().split('\0') if n}
    trees={}
    for name in ('conditional-residual-0006','antithetic-gradient-0007','tapnextpp-cache'):
        d=ROOT/name;assert d.is_dir();files={}
        for p in d.rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            relative=str(p.relative_to(d))
            if name=='conditional-residual-0006' and relative in references:continue
            files[relative]=sha(p)
        trees[name]=files
    markers={}
    for name in ('bootstapir-cache','tapnextpp-cache','tapnextpp-recovery'):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and not p.is_symlink() and ('started' in p.name or 'marker' in p.name):markers[str(p.relative_to(ROOT))]=sha(p)
    record={'status':'PASS','execution_main':MAIN,'source_commit':SOURCE,'freeze_commit':FREEZE,'canonical_sha256':sha(REPO/rel),'frozen_files':f['files'],'noisy_inputs':inputs,'references_not_opened_expected_hashes_only':references,'tracked':tracked,'historical_trees':trees,'markers':markers,'active_processes':active,'python':sys.version,'platform':platform.platform(),'cpu_model':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'new_asset_bytes':0,'asset_ceiling_bytes':7532690072}
    with (CTL/'pre-execution.json').open('x') as out:json.dump(record,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':'PASS','execution_main':MAIN,'required_files':len(f['files']),'noisy_input_hashes':len(inputs),'references_opened':0,'tracked_files':len(tracked),'prior_result_files':sum(n.startswith('results/') for n in tracked),'historical_tree_counts':{k:len(v) for k,v in trees.items()},'prior_markers':len(markers),'baseline_sha256':sha(CTL/'pre-execution.json')}))
if __name__=='__main__':main()
