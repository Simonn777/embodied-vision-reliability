"""Read-only Git/cache identity and preservation baseline. No array decoding."""
import hashlib,json,os,subprocess,sys,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; REPO=ROOT/'research-0007-gradient';CACHE=ROOT/'conditional-residual-0006'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def main():
    assert not (ROOT/'antithetic-gradient-0007').exists(),'existing one-shot output'
    assert not git('status','--porcelain'),'dirty checkout'
    expected='ca7fad653af31f93d770ede11a7f7ffdb5d12ec0'
    assert git('rev-parse','HEAD').decode().strip()==expected
    assert git('rev-parse','origin/main').decode().strip()==expected
    freeze='dd7c84c9afb774e2d411cf815aa17f953409d6e6';rel='protocols/0007-gradient-freeze.json'
    assert sha(REPO/rel)=='f6babe8162b6cc3abe2a26f018f9c747c486112b16698b08424dc942d4b4048a'
    assert git('show',freeze+':'+rel)==(REPO/rel).read_bytes()
    subprocess.run(['git','merge-base','--is-ancestor',freeze,expected],cwd=REPO,check=True)
    f=json.loads((REPO/rel).read_text());assert len(f['files'])==8
    for name,h in f['files'].items():
        for version in (':'+name,'HEAD:'+name,f['source_commit']+':'+name):assert hashlib.sha256(git('show',version)).hexdigest()==h
        assert sha(REPO/name)==h
    cfg=json.loads((REPO/'configs/0007-gradient-screen.json').read_text());cache=[]
    for item in cfg['cases']:
        for kind in ('input','weights','receipt'):
            p=CACHE/item[kind+'_relative'];assert p.is_file() and not p.is_symlink()
            h=sha(p);assert h==item[kind+'_sha256'];cache.append({'relative':item[kind+'_relative'],'sha256':h,'bytes':p.stat().st_size})
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        names={Path(x.decode(errors='replace')).name for x in args if x}
        hits=names & {'run_antithetic_gradient_screen.py','run_conditional_residual_stage0.py','score_conditional_residual_stage0.py','run_tapnext_recovered_capability.py'}
        if hits:active.append({'pid':int(p.name),'scripts':sorted(hits)})
    assert not active,active
    files={name:sha(REPO/name) for name in git('ls-files','-z').decode().split('\0') if name}
    # Historical failed cache is hashed as bytes only; no scientific arrays decoded.
    old=ROOT/'tapnextpp-cache'
    oldfiles={str(p.relative_to(old)):sha(p) for p in old.rglob('*') if p.is_file() and not p.is_symlink()}
    markers={}
    for d in ('bootstapir-cache','tapnextpp-cache','tapnextpp-recovery'):
        folder=ROOT/d
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file() and not p.is_symlink() and ('started' in p.name or 'marker' in p.name):markers[str(p.relative_to(ROOT))]=sha(p)
    result={'status':'PASS','execution_main':expected,'freeze_commit':freeze,'source_commit':f['source_commit'],'freeze_sha256':sha(REPO/rel),'frozen_files':f['files'],'cache_inputs':cache,'active_scientific_processes':active,'tracked':files,'old_failed_cache':oldfiles,'old_markers':markers,'python':sys.version,'platform':platform.platform(),'numeric_array_decodes':0,'new_asset_bytes':0,'asset_download_ceiling_bytes':7532690072}
    target=Path(__file__).parent/'pre-execution.json'
    with target.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':'PASS','tracked_count':len(files),'failed_cache_files':len(oldfiles),'old_marker_count':len(markers),'frozen_inputs':len(f['files']),'cached_hashes':len(cache),'pre_execution_sha256':sha(target)}))
if __name__=='__main__':main()
