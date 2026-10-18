"""Publish only text evidence; private argv and arrays are never copied."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0007-gradient';OUT=ROOT/'antithetic-gradient-0007';CTL=Path(__file__).parent;DEST=REPO/'results/0007-gradient-screen'
PYTHON='<PYTHON>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    assert not DEST.exists();DEST.mkdir()
    assert read(CTL/'integrity.json')['status']=='PASS'
    assert read(CTL/'independent-replay.stdout.log')['status']=='PASS'
    manifest=[]
    def record(src,dst,mode):
        manifest.append({'source':str(src.relative_to(ROOT)),'source_sha256':sha(src),'destination':str(dst.relative_to(REPO)),'destination_sha256':sha(dst),'mode':mode})
    def copy(src,rel):
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(src,dst);record(src,dst,'byte_exact_copy')
    def redact(src,rel):
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        text=src.read_text().replace(PYTHON,'<PYTHON>').replace(str(ROOT),'<PROJECT_ROOT>')
        with dst.open('x') as f:f.write(text)
        record(src,dst,'path_redaction_not_original_bytes')
    for p in sorted(OUT.glob('*.json')):copy(p,Path('study')/p.name)
    for p in sorted(OUT.glob('cases/*/*.json')):copy(p,Path('cases')/p.parent.name/p.name)
    for p in sorted((OUT/'control').iterdir()):
        if p.name.endswith('.command.private.json'):redact(p,Path('case-control')/p.name.replace('.command.private.json','.command.recipe.json'))
        elif p.suffix in ('.json','.log'):copy(p,Path('case-control')/p.name)
    for p in sorted(CTL.iterdir()):
        if p.suffix not in ('.json','.log','.py') or p.name in ('pre-execution.json','publication.json'):continue
        if p.name.endswith('.command.private.json'):redact(p,Path('outer-control')/p.name.replace('.command.private.json','.command.recipe.json'))
        elif str(ROOT) in p.read_text() or PYTHON in p.read_text():redact(p,Path('outer-control')/p.name)
        else:copy(p,Path('outer-control')/p.name)
    before=read(CTL/'pre-execution.json')
    for name,h in before['tracked'].items():assert sha(REPO/name)==h,name
    for name,h in before['old_failed_cache'].items():assert sha(ROOT/'tapnextpp-cache'/name)==h
    for name,h in before['old_markers'].items():assert sha(ROOT/name)==h
    for item in before['cache_inputs']:assert sha(ROOT/'conditional-residual-0006'/item['relative'])==item['sha256']
    compact={k:v for k,v in before.items() if k not in ('tracked','old_failed_cache')}
    compact.update(private_full_baseline_sha256=sha(CTL/'pre-execution.json'),tracked_baseline_count=len(before['tracked']),prior_result_files=sum(p.startswith('results/') for p in before['tracked']),failed_cache_file_count=len(before['old_failed_cache']),all_prior_bytes_preserved=True,baseline_storage='Full pre-execution path/hash baseline remains private; identities are tied to execution main. No old result arrays were decoded.')
    write(DEST/'pre-execution-summary.json',compact)
    write(DEST/'copy-manifest.json',{'schema':'0007-publication-copy-manifest-v1','entries':manifest,'private_file_policy':'No command.private.json or scientific array is published. Path-redacted recipes are explicitly transformed, their original SHA is retained. Byte-exact copies are identified individually.','private_source_prefix':'<PROJECT_ROOT>','python_recipe':'<PYTHON> is the existing fixed Python3.12.14 interpreter.'})
    sources=read(REPO/'protocols/0007-gradient-freeze.json')
    write(DEST/'freeze-bindings.json',{'execution_main':before['execution_main'],'canonical_freeze_commit':before['freeze_commit'],'canonical_freeze_path':'protocols/0007-gradient-freeze.json','canonical_freeze_sha256':before['freeze_sha256'],'source_commit':sources['source_commit'],'required_files':sources['files'],'input_hashes':before['cache_inputs'],'independent_checker_sha256':sha(CTL/'independent_checker.py'),'new_asset_bytes':0,'historical_asset_ceiling_bytes':7532690072,'raw_gradient_files':'Private at <PROJECT_ROOT>/antithetic-gradient-0007/cases/{case}/gradients.npy; file and array hashes are public in receipts.'})
    # All public outputs must be UTF8 text, no absolute project paths or credentials.
    for p in DEST.rglob('*'):
        if not p.is_file():continue
        text=p.read_text();assert str(ROOT) not in text and PYTHON not in text,p
        assert '\x00' not in text and '-----BEGIN PRIVATE KEY' not in text,p
    print(json.dumps({'status':'archived','files':sum(p.is_file() for p in DEST.rglob('*')),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'manifest_sha256':sha(DEST/'copy-manifest.json')}))
if __name__=='__main__':main()
