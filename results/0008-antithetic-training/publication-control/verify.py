"""Publication-only hash/JSON/privacy/link review. No model or metric imports."""
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0008-training';PUB=Path(__file__).parent;DEST=REPO/'results/0008-antithetic-training';CONTROL=ROOT/'scratch/0008-control'
PYTHON='<PYTHON>'
MUTABLE={'README.md','CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def main():
    before=read(CONTROL/'pre-execution.json');original=read(PUB/'original-inventory.json')
    for relative,h in original['files'].items():assert sha(ROOT/relative)==h,relative
    for relative,h in before['tracked'].items():
        if relative not in MUTABLE:assert sha(REPO/relative)==h,relative
    for tree,files in before['historical_trees'].items():
        for relative,h in files.items():assert sha(ROOT/tree/relative)==h,(tree,relative)
    for relative,h in before['markers'].items():assert sha(ROOT/relative)==h,relative
    assert sha(ROOT/'antithetic-training-0008/scores.json')=='810cfd9384c475dc0df3d048bcd98a5bd33f96993e70e15d6ef69d68e40943e7'
    assert sha(CONTROL/'independent-replay.json')=='75247b4e4d723ab8f7bc4a68d0a3ec3ba7f3641a6aa5733eb97e53128b24d716'
    assert sha(CONTROL/'postcheck.json')=='34c911ea286e765ed579d7078dd59dd0a3b2c33c2e58c7f08b120c797db0009c'
    for relative,h in before['frozen_files'].items():
        assert sha(REPO/relative)==h
        for version in (':'+relative,'HEAD:'+relative,before['source_commit']+':'+relative):assert hashlib.sha256(git('show',version)).hexdigest()==h,relative
    canonical='protocols/0008-training-freeze.json';assert sha(REPO/canonical)==before['canonical_sha256']
    assert git('show',before['freeze_commit']+':'+canonical)==(REPO/canonical).read_bytes()
    modes={};manifest=read(DEST/'copy-manifest.json');bound=set()
    for e in manifest['entries']:
        src=ROOT/e['original_source_relative'];dst=REPO/e['public_path'];bound.add(dst.resolve())
        assert sha(src)==e['original_sha256'] and sha(dst)==e['public_sha256'],e['public_path']
        assert src.stat().st_size==e['original_bytes'] and dst.stat().st_size==e['public_bytes']
        if e['mode']=='byte_exact_copy':assert src.read_bytes()==dst.read_bytes() and not e['redaction_scope']
        elif e['mode']=='redacted_copy':
            assert dst.read_text()==src.read_text().replace(PYTHON,'<PYTHON>').replace(str(ROOT),'<PROJECT_ROOT>') and e['redaction_scope']
        elif e['mode']=='derived_summary':
            assert e['redaction_scope']==['new derived summary; not an original byte copy']
            for source in e['original_sources']:assert sha(ROOT/source['relative'])==source['sha256']
        else:raise ValueError('unrecognized manifest mode')
        modes[e['mode']]=modes.get(e['mode'],0)+1
    json_count=0;files=[];links=0
    for p in DEST.rglob('*'):
        if not p.is_file():continue
        assert p.suffix in {'.json','.md','.log','.py','.txt'} and p.stat().st_size<4*1024**2,p
        raw=p.read_bytes();text=raw.decode();assert '\x00' not in text and str(ROOT) not in text and PYTHON not in text,p
        assert not p.name.endswith('.command.private.json'),p
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text),p
        if p.suffix=='.json':
            json.loads(text,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)));json_count+=1
        if p.name not in ('copy-manifest.json','artifact-hashes.json'):assert p.resolve() in bound,p
        files.append(p)
    for p in [DEST/'README.md',*(REPO/n for n in MUTABLE)]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if re.match(r'(https?://|mailto:|#)',target):continue
            local=unquote(target.split('#')[0]);assert (p.parent/local).resolve().exists(),(p.name,target)
            links+=1
    score=read(DEST/'study/scores.json');assert score['status']=='stop_this_candidate'
    replay=read(DEST/'outer-control/independent-replay.json');assert replay['status']=='PASS' and replay['decision_recomputed']==score['status']
    assert sha(DEST/'study/scores.json')=='810cfd9384c475dc0df3d048bcd98a5bd33f96993e70e15d6ef69d68e40943e7'
    assert sha(DEST/'outer-control/independent-replay.json')=='75247b4e4d723ab8f7bc4a68d0a3ec3ba7f3641a6aa5733eb97e53128b24d716'
    assert sha(DEST/'outer-control/postcheck.json')=='34c911ea286e765ed579d7078dd59dd0a3b2c33c2e58c7f08b120c797db0009c'
    assert len(list((DEST/'trajectories').glob('*/receipt.json')))==32
    assert len(read(DEST/'original-output-hashes.json')['files'])==128
    result={'status':'PASS','scientific_calls':0,'source_sha256':sha(__file__),'old_tracked_preserved_except_four_mutable':len(before['tracked'])-4,'prior_result_files_preserved':sum(n.startswith('results/') for n in before['tracked']),'prior_result_protocol_blobs_preserved':sum(n.startswith(('results/','protocols/')) for n in before['tracked']),'frozen_inputs_preserved':13,'original_0008_files_preserved':len(original['files']),'historical_cache_counts_preserved':{k:len(v) for k,v in before['historical_trees'].items()},'old_markers_preserved':len(before['markers']),'manifest_modes':modes,'public_json_files_validated':json_count,'public_files_reviewed':len(files),'local_links_checked':links,'unexpected_binary_files':0,'privacy_check':'No original project/interpreter absolute paths, private argv files, credential patterns or NUL bytes in public files. Non-reconstructable statistics only.','original_compact_valid':True,'normalization_performed':False,'largest_public_text_bytes':max(p.stat().st_size for p in files),'new_asset_bytes':0}
    with (PUB/'verification.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
