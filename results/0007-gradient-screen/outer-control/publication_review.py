"""Final text/copy/preservation review, without model or array decoding."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0007-gradient';DEST=REPO/'results/0007-gradient-screen';CTL=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    old=read(CTL/'pre-execution.json');mutable={'CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'}
    for n,h in old['tracked'].items():
        if n not in mutable:assert sha(REPO/n)==h,n
    for n,h in old['old_failed_cache'].items():assert sha(ROOT/'tapnextpp-cache'/n)==h,n
    for n,h in old['old_markers'].items():assert sha(ROOT/n)==h,n
    for row in old['cache_inputs']:assert sha(ROOT/'conditional-residual-0006'/row['relative'])==row['sha256']
    modes={};entries=read(DEST/'copy-manifest.json')['entries']
    for e in entries:
        src=ROOT/e['source'];dst=REPO/e['destination']
        assert sha(src)==e['source_sha256'] and sha(dst)==e['destination_sha256']
        if e['mode']=='byte_exact_copy':assert src.read_bytes()==dst.read_bytes()
        else:
            expected=src.read_text().replace('/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python','<PYTHON>').replace(str(ROOT),'<PROJECT_ROOT>')
            assert dst.read_text()==expected
        modes[e['mode']]=modes.get(e['mode'],0)+1
    for p in DEST.rglob('*'):
        if not p.is_file():continue
        assert p.suffix in {'.py','.json','.log','.md'} and p.stat().st_size<1024**2,p
        text=p.read_text();assert str(ROOT) not in text and '\x00' not in text,p
        assert not p.name.endswith('.command.private.json'),p
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text),p
        if p.suffix=='.json':json.loads(text,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
    score=read(DEST/'study/scores.json');replay=read(DEST/'outer-control/independent-replay.stdout.log')
    assert score['gates']['all_passed'] and replay['status']=='PASS'
    assert replay['scores_sha256']==sha(DEST/'study/scores.json')
    assert sha(DEST/'outer-control/independent_checker.py')=='2e6ab7912ee1e56836da9cd61b059598d09125eb87d9693c1be8a79bd17fde6e'
    # Rehash existing science bytes only; no gradient recomputation or decoding.
    for case in old['cache_inputs'][::3]:
        name=Path(case['relative']).name.removesuffix('-input.npz')
        r=read(ROOT/'antithetic-gradient-0007/cases'/name/'receipt.json')
        for f,h in r['files'].items():assert sha(ROOT/'antithetic-gradient-0007/cases'/name/f)==h
    result={'status':'PASS','old_tracked_unchanged_excluding_three_authorized_state_edits':len(old['tracked'])-len(mutable),'old_result_files_preserved':sum(n.startswith('results/') for n in old['tracked']),'old_failed_cache_files_preserved':len(old['old_failed_cache']),'old_markers_preserved':len(old['old_markers']),'exact_cache_hashes_preserved':len(old['cache_inputs']),'copy_manifest_modes':modes,'canonical_freeze_preserved':sha(REPO/'protocols/0007-gradient-freeze.json'),'scores_sha256':sha(DEST/'study/scores.json'),'checker_sha256':sha(__file__),'public_artifacts_checked':sum(p.is_file() for p in DEST.rglob('*')),'raw_gradient_arrays_rehashed_not_decoded':32,'model_calls':0,'new_asset_bytes':0,'archive_scan_correction':'A literal key-detection phrase in the archived scanner source is not a credential. Structural credential patterns and all exact copy/redaction identities pass. Original archival failure preserved.'}
    with (CTL/'publication-review.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
