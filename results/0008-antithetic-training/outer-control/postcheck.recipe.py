"""Saved bytes and original history preservation only; no model/array decoding."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0008-training';CTL=Path(__file__).parent;OUT=ROOT/'antithetic-training-0008'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def main():
    old=read(CTL/'pre-execution.json');assert not git('status','--porcelain')
    assert git('rev-parse','HEAD').decode().strip()==old['execution_main']
    for n,h in old['tracked'].items():assert sha(REPO/n)==h,n
    for tree,files in old['historical_trees'].items():
        for n,h in files.items():assert sha(ROOT/tree/n)==h,(tree,n)
        current={str(p.relative_to(ROOT/tree)) for p in (ROOT/tree).rglob('*') if p.is_file() and not p.is_symlink()}
        if tree=='conditional-residual-0006':current-=set(old['references_not_opened_expected_hashes_only'])
        assert current==set(files),tree
    for n,h in old['markers'].items():assert sha(ROOT/n)==h,n
    for n,h in old['references_not_opened_expected_hashes_only'].items():assert sha(ROOT/'conditional-residual-0006'/n)==h,n
    for n,h in old['frozen_files'].items():
        assert sha(REPO/n)==h
        for version in (':'+n,'HEAD:'+n,old['source_commit']+':'+n):assert hashlib.sha256(git('show',version)).hexdigest()==h,n
    canonical='protocols/0008-training-freeze.json';assert sha(REPO/canonical)==old['canonical_sha256']
    assert git('show',old['freeze_commit']+':'+canonical)==(REPO/canonical).read_bytes()
    labels=('preflight','frozen-verify','runtime-tests','independent-self-test','scientific-run','frozen-audit','frozen-score','independent-replay')
    exits={}
    for label in labels:
        e=read(CTL/(label+'.execution.json'));assert e['actual_exit']==e['raw_wait_status']==0,label
        for suffix,key in (('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')):assert sha(CTL/(label+suffix))==e[key],(label,suffix)
        exits[label]={k:e[k] for k in ('actual_exit','raw_wait_status','wall_seconds','peak_rss_kib')}
    start=read(OUT/'study.started.json');study=read(OUT/'study.completed.json')
    assert len(study['cases'])==32 and not (OUT/'study.failed.json').exists()
    assert start['source_commit']==study['source_commit']==old['source_commit']
    assert start['inputs']==study['inputs_after'] and start['freeze_sha256']==old['canonical_sha256']
    saved_hashes={};wallcaptures=[];steps=[]
    for row in study['cases']:
        name=row['case'];control=OUT/'control'/name;e=read(control.with_suffix('.execution.json'));rpath=OUT/'cases'/name/'receipt.json';r=read(rpath)
        assert row=={'case':name,**e} and e['actual_exit']==e['raw_wait_status']==0 and e['hard_timeout'] is False
        assert sha(rpath)==e['receipt_sha256'] and r['status']=='completed'
        for suffix,key in (('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')):assert sha(control.with_suffix(suffix))==e[key]
        for n,h in r['files'].items():
            p=OUT/'cases'/name/n;assert sha(p)==h;saved_hashes[str(p.relative_to(OUT))]=h
        wallcaptures.append(r['endpoints']['wall5s']['training']['training_elapsed_seconds']);steps.append(r['fit']['completed_updates'])
        assert r['endpoints']['updates500']['training']['step']==500
    assert len(saved_hashes)==128 and len(list(OUT.glob('cases/*/*-prediction.npy')))==64
    assert min(wallcaptures)>=5 and max(wallcaptures)<=5.05
    replay=read(CTL/'independent-replay.json');assert replay['status']=='PASS' and replay['scores_sha256']==sha(OUT/'scores.json')
    assert replay['checker_sha256']=='37b40d62dbfc849a1eb579af007d081a1d83de79b01309190c8064b1a9f2c91a'
    result={'status':'PASS','source_commit':old['source_commit'],'execution_main':old['execution_main'],'freeze_commit':old['freeze_commit'],'canonical_sha256':old['canonical_sha256'],'frozen_inputs_unchanged':len(old['frozen_files']),'tracked_files_preserved':len(old['tracked']),'prior_public_result_files_preserved':sum(n.startswith('results/') for n in old['tracked']),'historical_tree_counts_preserved':{k:len(v) for k,v in old['historical_trees'].items()},'old_markers_preserved':len(old['markers']),'pinned_references_verified_after_scoring':4,'references_opened_before_scoring_by_preflight':False,'trajectory_count':32,'prediction_count':64,'saved_output_hashes':saved_hashes,'wall_endpoint_capture_min_seconds':min(wallcaptures),'wall_endpoint_capture_max_seconds':max(wallcaptures),'trajectory_update_range':[min(steps),max(steps)],'actual_outer_records':exits,'scores_sha256':sha(OUT/'scores.json'),'independent_replay_sha256':sha(CTL/'independent-replay.json'),'postcheck_source_sha256':sha(__file__),'new_asset_bytes':0,'asset_ceiling_bytes':7532690072,'repository_unchanged':True,'publication_performed':False}
    with (CTL/'postcheck.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('saved_output_hashes','actual_outer_records')},sort_keys=True))
if __name__=='__main__':main()
