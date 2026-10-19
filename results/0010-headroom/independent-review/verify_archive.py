"""Specific public archive + actual tool closure supplement; no accuracy recomputation."""
import argparse,hashlib,json,re,shlex
from pathlib import Path

def run(bundle,actual_path,arithmetic_path):
    checks=0
    def ck(v,msg):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(msg)
    def read(n):return json.loads((bundle/n).read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def identity(p,s):
        ck(p.is_file() and not p.is_symlink(),'regular '+str(p));ck(p.stat().st_size==s['bytes'] and sha(p)==s['sha256'],'identity '+str(p))
    m=read('copy-manifest.json');ck(m['file_count']==37 and m['total_original_bytes']==405907,'expected archive count')
    for n,s in m['files'].items():identity(bundle/n,s)
    ck(sum(s['bytes'] for s in m['files'].values())==405907,'sum archive bytes')
    ck({str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}==set(m['files'])|{'copy-manifest.json'},'exact member set')
    result=json.loads(arithmetic_path.read_text());ck(result['status']=='PASS_PUBLIC_SAVED_ARITHMETIC','prior independent arithmetic')
    ctl='scratch/0010-headroom-control/';out='scratch/0010-headroom-output/';sup='scratch/0010-headroom-supervision/'
    observations=json.loads(actual_path.read_text())['commands'];byid={c['id']:c for c in observations}
    ck(len(byid)==len(observations) and len(observations) in (8,14),'full14 or selected8 unique actual observations')
    failures={'exec-66fa4089-7135-4095-a10d-54f83e6e3a18':2,'exec-096d1412-8cf1-45c8-850e-3b069aeee4a8':1,'exec-31dda8a5-253a-427e-984e-23223b5bc238':1}
    ck({c['id']:c['exitCode'] for c in observations if c['exitCode']}==failures,'all three auxiliary failures')
    launch=byid['exec-e5001e1d-a4df-447b-9103-ecbb07b31da4'];ck(launch['id']=='exec-e5001e1d-a4df-447b-9103-ecbb07b31da4' and launch['status']=='completed' and launch['exitCode']==0 and not launch['output']['truncated'],'actual scientific tool exit')
    trans=read(ctl+'launch-observed-tool-return.json');ck(launch['output']['text']==trans['actual_tool_stdout']=='launch_actual_exit=0\n' and trans['actual_tool_exit']==0,'transcription matches actual')
    lines=[line for line in launch['command'].splitlines() if line.startswith(trans['argv'][0]+' ')]
    ck(len(lines)==1 and shlex.split(lines[0])[:len(trans['argv'])]==trans['argv'],'actual exact launch argv')
    ck(launch['cwd']==trans['cwd'] and ('PYTHONPATH=\''+trans['PYTHONPATH']+'\'') in launch['command'],'actual cwd/environment')
    identity(bundle/sup/'execution.json',trans['supervision']);identity(bundle/ctl/'launch.stdout.log',trans['stdout']);identity(bundle/ctl/'launch.stderr.log',trans['stderr'])
    e=read(sup+'execution.json');argv=e['argv'];plan=read('research-0010-headroom/experiments/0010-headroom/plan.json')
    plan_sha=sha(bundle/'research-0010-headroom/experiments/0010-headroom/plan.json')
    ck(argv[argv.index('--plan-sha256')+1]==plan_sha,'child/plan binding')
    ck(trans['source_main']==result['execution_commit'] and trans['source_commit']==result['source_commit'] and trans['source_run_sha256']==plan['driver_files']['run.py'],'actual source identity')
    ck(sha(bundle/'research-0010-headroom/protocols/0010-real-data-headroom-freeze.json')==result['freeze_sha256'],'exported freeze byte match')
    lm=read('clip-0010-assets/0010-headroom-launch.started.json');cm=read('clip-0010-assets/0010-headroom-started.json')
    ck(lm['plan_sha256']==plan_sha and lm['output']==cm['output']==argv[argv.index('--output')+1] and cm['status']=='started','exclusive launch/child marker binding')
    reviewed=byid['exec-98fb9239-cf79-47a0-a644-163c4be42545'];ck(reviewed['exitCode']==0 and reviewed['status']=='completed' and not reviewed['output']['truncated'],'actual cloud saved review')
    review=read(ctl+'review.execution.json');actual_json=[json.loads(line) for line in reviewed['output']['text'].splitlines() if line.startswith('{')]
    ck(review in actual_json and review['actual_exit']==review['raw_wait_status']==0,'saved review exact execution')
    for name,s in review['logs'].items():identity(bundle/ctl/name,s)
    command=read(ctl+'review.command.json');ck(command['source_sha256']==sha(bundle/ctl/'review_saved.py'),'cloud saved checker source')
    row_report=read(ctl+'saved-review.json')
    for k in ('decision','correct','accuracy','strongest_fixed_controls','paired','batches','occupancy_strata','singleton_assessed','gates'):
        ck(row_report['recomputed'][k]==result['summary'][k],'cloud/parent arithmetic '+k)
    pre=read(ctl+'preflight-public.json');ck(pre['status']=='PASS_PREFLIGHT' and pre['head']==result['execution_commit'] and pre['source']==result['source_commit'],'preflight source')
    ck(pre['repository_inputs']==45 and pre['external_inputs']==9 and pre['preserved_count']==9818,'preflight counts')
    ck(pre['original_full_preflight']==m['omitted_full_preflight'],'omitted preflight disclosure')
    r=read(out+'receipt.json')
    for name,key in [('features.npz','features_sha256'),('partial-features.npz','partial_features_sha256')]:ck(m['private_npz_not_exported'][name]['sha256']==r[key],'private hash-only boundary')
    ck(not list(bundle.rglob('*.npz')),'no NPZ published or read')
    suspicious=[]
    for p in bundle.rglob('*'):
        if not p.is_file():continue
        raw=p.read_bytes()
        if b'/Users/' in raw:suspicious.append(str(p.relative_to(bundle))+': local-home path')
        for u in re.findall(rb'https?://[^\s<>"\x27]+',raw):
            if re.search(rb'[?&](?:token|access_token|X-Amz-Signature|X-Goog-Signature)=',u,re.I):suspicious.append(str(p.relative_to(bundle))+': signed/token URL')
    ck(not suspicious,'publication path/URL scan')
    return dict(status='PASS_ARCHIVE_AND_ACTUAL_TOOL_CLOSURE',checks=checks,files=37,bytes=405907,actual_commands=len(observations),scientific_tool_id=launch['id'],scientific_actual_exit=launch['exitCode'],scientific_tool_duration_ms=launch['durationMs'],supervisor_wall_seconds=e['wall_seconds'],peak_rss_kib=e['max_rss_kib'],auxiliary_failures=failures,raw_NPZ_replayed=False,omitted_full_preflight=m['omitted_full_preflight'],old_file_preservation_cloud_attested_count=9818,old_file_preservation_parent_full_verified=False,reviewer_source_sha256=sha(Path(__file__)),arithmetic_report_sha256=sha(arithmetic_path),actual_observations_sha256=sha(actual_path))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('bundle','observations','arithmetic','report'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=run(Path(a.bundle),Path(a.observations),Path(a.arithmetic))
    with Path(a.report).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
