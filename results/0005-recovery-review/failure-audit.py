import hashlib,json,os,subprocess,sys,tempfile,platform
from pathlib import Path
ROOT=Path('/Users/simonxie/Documents/Codex/2026-10-12/ban/work/embodied-vision-reliability')
BASE='e168ba830f9f0eff37192ee8745207ff7d3484ef'
HEAD='dec0546a33c41eabd731da903d95cd2651bf080e'
P='results/0005-native-capability/'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def digest(x):return hashlib.sha256(x).hexdigest()
def raw(p,ref=HEAD):return git('show',ref+':'+p)
def load(p):return json.loads(raw(p))
def tree(ref):
 out={}
 for r in git('ls-tree','-r','-z',ref).split(b'\0'):
  if r:
   meta,p=r.split(b'\t',1);mode,kind,oid=meta.split();out[p.decode()]=(mode.decode(),kind.decode(),oid.decode())
 return out
assert git('rev-parse','HEAD').decode().strip()==HEAD
worktree_status=git('status','--porcelain=v1').decode().splitlines()
bt,ht=tree(BASE),tree(HEAD)
freeze=load('protocols/0005-native-capability-freeze.json')
frozen={}
for p,h in freeze['files'].items():
 checks={ref:digest(raw(p,ref)) for ref in [BASE,HEAD,freeze['source_commit']]}
 checks['index']=digest(git('show',':'+p));checks['worktree']=digest((ROOT/p).read_bytes())
 assert set(checks.values())=={h};frozen[p]=h
assert len(frozen)==7
fr='protocols/0005-native-capability-freeze.json';frsha=digest(raw(fr))
assert raw(fr)==raw(fr,BASE)==(ROOT/fr).read_bytes()==git('show',':'+fr)
manifest=load(P+'copy-manifest.json');seen=set()
for x in manifest['files']:
 assert x['copied'] not in seen;seen.add(x['copied'])
 b=raw(x['copied']);assert len(b)==x['bytes'] and digest(b)==x['sha256']
assert len(seen)==35 and manifest['self_excluded'] is True
artifacts={p:{'bytes':len(raw(p)),'sha256':digest(raw(p))}for p in ht if p.startswith(P)}
assert set(artifacts)-seen=={P+'README.md',P+'copy-manifest.json'}
baseline=load(P+'review/tracked-baseline.json');assert set(baseline)==set(bt)
# Hash all baseline Git blobs without touching any cloud cache or scientific data.
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for p in baseline:
 proc.stdin.write((bt[p][2]+'\n').encode());proc.stdin.flush()
 h=proc.stdout.readline().split();assert h[1]==b'blob';size=int(h[2]);b=proc.stdout.read(size);assert proc.stdout.read(1)==b'\n'
 assert digest(b)==baseline[p],p
proc.stdin.close();proc.wait();assert proc.returncode==0
changed=[p for p in bt if ht.get(p)!=bt[p]]
assert set(changed)=={'CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'}
assert raw('notes/2026-10-18.md').startswith(raw('notes/2026-10-18.md',BASE))
assert all(ht[p]==bt[p] for p in bt if p.startswith('results/'))
phase={}
parent_sha=digest(raw(P+'review/stage_parent.py'))
for mode,d in [('verify','verification'),('dependencies','preparation-dependencies'),('software','software'),('review','review')]:
 rec=load(P+d+'/'+mode+'-execution.json')
 for stream in ['stdout','stderr']:assert digest(raw(P+d+'/'+mode+'.'+stream+'.log'))==rec[stream+'_sha256']
 expected=digest(raw(P+'review/review_failure.py')) if mode=='review' else parent_sha
 assert rec['source_sha256']==expected
 if mode!='review':
  assert rec['frozen_input_sha256']==frozen and rec['freeze_sha256']==frsha and rec['execution_main']==BASE
  assert rec['command'][:3]==['timeout','--kill-after=5s','1800']
 assert os.waitstatus_to_exitcode(rec['raw_wait_status'])==rec['exit_code']
 phase[mode]={k:rec[k] for k in ['exit_code','raw_wait_status','wall_seconds','peak_rss_kib','source_sha256','stdout_sha256','stderr_sha256']}
assert phase['dependencies']['exit_code']==1
assert all(phase[m]['exit_code']==0 for m in ['verify','software','review'])
assert b'Ran 44 tests in 1.628s' in raw(P+'software/software.stderr.log')
prep=load(P+'preparation-dependencies/preparation-dependencies.json');install=load(P+'preparation-dependencies/install.json')
assert prep['installation']==install and install['exit_code']==install['raw_wait_status']==0
assert prep['status']=='failed' and prep['checkout_commit']==BASE
assert prep['config_sha256']==frozen['configs/0005-native-capability.json'] and prep['source_sha256']==frozen['scripts/prepare_tapnext_assets.py']
assert prep['error']=={'type':'PrepFailure','reason':'Unexpected or changed source entry: tapnet/tapnext/tapnext_lru_modules.py.partial'}
assert 'import_check' not in prep and 'overlay_manifest' not in prep
for s in ['stdout','stderr']:assert digest(raw(P+'preparation-dependencies/install.'+s+'.log'))==install[s+'_sha256']
assert prep['marker_sha256']==digest(raw(P+'preparation-dependencies/markers/dependencies.started.json'))
cfg=load('configs/0005-native-capability.json')['preparation'];specs={a['id']:a for a in cfg['assets']}
inv=load(P+'review/postfailure-cache-inventory.json');ov=load(P+'review/postfailure-overlay-manifest.json')
assert {k.removeprefix('overlay/'):v for k,v in inv.items() if k.startswith('overlay/')}==ov['files'];assert len(ov['files'])==378
for a in prep['assets']:
 spec=specs[a['asset']['id']];assert a['asset']==spec
 assert a==load(P+'preparation-dependencies/asset-records/'+spec['id']+'.json')
 assert inv[spec['relative_path']]=={'bytes':spec['bytes'],'sha256':spec['sha256']}
 assert a['status']=='downloaded_verified' and a['http_status']==200 and a['http_payload_attempts']==1
 assert a['payload_bytes_observed']==spec['bytes'] and a['observed_payload_sha256']==spec['sha256']==a['sha256']
assert len(prep['assets'])==8
payload=sum(a['payload_bytes_observed'] for a in prep['assets']);assert payload==9179574
ledger=load(P+'preparation-dependencies/acquisition-ledger.json');snap=P+'preparation-dependencies/acquisition-ledger-dependencies.json'
assert raw(snap)==raw(P+'preparation-dependencies/acquisition-ledger.json')
assert digest(raw(snap))==prep['ledger_sha256']==prep['ledger_snapshot']['sha256']
assert ledger['reservations']=={a['asset']['id']:a['asset']['bytes']+1 for a in prep['assets']}
assert ledger['payload_bytes_observed']==payload
assert ledger['base_ceiling_bytes']==cfg['budget']['historical_ceiling_bytes']+cfg['budget']['metadata_allowance_bytes']
assert ledger['conservative_cumulative_ceiling_bytes']==ledger['base_ceiling_bytes']+sum(ledger['reservations'].values())==4966784411
for n in ['preparation-checkpoint.json','markers/checkpoint.started.json','0005-native-started.json','import-check.json','overlay-manifest.json']:
 assert n not in inv
partials={p:v for p,v in inv.items() if p.endswith('.partial')}
assert len(partials)==5 and all(v=={'bytes':0,'sha256':digest(b'')}for v in partials.values())
review=load(P+'review/failure-review.json')
assert review['cache_inventory_sha256']==digest(raw(P+'review/postfailure-cache-inventory.json'))
assert review['overlay_inventory_sha256']==digest(raw(P+'review/postfailure-overlay-manifest.json'))
assert review['old_markers_preserved']==load(P+'review/old-cache-markers.json')
assert review['baseline_files_preserved']==len(bt)==1769
assert review['prior_result_files_preserved']==sum(p.startswith('results/') for p in bt)==1595
assert review['frozen_files_unchanged']==frozen
# Minimal ordinary-local-filesystem lifecycle fixture: no candidate imports or external IO.
with tempfile.TemporaryDirectory(prefix='evr-0005-partial-posix-') as td:
 final=Path(td)/'final';partial=Path(td)/'final.partial';payload_fixture=b'synthetic probe\n'
 with partial.open('xb') as stream:
  created_size=partial.stat().st_size
  stream.write(payload_fixture);stream.flush();os.fsync(stream.fileno())
  os.link(partial,final);partial.unlink();inside_exists=partial.exists()
 closed_exists=partial.exists();assert final.read_bytes()==payload_fixture and not inside_exists and not closed_exists
fixture={'operation':'open-xb/write/flush/fsync/link/unlink-before-close','payload_bytes':len(payload_fixture),'created_bytes':created_size,'partial_exists_after_unlink_while_open':inside_exists,'partial_exists_after_close':closed_exists,'final_sha256':digest(payload_fixture),'platform':platform.system(),'python':platform.python_version(),'boundary':'Synthetic local filesystem only; does not identify cloud filesystem behavior.'}
report={'schema_version':1,'verdict':'APPROVE_FAILURE_EVIDENCE_ONLY_RECOVERY_REQUIRES_NEW_BOUND_PROCEDURE','reviewed_head':HEAD,'execution_base':BASE,'scientific_source_commit':freeze['source_commit'],'scope':'Read-only Git and saved small receipts; synthetic local link/unlink fixture only. No runtime/model/data/network/preparation execution.','worktree_status_at_audit':worktree_status,'checks':{'copy_manifest_entries_verified':35,'result_files_hashed':len(artifacts),'frozen_inputs_current_index_head_base_source_verified':7,'canonical_receipt_sha256':frsha,'baseline_map_entries_verified_against_base_blobs':len(bt),'prior_files_byte_preserved_in_head':len(bt)-len(changed),'prior_results_byte_preserved':1595,'permitted_changed_paths':changed,'notes_prefix_preserved':True,'overlay_saved_manifest_entries':378,'asset_records_verified_against_config_and_saved_inventory':8,'new_download_payload_bytes':payload,'conservative_cumulative_ceiling_bytes':ledger['conservative_cumulative_ceiling_bytes'],'synthetic_software_tests_in_saved_log':44},'phase_receipts':phase,'installation_receipt':install,'frozen_input_sha256':frozen,'source_partials_failure':prep['error'],'all_saved_empty_partial_entries':partials,'root_cause':{'verified_code_trace':['scripts/prepare_tapnext_assets.py:169 creates an empty exclusive partial before HTTP','scripts/prepare_tapnext_assets.py:190-194 verifies flushed partial length and digest','scripts/prepare_tapnext_assets.py:195-197 links final then unlinks partial before close and success status','scripts/prepare_tapnext_assets.py:203 conflates absent and zero-byte partial in its counter','scripts/prepare_tapnext_assets.py:222-223 rejects any unbound source entry before the import child'],'conclusion':'Empty extras are observed, but their persistence/recreation cause is unresolved. Ordinary local lifecycle does not reproduce it. No deterministic downloader or cloud-filesystem cause claimed.','local_fixture':fixture},'limitations':['Original cloud files were not reread; public copy hashes, inventory consistency and executed audit source were checked. Byte-exact origin claims remain bounded by the preserved cloud copy/audit records.','Phase wait4 receipts substantiate their immediate children; no further durable wait4 receipt for stage_parent.py or the review parent is published.','No import-only validation, checkpoint acquisition, model construction, forwards or GT access is evidenced by this attempted path; this is not a global attestation about unrelated processes.','The postfailure overlay inventory is not a completed or compatible preparation approval.','Three old cache marker digests match the saved baseline/audit, but actual cloud markers were not independently reread.'],'recovery_recommendation':['Preserve frozen v1 and the entire failed cache, stage marker, receipt, logs, ledger and inventory; do not delete extras or rewrite failed status.','Review and freeze a separate offline recovery helper/config/receipt bound to this failure, all eight asset records, saved complete overlay map and v1 seven-file hashes.','Under an exclusive fresh recovery marker, rehash existing five final source files and three wheels; create a clean separate exact-five-file source tree by byte copy, retaining original partial evidence. Verify no symlinks, unbound initializers or extra entries.','Reuse existing 378-file installed overlay only after exact path/size/digest equality; perform a separately supervised import-only version/origin check with unchanged package pins and limits, no pip/network/checkpoint/model. Record absence of new transfer and carry the prior cumulative ledger without resetting history.','Any mismatch fails closed; checkpoint acquisition and native execution remain separate uncompleted stages and are outside this offline-recovery review.'],'artifact_sha256':artifacts,'reviewer_script_sha256':digest(Path(__file__).read_bytes())}
Path('/tmp/evr-0005-failure-parent-review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'checks':report['checks'],'phase_exits':{k:v['exit_code'] for k,v in phase.items()},'local_fixture':fixture,'report_sha256':digest(Path('/tmp/evr-0005-failure-parent-review.json').read_bytes())},indent=2))
