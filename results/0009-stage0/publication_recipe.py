"""Publication-only saved-byte packaging; no scientific implementation imports."""
import collections,gzip,hashlib,io,json,os,re,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('/workspace/scratch/a349d03a0e0a');CONTROL=ROOT/'scratch/0009-stage0-control';REPO=ROOT/'research-0009-stage0';OUT=REPO/'results/0009-stage0'
MAIN='cd144b90d337a172a892b56d87cd794524605753';BRANCH='research/0009-stage0-records'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def inventory(root):return {str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()}
def main():
 assert not OUT.exists() and not git('status','--porcelain')
 assert git('rev-parse','HEAD').decode().strip()==git('rev-parse','origin/main').decode().strip()==MAIN
 assert sha(CONTROL/'compact-execution.json')=='7eed0bc54abd17c8d4abf96bc0bab1bc7bc2ae342d5fa037b6c357e7a7c1bb81'
 before=inventory(CONTROL);baseline=json.loads((CONTROL/'pre-execution.json').read_text())
 for n,h in baseline['tracked'].items():assert sha(REPO/n)==h,n
 receipt='protocols/0009-geometric-stage0-freeze.json'
 assert sha(REPO/receipt)==baseline['canonical_sha256']
 assert git('show',baseline['freeze_commit']+':'+receipt)==(REPO/receipt).read_bytes()==git('show',':'+receipt)
 for n,h in baseline['frozen_files'].items():
  assert sha(REPO/n)==h
  for ref in (':'+n,'HEAD:'+n,baseline['source_commit']+':'+n):assert hashlib.sha256(git('show',ref)).hexdigest()==h
 subprocess.run(['git','switch','-c',BRANCH,MAIN],cwd=REPO,check=True,capture_output=True)
 OUT.mkdir();copies=[]
 def copy(src,rel):
  dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);data=src.read_bytes()
  with dest.open('xb') as f:f.write(data)
  assert sha(dest)==sha(src)
  copies.append({'original_source':str(src.relative_to(CONTROL)),'original_bytes':len(data),'original_sha256':sha(src),'public_path':str(dest.relative_to(OUT)),'public_bytes':len(data),'public_sha256':sha(dest),'mode':'byte-exact','redaction_scope':'none; cloud project and runtime paths retained as execution-location evidence'})
 for src in sorted(CONTROL.iterdir()):
  if src.is_file() and src.name!='pre-execution.json':copy(src,'control/'+src.name)
 for sub in ('supervision','evaluation'):
  for src in sorted((CONTROL/sub).iterdir()):
   assert src.is_file();copy(src,sub+'/'+src.name)
 # Do not publish full unrelated cache/old-repository inventory maps.
 public_before={k:v for k,v in baseline.items() if k not in ('tracked','historical_trees')}
 public_before['omitted_private_inventory_maps']={'tracked_files':len(baseline['tracked']),'historical_trees':{k:len(v) for k,v in baseline['historical_trees'].items()},'reason':'Unrelated historical inventories omitted; complete original remains private and hash-bound.'}
 write(OUT/'control/pre-execution.summary.json',public_before)
 copies.append({'original_source':'pre-execution.json','original_bytes':(CONTROL/'pre-execution.json').stat().st_size,'original_sha256':sha(CONTROL/'pre-execution.json'),'public_path':'control/pre-execution.summary.json','public_bytes':(OUT/'control/pre-execution.summary.json').stat().st_size,'public_sha256':sha(OUT/'control/pre-execution.summary.json'),'mode':'explicit summary derivative','redaction_scope':'omitted full tracked/historical path-to-hash maps; all other fields retained'})
 collection=CONTROL/'collection';members=inventory(collection)
 assert len(members)==515
 with (OUT/'collection.tar.gz').open('xb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=9) as compressed:
   with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as tf:
    for n,record in members.items():
     data=(collection/n).read_bytes();entry=tarfile.TarInfo('collection/'+n);entry.size=len(data);entry.mode=0o644;entry.uid=entry.gid=entry.mtime=0;entry.uname=entry.gname='';tf.addfile(entry,io.BytesIO(data))
 assert (OUT/'collection.tar.gz').stat().st_size<20*1024**2
 seen={}
 with tarfile.open(OUT/'collection.tar.gz','r:gz') as tf:
  for m in tf:
   assert m.isfile() and m.name.startswith('collection/') and '..' not in Path(m.name).parts
   data=tf.extractfile(m).read();n=m.name[len('collection/'):];seen[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 assert seen==members
 write(OUT/'collection-file-manifest.json',{'schema':'0009-original-synthetic-archive-v1','archive':'collection.tar.gz','archive_sha256':sha(OUT/'collection.tar.gz'),'original_source':'collection/','file_count':len(members),'original_total_bytes':sum(x['bytes'] for x in members.values()),'roundtrip_all_bytes_verified':True,'deterministic_encoding':'sorted USTAR regular files; uid/gid/mtime=0, mode0644; gzip empty filename and mtime=0','files':members})
 # Read saved decisions only. No score computation, RNG, geometric solver or DP.
 evaluation=json.loads((CONTROL/'evaluation/evaluation.json').read_text());cases={}
 unsupported=[]
 for src in sorted((collection/'cases').glob('*.json')):
  c=json.loads(src.read_text());cases[c['case_id']]=c
  failed=[(i,h) for i,h in enumerate(c['hypotheses']) if h['tail']['status']=='unsupported']
  if failed:
   assert len(failed)==1;i,h=failed[0];cid=h['canonical_id']
   unsupported.append({'case_id':c['case_id'],'first_unsupported_hypothesis_record_index':i,'canonical_hypothesis_id':cid,'H':c['inputs']['candidates'][cid],'H_sha256':h['H_sha256'],'reason':h['tail']['reason'],'operations':h['tail']['operations'],'graph_components':h['graph_components'],'tail_attempted':c['tail_attempted']})
 harms=[r['case_id'] for r in evaluation['cases'] if r['methods']['raw']['success'] and not r['methods']['tail']['success']]
 failed_ids={r['case_id'] for r in unsupported};wrong=[]
 for row in evaluation['cases']:
  if row['status']=='completed' and not row['methods']['tail']['success']:
   wrong.append({'case_id':row['case_id'],'saved_tail_evaluation':row['methods']['tail'],'saved_raw_evaluation':row['methods']['raw'],'all_six_saved_selections':cases[row['case_id']]['selections'],'interpretation':'Completed but incorrect tail selection; raw also incorrect. This is not a raw-success paired harm and is not a solver failure.'})
 assert len(unsupported)==12 and len(harms)==12 and set(harms)==failed_ids and len(wrong)==1
 write(OUT/'saved-case-description.json',{'scope':'Post-execution descriptive lookup of saved records; no rescoring, regeneration, RNG or tail DP.','unsupported_cases':unsupported,'raw_success_tail_failure_ids':harms,'overlap_with_unsupported':sorted(set(harms)&failed_ids),'raw_success_tail_failure_not_unsupported':sorted(set(harms)-failed_ids),'unsupported_not_raw_success_tail_failure':sorted(failed_ids-set(harms)),'completed_but_incorrect_tail':wrong})
 post=json.loads((CONTROL/'postcheck.json').read_text());sup=json.loads((CONTROL/'supervision/execution.json').read_text())
 rows=['| Family (64 cases each) | raw | msac | cell | excess | ratio | tail |','|---|---:|---:|---:|---:|---:|---:|']
 for family in ('uniform','clustered_outlier','compact_true','sparse_isolated'):rows.append('| '+family+' | '+' | '.join(str(post['success_counts'][family][m]) for m in ('raw','msac','cell','excess','ratio','tail'))+' |')
 README='''# 0009 frozen synthetic stage0: NO_GO

This exact candidate is stopped. The single authorized collection recorded all 256 cases; 244 tail cases completed and 12 were unsupported. This is a synthetic, oracle-covered translation-ranking development screen, not evidence about real images, a complete geometric estimator, novelty, or a manuscript.

'''+ '\n'.join(rows)+'''

The primary clustered-outlier gain over the strongest aggregate cheap control (`raw`, 64/64) is **0**, below the frozen required 8. Primary headroom is insufficient (64 > 56). Solver coverage fails. The paired-harm gate fails: compact_true harms versus raw/msac/cell/excess/ratio are 5/5/5/5/4; sparse_isolated harms are 7/7/0/7/0; uniform harms are all 0. The frozen limit is 2 per comparison. Resource and independent saved-result replay pass, but do not override these failures.

## Saved-case failure attribution

All 12 raw-success → tail-failure cases overlap exactly with the 12 unsupported cases; see [the complete lookup](saved-case-description.json) for each first unsupported H, canonical ID, matrix, reason and component. The unsupported reason is a non-complete component with smaller side greater than 12. Across 8,192 H slots, 7,985 tail calculations completed, 12 were unsupported and 195 remained unattempted after those stops. No failure was retried. All cheap controls have 8,192 recorded scores.

There is also one **completed but incorrect** tail selection, `sparse_isolated-23`: canonical H 13, translation (128,128), saved transfer error 181.0193359837562 pixels. Raw and all other methods also selected H 13 incorrectly. It is neither a solver failure nor a raw-success paired harm. Aggregate counts alone are not used to assign failure causes.

## Execution and evidence

- Execution HEAD `cd144b90d337a172a892b56d87cd794524605753`; source `4b11f1de4d6fcc9bef45c8eee824ff8e7241abc0`; separate freeze `17f417efcccf85047e0935569f87f94b27705e82` (15 inputs). Canonical receipt SHA256 `2485a844f95c47ee420617d0464be7ce8bf5e747b8539b70f2450df2306d6631`.
- Python 3.12.14, cached NumPy 2.0.2, Linux CPU, affinity [0,1,2,3], four numeric threads, actual 8 GiB cgroup and per-process address-space cap. Frozen outer deadline 900 s; case cap 2.5 s.
- 73 targeted artificial tests passed with zero skips; all seven Linux process/resource/termination tests ran. Saved checker self-test passed three groups.
- Frozen supervisor observed collection exit 0/raw wait 0, wall 19.198018779999984 s, user CPU 19.347988 s, system CPU 0.143313 s, peak RSS 34,656 KiB, no timeout. The additional parent observed wrapper exit 0/raw wait 0 and wall 19.234951837002882 s.
- Separate evaluation exit 0 (0.8873839939988102 s); independent replay exit 0 (9.645058818001417 s), PASS for arithmetic/resource evidence and **NO_GO** for the candidate. Replay checked 1,549,368 items, including 227,998 numerical comparisons; maximum absolute difference 2.842170943040401e-14. It did not regenerate RNG or recompute tail DP.
- Preflight, verify-only, tests, self-test, collection, evaluation, independent review and postcheck actual exits/raw waits are all 0. Exact commands, UTC, closed logs and parent wait4 records are in `control/`; frozen child supervision is in `supervision/`. Exit 0 means the declared operation recorded its results, not that scientific gates passed.
- Postcheck preserved all 2,892 prior tracked files (2,672 result/protocol files), all 15 scientific inputs/canonical receipt, and 40,664 historical/cache regular files. Full unrelated private path/hash inventory maps are omitted from `control/pre-execution.summary.json`; the original inventory hash is retained. `postcheck.json` and `compact-execution.json` are unchanged original summaries.

## Archive and copy boundaries

`collection.tar.gz` contains the complete **original synthetic** collection: all 256 case records, all labels, raw synthetic coordinates, candidate matrices and recorded scores/tails/failures, planned cases, collection receipt and selection lock (515 files). No collection bytes were regenerated. The sorted deterministic USTAR/gzip archive has mtime 0; every extracted member was read back and matched its original byte length and SHA256 in [collection-file-manifest.json](collection-file-manifest.json).

To inspect: verify the archive SHA256 from the manifest, then `tar -xzf collection.tar.gz` in a new directory. Verify each extracted `collection/<relative-path>` against the per-file manifest. The original selection lock remains byte-exact inside the archive. Evaluator output is `evaluation/evaluation.json`; independent review is `control/review.json`.

`copy-manifest.json` distinguishes byte-exact copies from the explicitly labeled pre-execution summary derivative. Cloud project/runtime absolute paths are intentionally retained as execution-location evidence; no credentials, signed URLs, upstream datasets, images, weights or unrelated private inventory contents are published. `artifact-hashes.json` excludes itself to avoid circular hashes; source controls and the archive have their own original SHA bindings. The wrapper parent has no recursively added outer exit observer.

AI assisted this research, execution and documentation. New acquired asset bytes: **0**; no additional fees. The older asset ceiling record remains 7,532,690,072 bytes; HPatches metadata has 163,286 measured 206 bytes and an unresolved redirect-body gap, so no verified updated total ceiling is claimed. 0006's stdout evidence gap remains OPEN; 0003 remains deferred; older stopped candidates and results are unchanged. No parameter rescue, additional experiment or manuscript follows from this result.
'''
 with (OUT/'README.md').open('x') as f:f.write(README)
 write(OUT/'copy-manifest.json',{'schema':'0009-publication-copy-v1','original_control_root':'/workspace/scratch/a349d03a0e0a/scratch/0009-stage0-control','copies':copies,'collection_archive':'See collection-file-manifest.json for all original members.','self_exclusion':'This manifest does not claim its own hash; artifact-hashes.json hashes it.','publication_only':True})
 # All selected originals are unchanged; archive roundtrip verified. No cache writes.
 assert inventory(CONTROL)==before,'original control bytes changed'
 for n,h in baseline['tracked'].items():assert sha(REPO/n)==h,n
 text_count=0
 for p in OUT.rglob('*'):
  if not p.is_file():continue
  if p.name=='collection.tar.gz':continue
  text=p.read_text();text_count+=1
  assert '\x00' not in text
  assert not re.search(r'(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|Authorization:\s*Bearer|[?&](?:X-Amz-Signature|token|Signature)=)',text,re.I),p.name
  assert not re.search(r'/(?:Users|home)/[^\s"\']+',text),p.name
  if p.suffix=='.json':json.loads(text)
 write(OUT/'publication-validation.json',{'status':'PASS','scope':'Publication-only original/copy/archive/Git/privacy checks; no scientific computation rerun.','original_control_files_unchanged':len(before),'original_control_total_bytes':sum(x['bytes'] for x in before.values()),'original_control_inventory_digest':hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'tracked_originals_unchanged':len(baseline['tracked']),'old_result_protocol_files_unchanged':sum(n.startswith(('results/','protocols/')) for n in baseline['tracked']),'frozen_inputs_unchanged':15,'archive_members_roundtrip_verified':len(members),'text_files_scanned_before_this_receipt':text_count,'privacy_scan':'No credential/signed-query/personal-home path pattern found; cloud project and interpreter paths intentionally retained.','originals_modified':False,'new_asset_bytes':0,'publication_script_sha256':sha(__file__),'publication_errors':[]})
 # Include exact packaging recipe; it is not a scientific frozen input.
 with (OUT/'publication_recipe.py').open('xb') as f:f.write(Path(__file__).read_bytes())
 hashes=inventory(OUT);write(OUT/'artifact-hashes.json',{'schema':'0009-publication-artifact-hashes-v1','self_exclusion':['artifact-hashes.json'],'files':hashes})
 for n,v in hashes.items():assert (OUT/n).stat().st_size==v['bytes'] and sha(OUT/n)==v['sha256']
 assert inventory(CONTROL)==before
 files=inventory(OUT)
 print(json.dumps({'status':'PASS','branch':BRANCH,'files':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),'archive_bytes':(OUT/'collection.tar.gz').stat().st_size,'archive_sha256':sha(OUT/'collection.tar.gz'),'artifact_hashes_sha256':sha(OUT/'artifact-hashes.json'),'unsupported':len(unsupported),'paired_raw_harms':len(harms),'completed_wrong':wrong[0]['case_id'],'original_control_preserved':len(before)},sort_keys=True))
if __name__=='__main__':main()
