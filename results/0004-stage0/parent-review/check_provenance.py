import ast, collections, csv, datetime, hashlib, json, math, pathlib, re, subprocess
R=pathlib.Path.cwd(); O=R/'results/0004-stage0'
BASE='ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306'; HEAD='ca68cfd56984dedf1cf875d9b4a6c68440565ee1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
assert git('rev-parse','HEAD').decode().strip()==HEAD
f=j(R/'protocols/0004-stage0-freeze.json'); rec=j(O/'run/receipt.json'); cfg=j(R/'configs/0004-stage0.json')
assert sha(R/'protocols/0004-stage0-freeze.json')==rec['freeze_sha256']=='e14bb4f3f4b4dd91c96fe63b1d28bc9a95636d560854de190a701c25dc34667d'
assert rec['source_sha256']==f['files']
for p,h in f['files'].items():
 assert sha(R/p)==h
 for ref in [f['source_commit'],BASE,HEAD]:assert hashlib.sha256(git('show',ref+':'+p)).hexdigest()==h
basepaths=git('ls-tree','-r','--name-only',BASE).decode().splitlines(); baseline=j(O/'tracked-baseline.json')
assert set(basepaths)==set(baseline)
for p,h in baseline.items():assert sha(R/p)==h==hashlib.sha256(git('show',BASE+':'+p)).hexdigest()
changes=git('diff','--name-status',BASE,HEAD).decode().splitlines(); assert len(changes)==33 and all(x.startswith('A\tresults/0004-stage0/')for x in changes)
audit=j(O/'artifact-audit.json')
for p,h in audit['artifacts_sha256'].items():assert sha(R/p)==h
for mode in ['software','scientific']:
 child=j(O/(mode+'-child.json')); outer=j(O/(mode+'-outer.json'))
 assert child['exit_code']==child['raw_wait_status']==outer['observed_wrapper_exit']==outer['raw_wait_status']==0
 assert child['limits']=={'affinity':[0,1],'AS_bytes':6*1024**3,'timeout_seconds':900,'kill_after_seconds':5}
 assert child['frozen_inputs_sha256']==f['files']
 assert child['execution_commit']==BASE and child['source_commit']==f['source_commit']
 for p,h in child['outer_source_sha256'].items():assert sha(R/p)==h
 assert outer['source_sha256']==sha(O/'launch_supervisor.py')
 for s,h in child['logs_sha256'].items():assert sha(O/(mode+'.'+s+'.log'))==h
 for s,h in outer['logs_sha256'].items():assert sha(O/(mode+'-wrapper.'+s+'.log'))==h
 assert j(O/(mode+'-wrapper.stdout.log'))==child
assert rec['marker_sha256']==j(O/'marker-observation.json')['marker_sha256']==j(O/'scientific-child.json')['persistent_marker_sha256']
for p,h in rec['artifacts'].items():assert sha(O/'run'/p)==h
assert rec['predictions_sha256_before_scoring']==sha(O/'run/predictions.csv')
rows=list(csv.DictReader((O/'run/predictions.csv').open())); assert len(rows)==4488
ids=rec['point_indices']; count=collections.Counter(r['branch'] for r in rows); assert count=={'keep':2000,'zero':2000,'clone_control':40,'shadow':448}
values={}
for row in rows:
 key=(row['branch'],int(row['frame']),int(row['shadow_start']),int(row['point_index']))
 assert key not in values
 val=tuple(float(row[c])for c in ['x','y','occlusion_logit','expected_distance_logit']); assert all(map(math.isfinite,val)); values[key]=val
expected={(b,t,-1,p)for b in ['keep','zero']for t in range(250)for p in ids}|{('clone_control',t,-1,p)for t in cfg['control_frames']for p in ids}|{('shadow',t,s,p)for s in cfg['shadow_starts']for t in range(s,s+8)for p in ids}
assert set(values)==expected
controlmax=[]
for t in cfg['control_frames']:
 controlmax.append(max(abs(a-b)for p in ids for a,b in zip(values[('keep',t,-1,p)],values[('clone_control',t,-1,p)])))
controlmax.append(max(abs(a-b)for p in ids for a,b in zip(values[('keep',0,-1,p)],values[('zero',0,-1,p)])))
for s in cfg['shadow_starts']:
 controlmax.append(max(abs(a-b)for p in ids for a,b in zip(values[('zero',s,-1,p)],values[('shadow',s,s,p)])))
assert len(controlmax)==13 and max(controlmax)==0
assert len(rec['controls'])==13 and all(c['passed']and c['same_next_context']and c['max_abs_difference']==0 for c in rec['controls'])
timings=j(O/'run/timings.json'); assert [r['frame']for r in timings]==list(range(250))
for t,row in enumerate(timings):
 n=2+int(t in cfg['control_frames'])+sum(s<=t<s+8 for s in cfg['shadow_starts'])
 assert len(row['refiner_seconds'])==n
 assert all(math.isfinite(v)and v>0 for v in [row['encoder_seconds'],row['whole_frame_seconds'],*row['refiner_seconds']])
 assert row['whole_frame_seconds']>=row['encoder_seconds']+sum(row['refiner_seconds'])
assert sum(len(r['refiner_seconds'])for r in timings)==561
checker=j(O/'independent-check-execution.json'); verify=j(O/'independent-verification.json')
assert checker['observed_exit_code']==checker['raw_wait_status']==0
assert checker['checker_sha256']==verify['checker_source_sha256']==sha(O/'check_saved.py')
assert checker['parent_sha256']==sha(O/'check_parent.py')
for s,h in checker['logs_sha256'].items():assert sha(O/('independent-check.'+s+'.log'))==h
imports=[]
for n in ast.walk(ast.parse((O/'check_saved.py').read_text())):
 if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
 if isinstance(n,ast.ImportFrom):imports.append(n.module)
assert set(imports)=={'csv','hashlib','json','math','statistics','collections','pathlib','numpy'}
for p in O.rglob('*'):
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<1024**2
 text=p.read_text(); assert '\x00'not in text
 assert not re.search(r'sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text)
report={'status':'PASS_WITH_STATED_OBSERVATION_BOUNDARIES','reviewed_commit':HEAD,'base_commit':BASE,'frozen_files_verified':len(f['files']),'historical_files_byte_identical':len(basepaths),'historical_results_byte_identical':sum(p.startswith('results/')for p in basepaths),'new_files_only_under_results_0004':len(changes),'indexed_hashes_verified':len(audit['artifacts_sha256']),'rows':len(rows),'row_branch_counts':dict(count),'encoder_calls':len(timings),'refiner_calls':sum(len(r['refiner_seconds'])for r in timings),'controls_independently_equal_numeric':len(controlmax),'maximum_control_numeric_difference':max(controlmax),'checker_imports':imports,'scientific_outer_wall_seconds':j(O/'scientific-outer.json')['wall_seconds'],'scientific_peak_RSS_KiB':rec['peak_rss_kib'],'read_only_audit':True,'no_GT_NPZ_read':True,'no_model_inference':True,'no_cloud_contact':True,'boundaries':['Actual cloud cache and marker not accessible to this local audit; recorded digests plus hashed executed sources inspected.','Raw next-context tensors are not saved; exact context equality is a runtime assertion, not independently reconstructible from CSV.','Parent-observed exit for outermost supervisor not retained; existing README discloses this.','Frozen no-download scientific runner and isolated environment examined; this is not an audit of all cloud account activity.']}
print(json.dumps(report,indent=2))
