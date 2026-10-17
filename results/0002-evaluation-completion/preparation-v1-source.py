"""Bind the reviewed serial scaffold to evaluation allocation; no model/GT access."""
from pathlib import Path
import ast, datetime, fcntl, hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
BASE='6bdce8259d196a832a52ec0ab289066ecdf4907c'
ORDER=['g-031','g-015','g-001','g-005','g-038','g-004','g-021','g-010','g-014','g-007','g-011','g-049','g-028','g-025','g-019','g-006','g-037','g-008','g-027','g-024']
B=Path('results/0002-evaluation-completion');CACHE=Path('scratch/boot-cache')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def active():
 matches=[];patterns=['run_frozen_pilot.py','run_bootstapir_dev.py','supervise_g0','run_g0','postrun_g0','verify_g0','software_g0','publication_audit_g0','run_pilot_preflight.py','scripts/run_experiment.py','execute_evaluation.sh']
 for p in Path('/proc').iterdir():
  if not p.name.isdigit()or int(p.name)==os.getpid():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if argv and b'python'in Path(os.fsdecode(argv[0])).name.encode():
    scripts=[os.fsdecode(a)for a in argv[1:]if any(t in os.fsdecode(a)for t in patterns)]
    if scripts:matches.append({'pid':int(p.name),'entrypoint':Path(scripts[0]).name})
  except(OSError,PermissionError):pass
 return matches
def footprint(g):
 short=g.replace('-','')
 return {'markers':[p.name for p in CACHE.iterdir()if p.suffix=='.json'and(g in p.name or short in p.name)], 'outputs':[str(p.relative_to('..'))for p in Path('..').glob('*/results/0002-pilot-predictions/**/'+g)]}
def check_locks(g):
 for name in ['frozen-v2-research.lock','frozen-v2-'+g.replace('-','')+'.lock']:
  with(CACHE/name).open('a')as z:fcntl.flock(z,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(z,fcntl.LOCK_UN)
def transform(s,g):
 n=g[-3:];i=str(int(n))
 return s.replace('g-016',g).replace('g016','g'+n).replace('video16','video'+i).replace('video-016','video-'+n).replace('[16]','['+i+']').replace("{'16'}","{'"+i+"'}").replace("['16']","['"+i+"']").replace("'archive_index':16","'archive_index':"+i).replace('d6eb151874fbe2afe52651f0049bd2e009afe8d7',BASE)
if sys.argv[1]=='init':
 B.mkdir(parents=True,exist_ok=False);f=load('protocols/0002-freeze-v2.json');m=load(f['manifest_path']);ev=[g for g in m['groups']if g['partition']=='evaluation'];cal=[g['group_id']for g in m['groups']if g['partition']=='calibration']
 assert [g['group_id']for g in ev]==ORDER and all(g['members']==[int(g['group_id'][-3:])]for g in ev)
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()==BASE
 states={g:footprint(g)for g in ORDER};running=active()
 paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],text=True).splitlines()
 historical={p:h(p)for p in paths if p.startswith(('results/','docs/'))}
 for g in cal:assert load(Path('results/0002-pilot-predictions/v2-reconstructed')/g/'receipt.json')['status']=='completed'
 record={'source_commit':BASE,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation':ORDER,'calibration_allocation':cal,'batch_order':ORDER,'initial_processes':running,'initial_group_footprints':states,'scope':'visible processes/worktree outputs/cache reservations, not global or live scheduling attestation','historical_prior_group_files':historical,'historical_scope':'all tracked results and docs at execution main','new_download_bytes':0,'acquisition_ceiling_bytes':1482283622}
 (B/'baseline.json').write_text(json.dumps(record,indent=2)+'\n')
 assert not running and all(not v['markers']and not v['outputs']for v in states.values())
 for g in ORDER:check_locks(g)
 prefix=Path('results/0002-pilot-execution/g-016-20261017')
 templates={k:Path(p).read_text()for k,p in {'wrapper':prefix/'launcher-source.py','supervisor':str(prefix)+'-supervision/supervisor-source.py','verify':prefix/'verification-source.py','software':prefix/'software-checks-source.py','postrun':str(prefix)+'-postrun/postrun-supervisor-source.py','audit':str(prefix)+'-postrun/publication-audit-source.py'}.items()}
 static=[]
 for pos,g in enumerate(ORDER):
  short=g.replace('-','');n=g[-3:]
  for name,key in [('run','wrapper'),('supervise','supervisor'),('verify','verify'),('postrun','postrun'),('publication_audit','audit')]:
   s=transform(templates[key],g)
   if key in ('wrapper','verify'):s=s.replace("'calibration'","'evaluation'")
   if key=='verify':
    a=s.index('preserved={path:');b=s.index("for path,digest in prelaunch",a)
    s=s[:a]+"preserved={path:hashlib.sha256(blob(x['source_commit'],path)).hexdigest()for path in paths if path.startswith(('results/','docs/'))}\n"+s[b:]
    a=s.index("manifest=load(f['manifest_path'])");b=s.index('\nreport=',a)
    s=s[:a]+"manifest=load(f['manifest_path']);ev=[z['group_id']for z in manifest['groups']if z['partition']=='evaluation'];cal=[z['group_id']for z in manifest['groups']if z['partition']=='calibration']\nassert ev=="+repr(ORDER)+" and ev["+str(pos)+"]=='"+g+"' and len(cal)==10\n"+s[b:]
    s=s.replace("'calibration_groups_inferred':7","'calibration_groups_inferred':10").replace("'evaluation_groups_inferred':0","'evaluation_groups_inferred':"+str(pos+1)).replace("'next_calibration_group':cal[7]","'next_evaluation_group':"+('ev['+str(pos+1)+']'if pos+1<len(ORDER)else'None'))
   if key=='postrun'and pos>0:s=s.replace("'software':'scratch/software_"+short+".py',",'')
   if key=='postrun'and pos==len(ORDER)-1:
    s=s.replace("'publication_audit':'scratch/publication_audit_"+short+".py'","'publication_audit':'scratch/publication_audit_"+short+".py','batch_audit':'scratch/batch_audit.py'")
    s=s.replace("'publication_audit':DIRECTORY/'publication-audit-source.py'","'publication_audit':DIRECTORY/'publication-audit-source.py','batch_audit':DIRECTORY/'batch-audit-source.py'")
   if key=='audit':
    # Shared software record always belongs to the first evaluation group.
    s=s.replace(str(Path('results/0002-pilot-execution')/(g+'-20261017-postrun'))+"')if phase=='software'", "results/0002-pilot-execution/g-031-20261017-postrun')if phase=='software'")
    s=s.replace(str(Path('results/0002-pilot-execution')/(g+'-20261017'))+"')if phase=='software'", "results/0002-pilot-execution/g-031-20261017')if phase=='software'")
    s=s.replace('len(b)<100000','len(b)<2*1024**2')
    s=s.replace("'calibration_inferred':7","'calibration_inferred':10").replace("'evaluation_inferred':0","'evaluation_inferred':"+str(pos+1)).replace("v['next_calibration_group']","v['next_evaluation_group']")
   target=Path('scratch')/(name+'_'+short+'.py');target.write_text(s);ast.parse(s);compile(s,str(target),'exec')
   assert 'g-016'not in s and 'g016'not in s and 'd6eb151874fbe2afe52651f0049bd2e009afe8d7'not in s
   if key in ('wrapper','verify'):assert "'calibration'"not in s and "'evaluation'"in s
   if key=='wrapper':
    module=ast.parse(s);pre=next(ast.literal_eval(z.value)for z in module.body if isinstance(z,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PREFLIGHT'for t in z.targets));ast.parse(pre)
    assert "groups[0]['members']==["+str(int(n))+"]"in pre and "groups[0]['partition']=='evaluation'"in pre
    assert "'--group-id','"+g+"'"in s and "'infer'"in s and "'evaluate'"not in s
   if key=='verify':assert "'calibration_groups_inferred':10"in s and "'evaluation_groups_inferred':"+str(pos+1)in s and 'next_calibration_group'not in s
   if key=='audit':assert "g-031-20261017-postrun')if phase=='software'"in s
   static.append({'group':g,'kind':key,'path':str(target),'sha256':h(target),'syntax_and_allocation_checks':'passed'})
  if pos==0:Path('scratch/software_g031.py').write_text(transform(templates['software'],g))
 (B/'static-bindings.json').write_text(json.dumps({'source':BASE,'evaluation_order':ORDER,'scripts':static,'scope':'static syntax/literal allocation/partition/CLI/phase bindings for all 20 groups; scientific files unchanged; no script/model execution'},indent=2)+'\n')
 print(json.dumps({'prepared':ORDER,'static_scripts':len(static),'historical_files':len(historical),'source':BASE}))
else:
 g=sys.argv[1];pos=ORDER.index(g);running=active();state=footprint(g);baseline=load(B/'baseline.json');earlier={}
 for prior in ORDER[:pos]:
  d=Path('results/0002-pilot-execution')/(prior+'-20261017-postrun');r=load(d/'publication_audit.json');assert r['status']=='completed'and r['child_exit_code_observed_by_parent']==0
  for base in[Path('results/0002-pilot-execution')/(prior+'-20261017'),d,Path('results/0002-pilot-execution')/(prior+'-20261017-supervision'),Path('results/0002-pilot-predictions/v2-reconstructed')/prior]:earlier.update({str(p):h(p)for p in base.iterdir()if p.is_file()})
 pre={'source_commit':BASE,'group_id':g,'active_research_processes':running,'prior_group_footprint':state,'historical_prior_group_files':baseline['historical_prior_group_files'],'earlier_batch_artifact_sha256':earlier,'scope':'prelaunch visible process/output/reservation checks; prior completed batch audits checked; no global attestation'}
 with(B/('prelaunch-'+g+'.json')).open('x')as z:json.dump(pre,z,indent=2);z.write('\n')
 assert not running and not state['markers']and not state['outputs'];check_locks(g)
 for p,digest in pre['historical_prior_group_files'].items():assert h(p)==digest
 for p,digest in earlier.items():assert h(p)==digest
 Path('scratch/prelaunch.json').write_text(json.dumps(pre,indent=2)+'\n')
 print(json.dumps({'prepared_group':g,'earlier_batch_artifacts_preserved':len(earlier)}))
