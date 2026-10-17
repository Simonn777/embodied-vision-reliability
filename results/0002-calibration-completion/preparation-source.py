"""Prepare audited per-group bindings; no model import, labels or metrics."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
BASE='d6eb151874fbe2afe52651f0049bd2e009afe8d7'
ORDER=['g-016','g-032','g-035','g-026'];OLD=['g-033','g-013','g-022','g-040','g-039','g-020']
B=Path('results/0002-calibration-completion')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def active():
 matches=[];patterns=['run_frozen_pilot.py','run_bootstapir_dev.py','supervise_g0','run_g0','postrun_g0','verify_g0','software_g0','publication_audit_g0','run_pilot_preflight.py','scripts/run_experiment.py']
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
 return {'markers':[p.name for p in Path('scratch/boot-cache').iterdir()if g in p.name or short in p.name], 'outputs':[str(p.relative_to('..'))for p in Path('..').glob('*/results/0002-pilot-predictions/**/'+g)]}
def transform(text,g):
 n=g[-3:];i=str(int(n))
 return text.replace('g-020',g).replace('g020','g'+n).replace('video20','video'+i).replace('video-020','video-'+n).replace('[20]','['+i+']').replace("{'20'}","{'"+i+"'}").replace("['20']","['"+i+"']").replace("'archive_index':20","'archive_index':"+i).replace('697371746cbe3bd89d37472ec5889cb824db4f70',BASE)
if sys.argv[1]=='init':
 B.mkdir(parents=True,exist_ok=False);f=json.loads(Path('protocols/0002-freeze-v2.json').read_text());m=json.loads(Path(f['manifest_path']).read_text())
 cal=[g['group_id']for g in m['groups']if g['partition']=='calibration'];assert cal==OLD+ORDER
 assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
 assert subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()==BASE
 states={g:footprint(g)for g in ORDER};running=active()
 paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],text=True).splitlines();tokens=[x for g in OLD for x in[g,g.replace('-','')]]
 historical={p:h(p)for p in paths if any(t in p for t in tokens)}
 record={'source_commit':BASE,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation':cal,'batch_order':ORDER,'initial_processes':running,'initial_group_footprints':states,'scope':'visible processes/worktree outputs/cache reservations, not global or live scheduling attestation','historical_prior_group_files':historical,'new_download_bytes':0,'acquisition_ceiling_bytes':1482283622}
 (B/'baseline.json').write_text(json.dumps(record,indent=2)+'\n')
 assert not running and all(not v['markers']and not v['outputs']for v in states.values())
 templates={k:Path(p).read_text()for k,p in {'wrapper':'results/0002-pilot-execution/g-020-20261017/launcher-source.py','supervisor':'results/0002-pilot-execution/g-020-20261017-supervision/supervisor-source.py','verify':'results/0002-pilot-execution/g-020-20261017/verification-source.py','software':'results/0002-pilot-execution/g-020-20261017/software-checks-source.py','postrun':'results/0002-pilot-execution/g-020-20261017-postrun/postrun-supervisor-source.py','audit':'results/0002-pilot-execution/g-020-20261017-postrun/publication-audit-source.py'}.items()}
 for pos,g in enumerate(ORDER):
  short=g.replace('-','');ordinal=6+pos;n=g[-3:]
  for name,key in [('run','wrapper'),('supervise','supervisor'),('verify','verify'),('postrun','postrun'),('publication_audit','audit')]:
   s=transform(templates[key],g)
   if key=='verify':
    # The six committed groups are the immutable historical baseline.
    a=s.index("preserved={path:");b=s.index("for path,digest in prelaunch",a)
    s=s[:a]+"preserved={path:hashlib.sha256(blob(x['source_commit'],path)).hexdigest() for path in paths if any(token in path for token in "+repr([t for o in OLD for t in[o,o.replace('-','')]])+")}\n"+s[b:]
    a=s.index('assert cal[:6]');b=s.index('\n',a)
    s=s[:a]+'assert cal[:'+str(ordinal+1)+']=='+repr((OLD+ORDER)[:ordinal+1])+s[b:]
    s=s.replace("'calibration_groups_inferred':6","'calibration_groups_inferred':"+str(ordinal+1))
    s=s.replace("'next_calibration_group':cal[6]","'next_calibration_group':"+('cal['+str(ordinal+1)+']'if ordinal+1<10 else'None'))
    s=s.replace("manifest=load(f['manifest_path'])", "for prior_path,prior_digest in prelaunch['earlier_batch_artifact_sha256'].items():assert h(prior_path)==prior_digest\nmanifest=load(f['manifest_path'])")
   if key=='postrun'and pos>0:
    s=s.replace("'software':'scratch/software_"+short+".py',",'')
   if key=='postrun'and pos==3:
    s=s.replace("'publication_audit':'scratch/publication_audit_"+short+".py'", "'publication_audit':'scratch/publication_audit_"+short+".py','batch_audit':'scratch/batch_audit.py'")
    s=s.replace("'publication_audit':DIRECTORY/'publication-audit-source.py'", "'publication_audit':DIRECTORY/'publication-audit-source.py','batch_audit':DIRECTORY/'batch-audit-source.py'")
   if key=='audit':
    s=s.replace("'calibration_inferred':6","'calibration_inferred':"+str(ordinal+1))
    s=s.replace("r=load(P/(phase+'.json'))", "phase_dir=Path('results/0002-pilot-execution/g-016-20261017-postrun')if phase=='software'else P\n source_dir=Path('results/0002-pilot-execution/g-016-20261017')if phase=='software'else A\n r=load(phase_dir/(phase+'.json'))")
    s=s.replace("h(P/'postrun-supervisor-source.py')", "h(phase_dir/'postrun-supervisor-source.py')").replace("h(A/source)","h(source_dir/source)").replace("h(P/(phase+'.'+stream+'.log'))", "h(phase_dir/(phase+'.'+stream+'.log'))")
   target=Path('scratch')/(name+'_'+short+'.py');target.write_text(s);compile(s,str(target),'exec')
  if pos==0:Path('scratch/software_g016.py').write_text(transform(templates['software'],g))
 print(json.dumps({'prepared':ORDER,'historical_files':len(historical),'source':BASE}))
else:
 g=sys.argv[1];pos=ORDER.index(g);running=active();state=footprint(g);baseline=json.loads((B/'baseline.json').read_text());earlier={}
 for prior in ORDER[:pos]:
  d=Path('results/0002-pilot-execution')/(prior+'-20261017-postrun');r=json.loads((d/'publication_audit.json').read_text());assert r['status']=='completed'and r['child_exit_code_observed_by_parent']==0
  for base in[Path('results/0002-pilot-execution')/(prior+'-20261017'),d,Path('results/0002-pilot-execution')/(prior+'-20261017-supervision'),Path('results/0002-pilot-predictions/v2-reconstructed')/prior]:
   earlier.update({str(p):h(p)for p in base.iterdir()if p.is_file()})
 pre={'source_commit':BASE,'group_id':g,'active_research_processes':running,'prior_group_footprint':state,'historical_prior_group_files':baseline['historical_prior_group_files'],'earlier_batch_artifact_sha256':earlier,'scope':'prelaunch current process/output/reservation checks; prior completed batch audits checked; no global attestation'}
 with (B/('prelaunch-'+g+'.json')).open('x')as z:json.dump(pre,z,indent=2);z.write('\n')
 assert not running and not state['markers']and not state['outputs']
 Path('scratch/prelaunch.json').write_text(json.dumps(pre,indent=2)+'\n')
 print(json.dumps({'prepared_group':g,'earlier_batch_artifacts_preserved':len(earlier)}))
