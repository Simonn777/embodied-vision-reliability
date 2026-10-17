"""Export frozen row-builder outputs for independent review; never evaluate/infer."""
import csv, hashlib, json, sys
from pathlib import Path
from evr.pilot_frozen import verify_freeze, CANONICAL_RECEIPT, load_verified_dataset, prediction_rows
B=Path('results/0002-frozen-evaluation-v2-audit-20261017');D=B/'rows';D.mkdir(exist_ok=False)
f=verify_freeze(Path.cwd(),CANONICAL_RECEIPT);m=json.loads(Path(f['manifest_path']).read_text())
cache=Path('scratch/boot-cache');assert(cache/'rgb.zip').stat().st_size==187291581
assert hashlib.sha256((cache/'rgb.zip').read_bytes()).hexdigest()==f['archive_sha256']
dataset=load_verified_dataset(cache,m);counts={};artifacts={}
header=['archive_index','group_id','partition','point','frame','tie','post_query','visible','opportunity','coordinate_valid','logit_valid','available','error_px','failure','occlusion_logit','expected_dist_logit','x','y']+['score_'+k for k in ['native','augmented','visibility','local','random','oracle']]
for partition in ['calibration','evaluation']:
 rows,groups=prediction_rows(Path('results/0002-pilot-predictions/v2-reconstructed'),m,partition,dataset,f['input_commit']);counts[partition]=len(rows)
 for g in groups:
  members=next(z['members']for z in m['groups']if z['group_id']==g);raw={}
  for i in members:
   with(Path('results/0002-pilot-predictions/v2-reconstructed')/g/f'video-{i:03}.csv').open(newline='')as z:
    for r in csv.DictReader(z):raw[i,int(r['point']),int(r['frame'])]=r
  path=D/(partition+'-'+g+'.csv')
  with path.open('x',newline='')as out:
   w=csv.DictWriter(out,header,lineterminator='\n');w.writeheader()
   for r in rows:
    if r['group_id']!=g:continue
    source=raw[r['archive_index'],r['point'],r['frame']]
    record={k:r[k]for k in header if k in r};record['partition']=partition
    record.update({k:source[k]for k in ['x','y','expected_dist_logit']});record.update({'score_'+k:v for k,v in r['scores'].items()})
    record={k:(int(v)if type(v)is bool else v)for k,v in record.items()};w.writerow(record)
  assert path.stat().st_size<1024**2;artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
assert 'torch'not in sys.modules and not any(k.startswith('tapnet')for k in sys.modules)
report={'status':'passed','row_counts':counts,'files':len(artifacts),'sha256':artifacts,'scope':'second restricted decode and frozen prediction_rows/rows_from_arrays only; no evaluate, inference, model/torch import, weights load or GT xy publication','epe_visibility_source':'frozen row builder and verified archive/identities; independent checker consumes these labels/EPE, not an independent annotation re-labeling'}
(B/'row-export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='sha256'},indent=2))
