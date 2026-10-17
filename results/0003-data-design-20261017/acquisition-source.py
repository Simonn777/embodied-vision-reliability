"""Bounded unauthenticated primary-source acquisition. No environment credentials."""
import datetime,hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
B=Path('results/0003-data-design-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');C.mkdir(exist_ok=False)
with(C/'reservation.json').open('x')as f:json.dump({'status':'source audit started','source':'a2d4827c8d83d3745989d57c901dcddf16642e06'},f)
P='a053342a72fed32af4134b4ca0c44a78a3ac0fdb';H='4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb';T='730cda1c730877cfedbe01bf87fb1cadb78a565d'
urls={'project':'https://pointodyssey.com/','point_readme':f'https://raw.githubusercontent.com/y-zheng18/point_odyssey/{P}/README.md','reprojection':f'https://raw.githubusercontent.com/y-zheng18/point_odyssey/{P}/utils/reprojection.py','point_tree':f'https://api.github.com/repos/y-zheng18/point_odyssey/git/trees/{P}?recursive=1','hf_tree':f'https://huggingface.co/api/datasets/aharley/pointodyssey/tree/{H}?recursive=false&expand=false','hf_card':f'https://huggingface.co/datasets/aharley/pointodyssey/raw/{H}/README.md','tap_readme':f'https://raw.githubusercontent.com/google-deepmind/tapnet/{T}/README.md','tap3d_readme':f'https://raw.githubusercontent.com/google-deepmind/tapnet/{T}/tapnet/tapvid3d/README.md','tap_tree':f'https://api.github.com/repos/google-deepmind/tapnet/git/trees/{T}?recursive=1','tap3d_license':f'https://raw.githubusercontent.com/google-deepmind/tapnet/{T}/tapnet/tapvid3d/LICENSE'}
events=[]
def save():
 (B/'http-audit.json').write_text(json.dumps({'events':events,'new_http_body_bytes':sum(e['bytes_received']for e in events),'cache_scope':'external pages/archive stay untracked; only hashes and derived metadata published'},indent=2)+'\n')
def get(name,url,method='GET',limit=1048576,wall=30):
 event={'name':name,'url':url,'method':method,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes_received':0};events.append(event);start=time.perf_counter();h=hashlib.sha256()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,method=method,headers={'User-Agent':'EVR-public-research-audit/0003'}),timeout=15)as r:
   event.update(http_status=r.status,content_length=r.headers.get('Content-Length'),content_type=r.headers.get('Content-Type'))
   with(C/(name+'.body')).open('xb')as f:
    while True:
     chunk=r.read(min(1024**2,limit-event['bytes_received']))
     if not chunk:break
     f.write(chunk);h.update(chunk);event['bytes_received']+=len(chunk)
     if event['bytes_received']>=limit or time.perf_counter()-start>wall:break
   event['sha256_received']=h.hexdigest();event['body_complete']=method=='HEAD'or(event['content_length']is not None and event['bytes_received']==int(event['content_length']))
 except Exception as exc:
  event.update(error_type=type(exc).__name__,error=str(exc).split('?')[0][:250],sha256_received=h.hexdigest(),body_complete=False)
 finally:event.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start);save()
 return event
for name,url in urls.items():get(name,url)
# Header identity checked before any large body. Exactly one acquisition attempt.
u=f'https://huggingface.co/datasets/aharley/pointodyssey/resolve/{H}/sample.tar.gz'
e=get('sample_head',u,'HEAD')
if e.get('http_status')==200:
 e=get('sample_archive',u,limit=3324284510,wall=180)
 e['expected_full_length']=3324284510;e['expected_full_sha256']='436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f'
 e['whole_archive_verified']=e['bytes_received']==e['expected_full_length']and e['sha256_received']==e['expected_full_sha256'];save()
 if e['bytes_received']==3324284510 and not e['whole_archive_verified']:raise ValueError('archive digest mismatch')
print(json.dumps({'http_events':len(events),'received_bytes':sum(e['bytes_received']for e in events),'successful_names':[e['name']for e in events if e.get('http_status')==200]},indent=2))
