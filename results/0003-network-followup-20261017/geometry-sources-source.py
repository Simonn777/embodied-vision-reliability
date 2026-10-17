"""Pinned public source documents for the authorized geometry audit, once each."""
import datetime,hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
B=Path('results/0003-network-followup-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');revision='a053342a72fed32af4134b4ca0c44a78a3ac0fdb'
urls={'tracking':f'https://raw.githubusercontent.com/y-zheng18/point_odyssey/{revision}/utils/gen_tracking_indoor.py','scene_export':f'https://raw.githubusercontent.com/y-zheng18/point_odyssey/{revision}/utils/export_scene.py'}
events=[]
for name,url in urls.items():
 e={'url':url,'name':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes_received':0};t=time.perf_counter();body=b''
 try:
  with urllib.request.urlopen(url,timeout=15)as r:
   e['http_status']=r.status;body=r.read(1048576);e['bytes_received']=len(body);(C/('official-'+name+'.body')).write_bytes(body)
 except urllib.error.HTTPError as exc:e.update(error_type='HTTPError',http_status=exc.code)
 except Exception as exc:e.update(error_type=type(exc).__name__,reason_type=type(getattr(exc,'reason',None)).__name__)
 e.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t,received_sha256=hashlib.sha256(body).hexdigest());events.append(e)
(B/'geometry-source-http.json').write_text(json.dumps({'events':events,'bytes_received':sum(x['bytes_received']for x in events),'default_route_retained':True,'body_policy':'source bodies cached untracked only, no code executed or source pages republished'},indent=2)+'\n');print(json.dumps(events,indent=2))
