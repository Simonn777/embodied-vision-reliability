import datetime,hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
B=Path('results/0003-network-followup-20261017');url='https://raw.githubusercontent.com/y-zheng18/point_odyssey/a053342a72fed32af4134b4ca0c44a78a3ac0fdb/export_annotation.py';r={'url':url,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};start=time.perf_counter();body=b''
try:
 with urllib.request.urlopen(url,timeout=15)as response:r['http_status']=response.status;body=response.read(65536)
 Path('../bootstapir-cache/0003-pointodyssey/official-export.body').write_bytes(body)
except urllib.error.HTTPError as e:r.update(error_type='HTTPError',http_status=e.code)
except Exception as e:r['error_type']=type(e).__name__
r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,bytes_received=len(body),received_sha256=hashlib.sha256(body).hexdigest(),source_body_published=False);(B/'export-source-http.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
