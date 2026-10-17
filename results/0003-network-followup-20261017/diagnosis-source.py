"""One request using unchanged runtime defaults. Never expose environment values."""
import datetime,hashlib,json,os,platform,resource,socket,urllib.request,urllib.error,time
from pathlib import Path
B=Path('results/0003-network-followup-20261017');B.mkdir(exist_ok=True)
with(B/'request-reservation.json').open('x')as f:f.write('{"one_request_only":true}\n')
names=['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','REQUESTS_CA_BUNDLE']
presence={name:bool(os.environ.get(name))for name in names}
url='https://huggingface.co/api/datasets/aharley/pointodyssey/tree/4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb?recursive=false&expand=false'
r={'kind':'single bounded runtime-network diagnosis','source_commit':'b14e759b03ae992682c95b53466a9d220c4e86a1','variable_presence_only':presence,'environment_mode':'unchanged runtime default environment; no variable values logged, no security/route settings changed','url':url,'timeout_seconds':15,'body_cap_bytes':65536,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'received_bytes':0,'frozen_files_modified':False}
start=time.perf_counter();body=b''
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EVR-bounded-network-diagnosis/0003'}),timeout=15)as response:
  r.update(http_status=response.status,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'))
  while len(body)<65536:
   chunk=response.read(min(8192,65536-len(body)))
   if not chunk:r['complete_body']=True;break
   body+=chunk
   if time.perf_counter()-start>15:break
 r['outcome']='http_response_received';r['received_bytes']=len(body)
 if r['http_status']==200:
  data=json.loads(body);r['metadata_summary']=[{k:e[k]for k in ['path','type','size','oid','lfs']if k in e}for e in data]if isinstance(data,list)else{'json_type':type(data).__name__}
except urllib.error.HTTPError as exc:
 r.update(outcome='http_error',http_status=exc.code)
except Exception as exc:
 r.update(outcome='request_failed',error_type=type(exc).__name__,reason_type=type(getattr(exc,'reason',None)).__name__,errno=getattr(getattr(exc,'reason',None),'errno',None))
finally:
 r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,received_bytes=len(body),received_body_sha256=hashlib.sha256(body).hexdigest())
 (B/'diagnosis.json').write_text(json.dumps(r,indent=2)+'\n')
 if body:Path('scratch/0003-hf-metadata.body').write_bytes(body)
print(json.dumps(r,indent=2))
