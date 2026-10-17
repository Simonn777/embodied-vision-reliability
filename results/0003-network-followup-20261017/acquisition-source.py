"""One public sample GET on unchanged configured runtime route; no retries."""
import datetime,fcntl,hashlib,json,os,time,urllib.request,urllib.error
from pathlib import Path
B=Path('results/0003-network-followup-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');N=3324284510;EXPECTED='436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f'
locks=[]
for name in ['frozen-v2-research.lock','0003-sample.lock']:
 f=(C.parent/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
with(C/'sample-acquisition-reservation.json').open('x')as f:json.dump({'one_attempt_only':True,'source_commit':'b14e759b03ae992682c95b53466a9d220c4e86a1'},f)
path=C/'sample-v1.2.tar.gz.partial';url='https://huggingface.co/datasets/aharley/pointodyssey/resolve/4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb/sample.tar.gz'
r={'url':url,'expected_bytes':N,'expected_sha256':EXPECTED,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'received_bytes':0,'route':'unchanged runtime-configured proxy/CA defaults; values not logged','GET_attempts':1,'retries':0,'body_complete':False,'whole_archive_verified':False,'budget_ceiling_before_new_data':1499060838,'budget_bytes':10737418240,'total_wall_cap_seconds':840}
h=hashlib.sha256();start=time.perf_counter()
def save():
 r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,sha256_received=h.hexdigest(),new_cumulative_ceiling_bytes=1499060838+2328+r['received_bytes']);(B/'sample-acquisition.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 with path.open('xb')as out:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EVR-bounded-public-acquisition/0003'}),timeout=20)as response:
   r.update(http_status=response.status,content_length=response.headers.get('Content-Length'),content_type=response.headers.get('Content-Type'))
   if r['http_status']!=200:raise ValueError('unexpected sample HTTP status')
   if r['content_length']is not None and int(r['content_length'])!=N:raise ValueError('unexpected declared object length')
   while r['received_bytes']<N:
    if time.perf_counter()-start>840:raise TimeoutError('bounded acquisition wall cap')
    chunk=response.read(min(2**20,N-r['received_bytes']))
    if not chunk:break
    out.write(chunk);h.update(chunk);r['received_bytes']+=len(chunk)
    if r['received_bytes']%(64*2**20)<2**20:save()
   r['body_complete']=r['received_bytes']==N
   r['whole_archive_verified']=r['body_complete']and h.hexdigest()==EXPECTED
   if not r['whole_archive_verified']:raise ValueError('incomplete archive or digest mismatch')
 r['status']='verified_complete';path.rename(C/'sample-v1.2.tar.gz')
except urllib.error.HTTPError as exc:r.update(status='blocked',error_type='HTTPError',http_status=exc.code)
except Exception as exc:r.update(status='failed',error_type=type(exc).__name__,reason_type=type(getattr(exc,'reason',None)).__name__,errno=getattr(getattr(exc,'reason',None),'errno',None))
finally:
 save()
 for f in locks:f.close()
print(json.dumps(r,indent=2));raise SystemExit(0 if r['whole_archive_verified']else 1)
