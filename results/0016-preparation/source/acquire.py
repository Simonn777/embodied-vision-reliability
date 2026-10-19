"""Bounded isolated build inputs; never reads scientific data."""
from pathlib import Path
import hashlib,json,time,urllib.request

ROOT=Path(__file__).resolve().parent
ASSETS=[
 ('cmake.whl','https://files.pythonhosted.org/packages/7f/09/516b0d709672bc430eb13278f0316acd34869269447744f5d136daeef689/cmake-3.31.6-py3-none-macosx_10_10_universal2.whl',47224338,'da9d4fd9abd571fd016ddb27da0428b10277010b23bb21e3678f8b9e96e1686e'),
 ('pybind11.whl','https://files.pythonhosted.org/packages/cd/8a/37362fc2b949d5f733a8b0f2ff51ba423914cabefe69f1d1b6aab710f5fe/pybind11-3.0.1-py3-none-any.whl',293611,'aa8f0aa6e0a94d3b64adfc38f560f33f15e589be2175e103c0a33c6bce55ee89'),
 ('poselib.tar.gz','https://codeload.github.com/PoseLib/PoseLib/tar.gz/6a636b1d84273d5e3ba97676463476569611572a',5*1024*1024,None),
 ('eigen.tar.gz','https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz',5*1024*1024,None)]
with (ROOT/'started.json').open('x') as f:
 json.dump({'purpose':'Official current PoseLib comparison preparation, no science','max_asset_body_bytes':sum(x[2] for x in ASSETS),'metadata_allowance_bytes':2*1024*1024,'assets':ASSETS,'prior_remaining_bytes':2877929334},f,indent=2)
receipts=[]
for name,url,cap,expected in ASSETS:
 start=time.monotonic(); n=0; digest=hashlib.sha256(); row={'name':name,'url':url,'cap':cap,'status':'started'}; receipts.append(row)
 try:
  with (ROOT/name).open('xb') as f, urllib.request.urlopen(url,timeout=30) as r:
   if r.status !=200: raise RuntimeError('Unexpected HTTP status')
   row['headers']={k:r.headers.get(k) for k in ('Content-Length','ETag','Last-Modified')}
   while True:
    if time.monotonic()-start>180: raise TimeoutError('total download time')
    b=r.read(min(1024*1024,cap-n+1))
    if not b: break
    n+=len(b); digest.update(b)
    if n>cap: raise RuntimeError('download cap exceeded')
    f.write(b)
  if expected and (n!=cap or digest.hexdigest()!=expected): raise RuntimeError('package size/hash mismatch')
  row['status']='completed'
 except BaseException as e:
  row['error']=type(e).__name__+': '+str(e); raise
 finally:
  row.update(bytes_received=n,sha256=digest.hexdigest(),seconds=time.monotonic()-start)
  with (ROOT/(name+'.receipt.json')).open('x') as f: json.dump(row,f,indent=2)
  print(json.dumps(row),flush=True)
