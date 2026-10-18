import hashlib,importlib,importlib.metadata as m,json,platform,sys
from pathlib import Path
r={'python_path':sys.executable,'python_version':platform.python_version(),'packages':{}}
for name in ['torch','torchvision','einops','numpy']:
 item={};r['packages'][name]=item
 try:
  d=m.distribution(name);p=Path(d._path);item.update(metadata_version=d.version,dist_info=str(p),metadata_sha256=hashlib.sha256((p/'METADATA').read_bytes()).hexdigest())
 except m.PackageNotFoundError:item.update(metadata_status='MISSING');continue
 try:
  x=importlib.import_module(name)
  if name=='torch':x.set_num_threads(2);x.set_num_interop_threads(2);item.update(threads=x.get_num_threads(),interop_threads=x.get_num_interop_threads(),build_cuda=x.version.cuda)
  p=Path(x.__file__);item.update(import_status='OK',module_version=getattr(x,'__version__',None),module_path=str(p),module_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:item.update(import_status='FAILED',error_type=type(e).__name__,error_message=str(e))
print(json.dumps(r,indent=2))
