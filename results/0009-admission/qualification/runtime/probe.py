"""Fresh-process import/capability presence only; no geometry/data/model calls."""
import hashlib,importlib,json,platform,sys,traceback
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    name=sys.argv[1]
    try:
        module=importlib.import_module(name)
        result={'status':'available','module':name,'version':getattr(module,'__version__',None),'module_file':str(Path(module.__file__).resolve()),'module_file_sha256':sha(module.__file__),'hash_scope':'Selected module entry file only, not the entire dependency closure.','capabilities':{}}
        if name=='cv2':result['capabilities']={k:hasattr(module,k) for k in ('USAC_MAGSAC','findHomography','SIFT_create')}
        if name=='scipy':
            spatial=importlib.import_module('scipy.spatial')
            result['capabilities']['scipy.spatial.cKDTree']=hasattr(spatial,'cKDTree')
            result['spatial_module_file']=str(Path(spatial.__file__).resolve());result['spatial_module_file_sha256']=sha(spatial.__file__)
        code=0
    except Exception as exc:
        result={'status':'import_failed','module':name,'error_type':type(exc).__name__,'error':str(exc)}
        traceback.print_exc();code=1
    result.update(python_version=sys.version.split()[0],python_executable=sys.executable,platform=platform.platform(),probe_source_sha256=sha(__file__),scope='Import and attribute existence only; no function, model or dataset trial.')
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
