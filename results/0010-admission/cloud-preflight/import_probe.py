"""Import-only package qualification; no model construction, data or tensors."""
import hashlib,importlib,importlib.metadata,json,sys,traceback
from pathlib import Path
name=sys.argv[1]
try:
 module=importlib.import_module(name)
 path=Path(module.__file__).resolve()
 distribution='Pillow' if name=='PIL' else name
 try:version=importlib.metadata.version(distribution)
 except importlib.metadata.PackageNotFoundError:version=None
 result={'package':name,'status':'available','distribution_version':version,'module_version':getattr(module,'__version__',None),'module_path':str(path),'module_entry_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 if name=='torch':result.update(cpu_only=module.version.cuda is None,numeric_threads=module.get_num_threads(),interop_threads=module.get_num_interop_threads())
 print(json.dumps(result,allow_nan=False));sys.exit(0)
except Exception as e:
 print(json.dumps({'package':name,'status':'import_failed','error_type':type(e).__name__,'error':str(e)}));traceback.print_exc();sys.exit(1)
