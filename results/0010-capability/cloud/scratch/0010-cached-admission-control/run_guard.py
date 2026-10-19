import hashlib,sys,json
from pathlib import Path
from types import ModuleType
b=Path('/workspace/scratch/a349d03a0e0a');r=b/'research-0010-cached-admission'
p=r/'experiments/0010-solver/supervise_0009.py';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f'
m=ModuleType('_guard_supervisor');m.__file__=str(p);exec(compile(raw,str(p),'exec'),m.__dict__)
cmd=[sys.executable,str(r/'experiments/0010-guard/guard.py'),'--base',str(b),'--repo',str(b/'research-0010-capability'),'--output',str(b/'scratch/0010-cached-guard')]
d=m.supervise(cmd,b/'scratch/0010-cached-guard-supervision',seconds=120,cpu_ids=[0,1,2,3]);print(json.dumps({k:d[k] for k in ('child_exit_code','supervisor_exit_code')}));raise SystemExit(d['supervisor_exit_code'])
