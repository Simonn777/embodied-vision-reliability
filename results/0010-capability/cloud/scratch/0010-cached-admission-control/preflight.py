from pathlib import Path
import json,hashlib,subprocess,gzip,datetime
b=Path('/workspace/scratch/a349d03a0e0a');r=b/'research-0010-cached-admission';c=b/'scratch/0010-cached-admission-control'
sha=lambda v:hashlib.sha256(v).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=r)
main='3b874f53b44fb2b3a622caf624a1c0ffe2e802c4';src='4f3eedf1c614dd2b7ef95c1366994b601e273a09';rec='9fda4201231e7bdda2547e673db0bb45723f27d5';p='protocols/0010-cached-probe-solver-freeze.json'
assert git('rev-parse','HEAD').decode().strip()==main and not git('status','--porcelain')
assert git('rev-list','--parents','-n','1',rec).decode().split()==[rec,src]
assert git('diff-tree','--no-commit-id','--name-only','-r',src,rec).decode().splitlines()==[p]
subprocess.run(['git','merge-base','--is-ancestor',rec,main],cwd=r,check=True)
raw=(r/p).read_bytes();assert sha(raw)=='1f745cc649436664ca257ebdf10640921918121894452ab02e26036d56bcca3f'
assert all(raw==git('show',x+':'+p) for x in (rec,main,''))
f=json.loads(raw);assert f['source_commit']==src and len(f['inputs'])==18
records={}
for n,s in f['inputs'].items():
 d=(r/n).read_bytes();assert len(d)==s['bytes'] and sha(d)==s['sha256'];assert hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()==s['git_blob_sha1'];assert all(d==git('show',x+':'+n) for x in (src,main,''));records[n]=s
old=json.loads(gzip.decompress((b/'scratch/0010-partial-review/transfer.json.gz').read_bytes()));preserved={}
for n,s in old['files'].items():
 rel=n.replace('control/','scratch/0010-capability-control/',1) if n.startswith('control/') else n.replace('partial-review/','scratch/0010-partial-review/',1) if n.startswith('partial-review/') else n
 d=(b/rel).read_bytes();assert sha(d)==s['sha256'] and len(d)==s['bytes'];preserved[rel]={'bytes':len(d),'sha256':sha(d)}
d={'status':'PASS','main':main,'source':src,'receipt_commit':rec,'receipt_sha256':sha(raw),'inputs':records,'old_export_files':preserved,'old_export_count':len(preserved),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (c/'preflight.json').open('x') as out:json.dump(d,out,indent=2);out.write('\n')
print(json.dumps({'status':'PASS','frozen_inputs':len(records),'old_files_preserved':len(preserved)}))
