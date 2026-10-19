"""Saved-text transport only. Never invokes scientific code, imports a model or opens image/weight arrays."""
import base64,gzip,hashlib,json,math,re
from pathlib import Path
b=Path('/workspace/scratch/a349d03a0e0a');c=b/'scratch/0010-headroom-control';r=b/'research-0010-headroom';o=b/'scratch/0010-headroom-output'
def h(raw):return hashlib.sha256(raw).hexdigest()
def spec(p):return dict(bytes=p.stat().st_size,sha256=h(p.read_bytes()))
def write(name,value):(c/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
f=json.loads((c/'preflight.json').read_text());d={k:v for k,v in f.items() if k!='preserved'};d.update(original_full_preflight=spec(c/'preflight.json'),derivation='Byte-preservation map omitted only from this public summary; original retained privately. All entries independently verified after study.',preserved_count=len(f['preserved']));write('preflight-public.json',d)
command=json.loads((b/'scratch/0010-capability-control/imports.command.json').read_text());argv=[str(Path('/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python')),'experiments/0010-headroom/run.py','launch','--plan','experiments/0010-headroom/plan.json','--plan-sha256','d8b1891dd99ea6b7e1680b830da478f81b9e37cedf11933f8c878511cfca21ce','--assetdir',str(b/'clip-0010-assets'),'--output',str(o),'--supervision',str(b/'scratch/0010-headroom-supervision'),'--python','/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python']
write('launch-observed-tool-return.json',dict(record_type='Assistant transcription of actual tool return, not an invented outer wait4 receipt',launch_tool_chunk='1828fa',session_id=64576,final_tool_chunk='aa1f79',actual_tool_exit=0,actual_tool_stdout='launch_actual_exit=0\n',initial_tool_wait_seconds=1.001191611,final_poll_wait_seconds=0.846085639,tool_wait_boundary='Tool wait chunks are not the full invocation wall time. Real complete child wall/CPU/wait4/RSS is in frozen supervision/execution.json; no extra observer was added.',cwd=str(r),argv=argv,PYTHONPATH=command['PYTHONPATH'],numeric_threads=4,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',source_main=f['head'],source_commit=f['source'],source_run_sha256=spec(r/'experiments/0010-headroom/run.py')['sha256'],supervision=spec(b/'scratch/0010-headroom-supervision/execution.json'),stdout=spec(c/'launch.stdout.log'),stderr=spec(c/'launch.stderr.log')))
lock=json.loads((o/'selection-lock.json').read_text());classes=json.loads((r/'experiments/0010/config.json').read_text())['class_order'];write('saved-row-identities.json',dict(derivation='Names/order from already sealed selection lock; labels decoded only for saved review, after frozen scoring.',selection_lock=spec(o/'selection-lock.json'),class_order=classes,rows=[dict(id=i,filename=n,batch=k//64,label=classes.index(n.rsplit('_',1)[0])) for k,(i,n) in enumerate(zip(lock['ids'],lock['names']))]))
(c/'README.txt').write_text('''0010 fixed real-data development screen — NO_GO

AI-assisted execution at main49c0c0b22f89e5d395130dec63bafad9377ba5c5;
source d14134c34e7fa097de5556aca162af962d920880, canonical receipt-only
child de07509be1f314da2b8803a672b2d266b234f723. Parent verified exact CI
39181113693/39181219870 before this single launch.

Actual outer tool exit0; frozen supervisor child exit0/raw wait0,28.043947359 s,
user89.846956 s/system1.008021 s, peak1,321,436 KiB. Limit900 s,4 CPU,8 GiB.
32 batch4 image calls,128 images,0 text forwards;132 complete fitted states,
264 authored parity executions. All predictions sealed before1 score.

Correct/128: ZS65 (50.78125%),native alpha1 62 (48.4375%),native alpha4 52
(40.625%),exact LOO alpha1 61 (47.65625%),OTHER-out alpha1 63 (49.21875%).
LOO corrections/harms/net: ZS7/11/-4;native1 2/3/-1;native4 10/1/+9;
OTHER1/3/-2. Net vs ZS by fixed64-image batch:-1,-3.
Failed:net_eight_over_best,both_batches_positive. The singleton no-harm
boolean is vacuously true:0 singleton or doubleton samples; no singleton safety
is established.128/128 are in occupancy>=3. Decision NO_GO,not LOW_HEADROOM.
Full controls,batch and occupancy results are in unchanged scores.json.

Independent saved-only checker:11,441 checks passed, full fit lineage,
seals,input/feature hashes, paired arithmetic, all gates and9,818 preexisting
file hashes preserved. Image normalization unit-norm error1.1920928955078125e-7.
No repeated inference,fit,formal score,capability,guard or admission tests.
Private NPZs are NOT exported; original full file hashes/sizes and array shapes
are in saved-review.json. All old files/markers/partials/failures remain.

Pre-scientific preflight.py exited1 when trying to make a system package path
relative to project root. Its original source and stderr/stdout are retained.
preflight-v2.py corrected only inventory path representation and passed before
any scientific marker. No frozen code or scientific threshold changed. Prior
before-probe exit1 and mapping failure are untouched. Direct body charge remains
456,087,034 B; new scientific acquisition0 B; historical all-HTTP total UNKNOWN.

This is one fixed train-only EuroSAT development sample,not independent scenes,
not a population CI,not an efficient deployment or novelty/manuscript claim.
LOO and OTHER-out have different readout peer sets. Stop this exact direction
for this cohort; no outcome-driven alpha,prompt,cohort or threshold rescue.

Package contains saved text only, exact per-file bytes/SHA/base64. No raw JPEG,
checkpoint,wheel,overlay,feature NPZ or artificial NPZ. Project/runtime paths
remain as cloud execution evidence. Full large baseline preflight map stays
private; public derivative names its original SHA/size. Prior65/102 evidence
sets are referenced through frozen Git hashes,not recopied. Transport JSON,
gzip,base64 wrappers and metadata are self-excluded from their file manifest.
Decode base64,gzip-decompress,parse JSON; verify each base64 member against its
bytes and SHA256. All source original bytes remain unchanged.
''')
paths=[]
for p in sorted(c.iterdir()):
 if p.is_file() and p.suffix in ('.py','.log','.json','.txt') and p.name not in ['preflight.json','transfer.metadata.json','transfer.base64.txt'] and not p.name.startswith('transfer.'):paths.append(p)
for p in sorted(o.iterdir()):
 if p.suffix=='.json':paths.append(p)
for p in sorted((b/'scratch/0010-headroom-supervision').iterdir()):
 if p.is_file():paths.append(p)
for p in ['0010-headroom-launch.started.json','0010-headroom-started.json']:paths.append(b/'clip-0010-assets'/p)
for p in ['protocols/0010-real-data-headroom-freeze.json','experiments/0010-headroom/plan.json']:paths.append(r/p)
files={}
for p in paths:
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<1024**2
 raw=p.read_bytes();raw.decode('utf-8');assert not re.search(rb'https://[^\s\"<>]+[?&](?:' + b'X-Amz-' + rb'Signature|X-Goog-Signature|token|sig|Signature)=',raw)
 name=str(p.relative_to(b));assert name not in files and '..' not in Path(name).parts
 files[name]=dict(bytes=len(raw),sha256=h(raw),base64=base64.b64encode(raw).decode())
value=dict(schema='0010-headroom-saved-evidence-transfer-v1',files=files,file_count=len(files),total_original_bytes=sum(v['bytes'] for v in files.values()),omitted_full_preflight=spec(c/'preflight.json'),private_npz_not_exported={p.name:spec(p) for p in o.glob('*.npz')},self_exclusion='transfer wrappers,metadata/base64 files are excluded; no circular self hash',originals='All included original files are byte-exact; preflight-public.json is an explicitly labeled inventory-summary derivative, and launch-observed-tool-return.json is an assistant transcription of actual tool output.')
raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();gz=gzip.compress(raw,mtime=0);encoded=base64.b64encode(gz).decode();roundtrip=json.loads(gzip.decompress(gz))
for name,d in roundtrip['files'].items():a=base64.b64decode(d['base64']);assert len(a)==d['bytes'] and h(a)==d['sha256'] and a==(b/name).read_bytes()
(c/'transfer.json.gz').write_bytes(gz);(c/'transfer.base64.txt').write_text(encoded)
meta=dict(schema=value['schema'],file_count=len(files),total_original_bytes=value['total_original_bytes'],json_bytes=len(raw),json_sha256=h(raw),gzip_bytes=len(gz),gzip_sha256=h(gz),base64_chars=len(encoded),chunk_size=12000,chunks=math.ceil(len(encoded)/12000),roundtrip='PASS',private_npz_not_exported=value['private_npz_not_exported']);write('transfer.metadata.json',meta);print(json.dumps(meta))
