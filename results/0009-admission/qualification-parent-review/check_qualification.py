"""Read-only qualification archive audit. Never executes archived probes or recipes.

Private retrieval arguments are used only to verify redaction reversal and actual
command-tool observations. Their contents and prefixes are never emitted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True)
parser.add_argument('--private-runtime', required=True)
parser.add_argument('--private-directory', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo = Path(args.repo).resolve()
base = repo/'results/0009-admission/qualification'
review = Path(__file__).resolve().parent
output = Path(args.output)
if output.exists():
    raise FileExistsError('Use a fresh output path')
checks = 0
started = time.monotonic()
def check(condition, label):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(label)
def sha(data):
    return hashlib.sha256(data).hexdigest()
def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

before = {str(p.relative_to(base)):sha(p.read_bytes()) for p in base.rglob('*') if p.is_file()}
check(len(before)==69, '69 archive files')
check(sum((base/p).stat().st_size for p in before)==895428, 'archive byte count')
check(not any(p.is_symlink() for p in base.rglob('*')), 'no symlinks')
check(sha((base/'artifact-hashes.json').read_bytes())=='39aaf40e5d31897f3f38b96a42388bb61fcc806e58497f8f5170377907cc931a', 'pinned artifact inventory')
manifest=read(base/'copy-manifest.json'); inventory=read(base/'artifact-hashes.json')
check(set(inventory['files'])==set(before)-{'artifact-hashes.json'}, 'inventory complete excluding self')
for name,h in inventory['files'].items():
    check(before[name]==h, 'artifact hash '+name)
entries={x['public_relative']:x for x in manifest['entries']}
check(len(entries)==67==len(manifest['entries']), 'unique copy entries')
check(set(entries)==set(before)-{'artifact-hashes.json','copy-manifest.json'}, 'copy inventory complete')

safe={n:read(review/(n+'-observations.json')) for n in ('runtime','directory')}
private={}
for n,path in [('runtime',args.private_runtime),('directory',args.private_directory)]:
    raw=Path(path).read_bytes()
    check(sha(raw)==safe[n]['private_retrieval_sha256'], 'private retrieval identity '+n)
    private[n]=json.loads(raw)
# Learn only the two redaction prefixes from an already observed command.
first=next(x for x in private['runtime']['items'] if x.get('type')=='commandExecution')['command']
project=re.search(r"p=Path\('([^']+)/scratch/0009-runtime-qualification'\)",first).group(1)
runtime=re.search(r"(/[^\s\"']+)/bin/python",first).group(1)
def sanitize(text):
    return text.replace(runtime,'<PYTHON_RUNTIME>').replace(project,'<PROJECT>')
def reverse_candidates(text, original_sha, original_bytes):
    # Original source recipes can themselves contain placeholder literals.
    # Use byte length plus digest to choose occurrences, rather than assuming
    # every placeholder was introduced by redaction. Never emit private bytes.
    simple=text.replace('<PROJECT>',project).replace('<PYTHON_RUNTIME>',runtime).encode()
    if len(simple)==original_bytes and sha(simple)==original_sha:
        return simple, 'global'
    tokens=list(re.finditer(r'<PROJECT>|<PYTHON_RUNTIME>',text))
    check(len(tokens)<=16, 'bounded selective reverse search')
    mapping={'<PROJECT>':project,'<PYTHON_RUNTIME>':runtime}
    matches=[]
    delta=original_bytes-len(text.encode())
    for mask in range(1 << len(tokens)):
        if sum(len(mapping[t.group()].encode())-len(t.group().encode())
               for i,t in enumerate(tokens) if mask & (1 << i)) != delta:
            continue
        parts=[];at=0
        for i,t in enumerate(tokens):
            parts.append(text[at:t.start()])
            parts.append(mapping[t.group()] if mask & (1 << i) else t.group())
            at=t.end()
        parts.append(text[at:]);candidate=''.join(parts).encode()
        if len(candidate)==original_bytes and sha(candidate)==original_sha:
            matches.append(candidate)
    check(len(matches)==1, 'unique digest-matching selective reversal')
    return matches[0], 'selective-literal-aware' 
for n in ('runtime','directory'):
    commands=[x for x in private[n]['items'] if x.get('type')=='commandExecution']
    check(len(commands)==len(safe[n]['commands']), 'all actual commands represented '+n)
    for actual,pub in zip(commands,safe[n]['commands']):
        raw_output=actual.get('output',{}).get('text','')
        check(sha(actual['command'].encode())==pub['original_command_sha256'], 'actual command hash '+n)
        check(sha(raw_output.encode())==pub['original_output_sha256'], 'actual output hash '+n)
        check(sanitize(actual['command'])==pub['command_redacted'], 'command redaction '+n)
        check(sanitize(raw_output)==pub['output_redacted'], 'output redaction '+n)
        check(actual['status']==pub['status'] and actual['exitCode']==pub['exit_code'], 'actual status '+n)
        check(actual['durationMs']==pub['tool_duration_ms'], 'actual duration '+n)
        check(actual.get('output',{}).get('truncated',False)==pub['output_truncated']==False, 'untruncated command output '+n)

modes=Counter();reverse_modes=Counter()
for name,e in entries.items():
    p=base/name;raw=p.read_bytes();mode=e['mode'];modes[mode]+=1
    check(not Path(name).is_absolute() and '..' not in Path(name).parts, 'safe archive relative path')
    check(not Path(e['source_relative']).is_absolute() and '..' not in Path(e['source_relative']).parts, 'safe source identity')
    check(sha(raw)==e['public_sha256'] and len(raw)==e['public_bytes'], 'public copy binding '+name)
    if mode=='byte_exact_copy':
        check(sha(raw)==e['original_sha256'] and len(raw)==e['original_bytes'], 'original exact binding '+name)
    elif mode=='redacted_copy':
        original,strategy=reverse_candidates(raw.decode(),e['original_sha256'],e['original_bytes']);reverse_modes[strategy]+=1
        check(sha(original)==e['original_sha256'] and len(original)==e['original_bytes'], 'reverse redaction binding '+name)
        check(bool(e['redaction_scope']), 'redaction declared '+name)
    else:
        check(mode=='derived_summary' and name in ('README.md','admission-status.json'), 'only explicit interpretation summaries')
        for dep in e['source_dependencies']:
            matches=[x for x in manifest['entries'] if x['source_relative']==dep['relative'] and x['mode']!='derived_summary']
            check(len(matches)==1 and matches[0]['original_sha256']==dep['sha256'], 'derived source dependency')
check(dict(modes)=={'redacted_copy':36,'byte_exact_copy':29,'derived_summary':2}, 'final copy modes')
original=read(base/'publication/original-preservation.json')
old_entries=[x for x in entries.values() if x['source_relative'].startswith(('scratch/0009-runtime-qualification/','scratch/0009-hpatches-directory/')) and x['mode']!='derived_summary']
check(len(old_entries)==45, '45 original text files represented')
old_binding={x['source_relative']:x['original_sha256'] for x in old_entries}
excluded=manifest['excluded_private_binaries']
check(len(excluded)==2 and sum(x['bytes'] for x in excluded)==163286, 'two excluded range metadata binaries')
old_binding.update({x['source_relative']:x['sha256'] for x in excluded})
check(original['count']==47 and original['files']==old_binding, '47-item original inventory relation')

# Recorded execution hashes refer to original pre-redaction bytes.
def original_hash(name):
    return entries[name]['original_sha256']
for label in ('base-numpy','base-scipy','base-cv2','runtime-fixed-numpy','runtime-fixed-scipy','runtime-fixed-cv2'):
    e=read(base/('runtime/'+label+'.execution.json'))
    for key,suffix in [('stdout_sha256','.stdout.log'),('stderr_sha256','.stderr.log'),('command_sha256','.command.recipe.json')]:
        check(e[key]==original_hash('runtime/'+label+suffix),'probe receipt original binding')
for group in ('runtime','hpatches'):
    e=read(base/(group+'/execution.json'))
    for key,name in [('stdout_sha256','stdout.log'),('stderr_sha256','stderr.log'),('command_sha256','command.recipe.json'),('capture_source_sha256','capture.py')]:
        check(e[key]==original_hash(group+'/'+name), 'outer receipt original binding '+group)
    check(e['actual_exit']==0 and e['raw_wait_status']==0, 'operation exits observed zero')
    obs=json.loads(safe['runtime' if group=='runtime' else 'directory']['commands'][2]['output_redacted'])
    check(e==obs,'actual capture tool output matches '+group)
check(read(base/'runtime/execution.json')['qualification_source_sha256']==original_hash('runtime/qualify.py'),'qualification source hash')
check(read(base/'runtime/execution.json')['probe_source_sha256']==original_hash('runtime/probe.py'),'probe source hash')
check(read(base/'hpatches/execution.json')['operation_source_sha256']==original_hash('hpatches/hpatches_directory.py'),'directory source hash')
summary=read(base/'runtime/summary.json')
observed_runtime=json.loads(safe['runtime']['commands'][3]['output_redacted'])
check(len(summary['probes'])==6,'six actual import results')
for p,o in zip(summary['probes'],observed_runtime['probes']):
    label=p['variant']+'-'+p['module']
    check(p==read(base/('runtime/'+label+'.execution.json')),'summary equals saved probe')
    projected={k:p[k] for k in ('variant','actual_exit','raw_wait_status','wall_seconds','peak_rss_kib')};projected.update(p['probe'])
    check(projected==o,'actual final probe observation matches')
    check((p['actual_exit']==1 and p['probe']['status']=='import_failed') if p['module']=='cv2' else (p['actual_exit']==0 and p['probe']['status']=='available'),'availability qualifications')
check(observed_runtime['summary_sha256']==original_hash('runtime/summary.json'),'observed summary hash')

obs_directory=json.loads(safe['directory']['commands'][3]['output_redacted'])
ledger=read(base/'hpatches/ledger.json');result=read(base/'hpatches/result.json')
check(obs_directory['result']==result,'observed directory result')
check(obs_directory['ledger']=={k:v for k,v in ledger.items() if k!='declared_whole_archive_sha256'},'observed ledger')
check(obs_directory['result_sha256']==original_hash('hpatches/result.json'),'observed result hash')
check(obs_directory['ledger_sha256']==original_hash('hpatches/ledger.json'),'observed ledger hash')
metadata=read(base/'hpatches/metadata-summary.json')
check(metadata==json.loads(safe['directory']['commands'][4]['output_redacted']),'actual metadata summary output')
last=json.loads(safe['directory']['commands'][5]['output_redacted'])
correction=read(base/'hpatches/redirect-accounting-review.json')
check(last['review_sha256']==original_hash('hpatches/redirect-accounting-review.json'),'observed accounting correction hash')
ops=ledger['range_get_operations']
check(len(ops)==2 and sum(len(x['https_redirect_statuses']) for x in ops)==1,'two operations and one redirect')
check(sum(x['actual_body_bytes_read'] for x in ops)==163286,'206-only metered sum')
for x in ops:
    check(x['actual_body_bytes_read']==x['expected_body_bytes']==x['range_end']-x['range_start']+1,'range byte arithmetic')
    check(x['http_status']==206 and x['headers']['Content-Range']=='bytes %d-%d/1279545572'%(x['range_start'],x['range_end']),'206 range headers')
    check(x['headers']['Content-Length']==str(x['expected_body_bytes']),'range length')
    check(any(x['body_sha256']==b['sha256'] and x['actual_body_bytes_read']==b['bytes'] for b in excluded),'range-to-excluded-binary hash')
check(ops[1]['range_end']+1==ops[0]['range_start'],'nonoverlapping contiguous ranges')
check(ops[0]['headers']['ETag']==ops[1]['headers']['ETag'],'same declared ETag')
check(correction['physical_http_get_requests']==3 and correction['unmetered_redirect_body_bytes'] is None and correction['all_response_body_total_bytes'] is None and correction['one_MiB_total_body_limit_verified'] is False,'failure correction preserved')
check(correction['relevant_stdlib_statements']==['fp.read()','fp.close()','return self.parent.open(new, timeout=req.timeout)'],'observed stdlib redirect statements')
code=(base/'hpatches/hpatches_directory.py').read_text()
check('class Redirects(HTTPRedirectHandler)' in code and 'return super().redirect_request(req,fp,code,msg,headers,newurl)' in code,'source delegates redirect')
check("if len(LEDGER['range_get_operations'])>=2" in code and "row['https_redirect_statuses'].append(code)" in code,'source counts logical operations separately')
check("LEDGER['actual_response_body_bytes_read']+=len(block)" in code,'source counts explicit final-body reads only')
status=read(base/'admission-status.json')
check(status['primary_status']=='STRICT_HTTP_ACCOUNTING_FAILED_DATA_NOT_ADMITTED' and status['data_admitted'] is False,'data not admitted')
check(status['physical_http_GET_requests']==3 and status['allowed_maximum_GET_requests']==2 and status['strict_GET_limit_passed'] is False,'physical GET violation')
for field in ('unmeasured_302_body_bytes','all_HTTP_response_body_total_bytes','complete_updated_conservative_ceiling_bytes'):
    check(status[field] is None,'unknown amount preserved '+field)
check(status['historical_ceiling_plus_measured_206_only_base_bytes']==status['historical_asset_ceiling_record_bytes']+163286,'historical arithmetic not replacement ceiling')

# Recount names/metadata only; original range bodies and member payloads are absent.
directory=read(base/'hpatches/directory.json'); members=directory['members']; seq=directory['sequences']
check(len(members)==1463 and len(seq)==116,'directory counts')
check(Counter(x['kind'] for x in members)==Counter(directory=117,file=1346),'file/directory metadata counts')
check(Counter(x['compression_method'] for x in members)==Counter({0:181,8:1282}),'compression metadata counts')
check(len({x['name'] for x in members})==len(members),'unique metadata names')
check(sum(x['compressed_bytes'] for x in members)==result['compressed_payload_sum_bytes']==1279236578,'compressed-size metadata sum')
extras=Counter();images=hs=0
for name,s in seq.items():
    images+=s['image_file_count'];hs+=s['homography_file_count'];extras.update(s['unexpected_filenames'])
    check(not s['missing_expected_images'] and not s['missing_expected_homographies'],'no missing expected names')
check(images==696 and hs==580,'expected filename counts')
check(extras==Counter({'attribs.txt':64,'readme.txt':4,'pt.txt':1,'5.png':1}),'70 unexpected names')
check(sum(bool(s['unexpected_filenames']) for s in seq.values())==68,'68 containers with extras')
check(directory['whole_sha256_verified'] is False and status['scene_independence_established'] is False,'whole hash and independence unverified')
check(all(status[k]==0 for k in ('dataset_payloads_opened','local_headers_read','scientific_calls')),'declared zero payload/scientific reads')
check(directory['eocd']['central_directory_bytes']+22==163286,'directory plus EOCD size')
check(sha((base/'hpatches/directory.json').read_bytes())=='fcc34ad30d333e4b985c90cec3a528c6e4adbe4b88212d238aaacbe715108170','full directory public byte identity')

# Publication repair is archive-only evidence; no parent tool-chain replay exists.
for label in ('archive','verify','verify-v2'):
    e=read(base/('publication/'+label+'.execution.json'))
    for key,suffix in [('stdout_sha256','.stdout.log'),('stderr_sha256','.stderr.log'),('command_file_sha256','.command.recipe.json')]:
        check(e[key]==original_hash('publication/'+label+suffix),'publication receipt original binding')
    check(e['outer_source_sha256']==original_hash('publication/parent.recipe.py'),'publication parent source binding')
check(read(base/'publication/verify.execution.json')['actual_exit']==1,'first publication verifier failed')
check('AssertionError:' in (base/'publication/verify.stderr.log').read_text() and 'publication/verify.recipe.py' in (base/'publication/verify.stderr.log').read_text(),'self-scanner failure retained')
check(read(base/'publication/verify-v2.execution.json')['actual_exit']==0,'second publication verifier exited zero')
verification=read(base/'publication/verification-v2.json')
check(verification['source_sha256']==original_hash('publication/verify_v2.recipe.py'),'v2 source identity')
check(verification['public_files_checked']==63 and verification['status']=='PASS_PUBLICATION_INTEGRITY_ONLY' and verification['data_admission_passed'] is False,'historical 63-file verifier scope')
repair=read(base/'publication/refresh-command-error.json')
check(repair['observed_whole_shell_exit']==0 and repair['individual_refresh_exit_observed'] is None,'aggregate exit not individual success')
check(repair['new_probe_or_data_request'] is False and repair['original_qualification_files_modified'] is False,'repair scope declared')

url_count=0
for name in before:
    path=base/name;raw=path.read_bytes();text=raw.decode()
    check(path.suffix in ('.json','.py','.md','.log') and b'\0' not in raw,'text/source only')
    check(project not in text and runtime not in text,'private prefixes absent')
    check(not re.search(r'/Users/|/home/|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text),'private account/token pattern absent')
    if path.suffix=='.json':read(path)
    for u in re.findall(r'https?://[^\s\x22\x27<>]+',text):
        parsed=urlsplit(u.rstrip('),.;'))
        check(not parsed.query and not parsed.fragment and not parsed.username and not parsed.password,'no signed/credential URL')
        url_count+=1
for n,o in safe.items():
    text=json.dumps(o)
    check(project not in text and runtime not in text,'safe parent observation prefixes '+n)
    check(not re.search(r'/Users/|/home/|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}',text),'safe observations token pattern '+n)
check(before=={str(p.relative_to(base)):sha(p.read_bytes()) for p in base.rglob('*') if p.is_file()},'archive unchanged by audit')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),text=True).strip()
report={'status':'PASS_WITH_LIMITATIONS_PUBLICATION_EVIDENCE_ONLY','data_admission_passed':False,'checks':checks,'archive_files':69,'archive_bytes':895428,'copy_modes':dict(modes),'redacted_files_reverse_hash_verified':36,'reverse_strategies':dict(reverse_modes),'byte_exact_bindings_verified':29,'derived_summaries':2,'original_text_files_represented':45,'excluded_private_range_binaries':2,'private_range_bodies_read_locally':False,'runtime_actual_parent_commands_verified':4,'directory_actual_parent_commands_verified':6,'publication_actual_parent_tool_chain_replayed':False,'publication_verifier_historical_file_count':63,'physical_GETs_recorded':3,'unknown_redirect_body_bytes':None,'whole_body_limit_verified':False,'data_or_models_acquired':False,'probes_or_scientific_calls_reexecuted':False,'archive_unchanged':True,'URL_occurrences_without_queries':url_count,'repo_head_at_check':head,'source_sha256':sha(Path(__file__).read_bytes()),'input_artifact_inventory_sha256':before['artifact-hashes.json'],'parent_observation_sha256':{n:sha((review/(n+'-observations.json')).read_bytes()) for n in safe},'private_retrieval_sha256':{n:safe[n]['private_retrieval_sha256'] for n in safe},'wall_seconds':time.monotonic()-started,'limits':['47 private originals staying unchanged is an archived cloud claim; local reverse hashes check published-byte consistency, not current private filesystem','Raw two ZIP range bodies and complete archive are not available/read locally; range-body hashes are manifest/observed claims','No complete network syscall trace; exact 302 body and total response bytes remain unknown','Publication execution repair reviewed from archived source/log/receipt only; no independent actual publication tool-chain replay','Historical v2 verifier covered 63 files; this parent audit covers all final 69','Availability imports do not establish geometry capability or dependency-closure integrity']}
with output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps(report,indent=2))
