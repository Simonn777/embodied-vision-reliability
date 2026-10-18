"""Publication-only preservation/manifest/JSON/privacy review, no new probes."""
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0009-qualification-publication';PUB=Path(__file__).parent;DEST=REPO/'results/0009-admission/qualification';RUNTIME='<PYTHON_RUNTIME>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    original=read(PUB/'original-preservation.json');assert original['count']==47
    for relative,h in original['files'].items():assert sha(ROOT/relative)==h,relative
    now={str(p.relative_to(ROOT)) for n in ('0009-runtime-qualification','0009-hpatches-directory') for p in (ROOT/'scratch'/n).iterdir() if p.is_file()}
    assert now==set(original['files'])
    assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO),'tracked repository file modified'
    manifest=read(DEST/'copy-manifest.json');modes={};bound=set()
    for entry in manifest['entries']:
        src=ROOT/entry['source_relative'];dst=DEST/entry['public_relative'];bound.add(dst.resolve())
        assert not Path(entry['source_relative']).is_absolute() and '..' not in Path(entry['source_relative']).parts
        assert dst.resolve().is_relative_to(DEST.resolve())
        assert sha(src)==entry['original_sha256'] and src.stat().st_size==entry['original_bytes']
        assert sha(dst)==entry['public_sha256'] and dst.stat().st_size==entry['public_bytes']
        if entry['mode']=='byte_exact_copy':assert src.read_bytes()==dst.read_bytes() and not entry['redaction_scope']
        elif entry['mode']=='redacted_copy':assert dst.read_text()==src.read_text().replace(RUNTIME,'<PYTHON_RUNTIME>').replace(str(ROOT),'<PROJECT>') and entry['redaction_scope']
        else:
            assert entry['mode']=='derived_summary' and entry['redaction_scope']==['Original saved records interpreted; not an original-byte copy.']
            for s in entry['source_dependencies']:assert sha(ROOT/s['relative'])==s['sha256']
        modes[entry['mode']]=modes.get(entry['mode'],0)+1
    for e in manifest['excluded_private_binaries']:assert sha(ROOT/e['source_relative'])==e['sha256']
    assert len(manifest['excluded_private_binaries'])==2
    files=[];jsonfiles=0;urls=0
    for p in DEST.rglob('*'):
        if not p.is_file():continue
        assert p.suffix in ('.py','.json','.log','.md') and p.stat().st_size<=1024**2,p
        text=p.read_text();assert '\x00' not in text and '/workspace/' not in text and '/opt/codex/' not in text,p
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text),p
        assert not re.search(r'01a[0-9a-f]{5}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',text),p
        for u in re.findall(r'https?://[^\s\x22\x27<>]+',text):
            parsed=urlsplit(u.rstrip('),.;'));assert not parsed.query and not parsed.fragment and not parsed.username and not parsed.password,(p.name,'query/credential URL');urls+=1
        if p.suffix=='.json':json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)));jsonfiles+=1
        if p.name not in ('copy-manifest.json','artifact-hashes.json'):assert p.resolve() in bound,p
        files.append(p)
    links=0
    for target in re.findall(r'\]\(([^)]+)\)',(DEST/'README.md').read_text()):
        assert not target.startswith(('http:','https:')) and (DEST/target).exists(),target
        links+=1
    status=read(DEST/'admission-status.json');assert status['primary_status']=='STRICT_HTTP_ACCOUNTING_FAILED_DATA_NOT_ADMITTED' and not status['data_admitted'] and status['physical_http_GET_requests']==3 and not status['strict_GET_limit_passed'] and status['complete_updated_conservative_ceiling_bytes'] is None and status['unmeasured_302_body_bytes'] is None
    assert sha(DEST/'hpatches/directory.json')=='fcc34ad30d333e4b985c90cec3a528c6e4adbe4b88212d238aaacbe715108170'
    result={'status':'PASS_PUBLICATION_INTEGRITY_ONLY','data_admission_passed':False,'new_probes':0,'new_data_requests':0,'scientific_calls':0,'original_files_preserved':47,'original_binary_metadata_files_kept_private':2,'manifest_modes':modes,'public_files_checked':len(files),'json_files_checked':jsonfiles,'local_links_checked':links,'URL_occurrences_checked_no_queries':urls,'private_project_runtime_prefixes_present':False,'unexpected_binary_files':0,'directory_byte_exact_sha256':sha(DEST/'hpatches/directory.json'),'source_sha256':sha(__file__)}
    with (PUB/'verification.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
