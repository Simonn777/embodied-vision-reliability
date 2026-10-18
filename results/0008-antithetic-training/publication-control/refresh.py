"""Append publication-only evidence and refresh unpublished derivative hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0008-training';PUB=Path(__file__).parent;DEST=REPO/'results/0008-antithetic-training';PYTHON='<PYTHON>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((DEST/'copy-manifest.json').read_text());indexed={e['public_path']:e for e in manifest['entries']}
    for src in sorted(PUB.iterdir()):
        if not src.is_file() or src.name in ('publication.json','transfer.json'):continue
        if src.suffix not in ('.py','.json','.log'):continue
        name=src.name.replace('.command.private.json','.command.recipe.json');dst=DEST/'publication-control'/name
        text=src.read_text();public=text.replace(PYTHON,'<PYTHON>').replace(str(ROOT),'<PROJECT_ROOT>')
        if dst.exists():assert dst.read_text()==public,dst
        else:
            with dst.open('x') as f:f.write(public)
        key=str(dst.relative_to(REPO))
        if key in indexed:assert indexed[key]['original_sha256']==sha(src)
        else:
            scopes=[]
            if PYTHON in text:scopes.append('existing interpreter path replaced by <PYTHON>')
            if str(ROOT) in text:scopes.append('original project absolute prefix replaced by <PROJECT_ROOT>')
            entry={'original_source_relative':str(src.relative_to(ROOT)),'original_sha256':sha(src),'public_path':key,'public_sha256':sha(dst),'original_bytes':src.stat().st_size,'public_bytes':dst.stat().st_size,'mode':'redacted_copy' if scopes else 'byte_exact_copy','role':'publication_only_source_or_record','redaction_scope':scopes,'recipe_not_private_argv':src.name.endswith('.command.private.json')}
            manifest['entries'].append(entry)
    for e in manifest['entries']:
        if e['mode']=='derived_summary':
            dst=REPO/e['public_path'];e['public_sha256']=sha(dst);e['public_bytes']=dst.stat().st_size
    (DEST/'copy-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest_entries':len(manifest['entries']),'public_files':sum(p.is_file() for p in DEST.rglob('*'))}))
if __name__=='__main__':main()
