"""Append publication records and derivative bindings; never touches originals."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0009-qualification-publication';PUB=Path(__file__).parent;DEST=REPO/'results/0009-admission/qualification';RUNTIME='<PYTHON_RUNTIME>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((DEST/'copy-manifest.json').read_text());index={e['public_relative']:e for e in manifest['entries']}
    for src in sorted(PUB.iterdir()):
        if not src.is_file() or src.name in ('publication.json','remote-verification.json'):continue
        if src.suffix not in ('.py','.json','.log'):continue
        text=src.read_text();public=text.replace(RUNTIME,'<PYTHON_RUNTIME>').replace(str(ROOT),'<PROJECT>');changed=public!=text
        name=src.name
        if src.suffix=='.py' and changed:name=src.stem+'.recipe.py'
        if src.name.endswith('.command.private.json'):name=src.name.replace('.command.private.json','.command.recipe.json')
        relative='publication/'+name;dst=DEST/relative;dst.parent.mkdir(exist_ok=True)
        if dst.exists():assert dst.read_text()==public,dst
        else:
            with dst.open('x') as f:f.write(public)
        if relative not in index:
            scope=[]
            if RUNTIME in text:scope.append('Existing interpreter/runtime absolute prefix replaced by <PYTHON_RUNTIME>.')
            if str(ROOT) in text:scope.append('Private project absolute prefix replaced by <PROJECT>.')
            e={'source_relative':str(src.relative_to(ROOT)),'original_sha256':sha(src),'original_bytes':src.stat().st_size,'public_relative':relative,'public_sha256':sha(dst),'public_bytes':dst.stat().st_size,'mode':'redacted_copy' if changed else 'byte_exact_copy','redaction_scope':scope,'role':'publication_source_recipe' if changed and src.suffix=='.py' else 'publication_record'}
            manifest['entries'].append(e)
    primary=ROOT/'scratch/0009-hpatches-directory/redirect-accounting-review.json';runtime=ROOT/'scratch/0009-runtime-qualification/summary.json';directory=ROOT/'scratch/0009-hpatches-directory/directory.json'
    for name in ('README.md','admission-status.json'):
        if name in index:continue
        dst=DEST/name;manifest['entries'].append({'source_relative':str(primary.relative_to(ROOT)),'original_sha256':sha(primary),'original_bytes':primary.stat().st_size,'public_relative':name,'public_sha256':sha(dst),'public_bytes':dst.stat().st_size,'mode':'derived_summary','redaction_scope':['Original saved records interpreted; not an original-byte copy.'],'role':'qualification_failure_interpretation','source_dependencies':[{'relative':str(s.relative_to(ROOT)),'sha256':sha(s)} for s in (primary,runtime,directory)]})
    (DEST/'copy-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # This inventory can be refreshed as new publication receipts are added.
    hashes={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='artifact-hashes.json'}
    (DEST/'artifact-hashes.json').write_text(json.dumps({'schema':'0009-qualification-artifact-hashes-v1','files':hashes,'self_exclusion':['artifact-hashes.json'],'cycle_rule':'copy-manifest.json excludes this inventory and itself; this inventory hashes copy-manifest.json.'},indent=2)+'\n')
    print(json.dumps({'manifest_entries':len(manifest['entries']),'result_files':len(hashes)+1,'artifact_hashes_sha256':sha(DEST/'artifact-hashes.json')}))
if __name__=='__main__':main()
