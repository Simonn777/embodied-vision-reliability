"""Copy reviewed original source and small preparation receipts; no science arrays."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK / 'embodied-vision-reliability'
PROJECT = WORK.parent
own = ROOT / 'experiments/0035-transformation-support'
evidence = ROOT / 'results/0035-preparation'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    if own.exists() or evidence.exists():
        raise FileExistsError('fresh publication directories required')
    own.mkdir(parents=True)
    evidence.mkdir(parents=True)
    entries=[]
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in ('.py', '.md', '.json'):
            # Operation specs, dispatched after freeze, do not exist yet.
            raw=source.read_bytes()
            target=own/source.name
            if str(PROJECT).encode() in raw:
                raise ValueError('absolute private workspace path in source: '+source.name)
            target.write_bytes(raw)
            entries.append(dict(kind='source',path=str(target.relative_to(ROOT)),bytes=len(raw),sha256=sha(raw)))
    for folder in sorted(HERE.glob('*-tests-*')):
        if not folder.is_dir():
            continue
        for source in sorted(folder.rglob('*')):
            if source.is_file():
                if source.suffix not in ('.py','.json','.log','.txt','.md'):
                    raise ValueError('unexpected preparation artifact '+str(source))
                raw=source.read_bytes()
                sanitized=raw.replace(str(PROJECT).encode(),b'${WORKSPACE}')
                target=evidence/source.relative_to(HERE)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(sanitized)
                entries.append(dict(kind='preparation-evidence',path=str(target.relative_to(ROOT)),
                    original_bytes=len(raw),original_sha256=sha(raw),bytes=len(sanitized),sha256=sha(sanitized),
                    workspace_prefix_redacted=(raw!=sanitized)))
    prior=WORK/'0034-relative-rgb/pr71-verification.json'
    target=evidence/'pr71-verification.json'
    shutil.copyfile(prior,target)
    raw=target.read_bytes()
    entries.append(dict(kind='prior-publication-verification',path=str(target.relative_to(ROOT)),bytes=len(raw),sha256=sha(raw)))
    record=dict(schema='evr0035.preparation-publication.v1',scientific_results=False,
        translation='Original source bytes; evidence text only literal workspace-prefix replacement',entries=entries)
    (evidence/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries))))

if __name__=='__main__':
    main()
