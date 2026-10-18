"""Saved qualification text publication only. No requests/import probes/science."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0009-qualification-publication';PUB=Path(__file__).parent;DEST=REPO/'results/0009-admission/qualification'
RUNTIME='<PYTHON_RUNTIME>'
SOURCES=('0009-runtime-qualification','0009-hpatches-directory')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    assert not DEST.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='a915553fd63573df28661868a7c573e7fe2d3b76'
    original={str(p.relative_to(ROOT)):sha(p) for n in SOURCES for p in (ROOT/'scratch'/n).iterdir() if p.is_file()}
    write(PUB/'original-preservation.json',{'files':original,'count':len(original),'scope':'Original qualification directories, including two private central-directory metadata binaries. Nothing is overwritten.'})
    DEST.mkdir(parents=True);entries=[];excluded=[]
    for name in SOURCES:
        category='runtime' if name=='0009-runtime-qualification' else 'hpatches'
        for p in sorted((ROOT/'scratch'/name).iterdir()):
            if not p.is_file():continue
            if p.suffix=='.bin':
                excluded.append({'source_relative':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'reason':'Raw ZIP metadata bytes kept private; no binary payload copied.'});continue
            assert p.suffix in ('.py','.json','.log'),p
            raw=p.read_bytes();text=raw.decode();public=text.replace(RUNTIME,'<PYTHON_RUNTIME>').replace(str(ROOT),'<PROJECT>')
            changed=public!=text;newname=p.name
            if p.suffix=='.py' and changed:newname=p.stem+'.recipe.py'
            if p.name.endswith('command.json'):newname=p.name.removesuffix('command.json')+'command.recipe.json'
            target=DEST/category/newname;target.parent.mkdir(exist_ok=True)
            with target.open('xb') as f:f.write(public.encode())
            scopes=[]
            if RUNTIME in text:scopes.append('Existing interpreter/runtime absolute prefix replaced by <PYTHON_RUNTIME>.')
            if str(ROOT) in text:scopes.append('Private project absolute prefix replaced by <PROJECT>.')
            entries.append({'source_relative':str(p.relative_to(ROOT)),'original_sha256':sha(p),'original_bytes':len(raw),'public_relative':str(target.relative_to(DEST)),'public_sha256':sha(target),'public_bytes':target.stat().st_size,'mode':'redacted_copy' if changed else 'byte_exact_copy','redaction_scope':scopes,'role':'source_recipe' if p.suffix=='.py' and changed else 'command_recipe' if p.name.endswith('command.json') else 'saved_evidence'})
    d=read(ROOT/'scratch/0009-hpatches-directory/directory.json');review=read(ROOT/'scratch/0009-hpatches-directory/redirect-accounting-review.json')
    assert review['status']=='INCOMPLETE_STRICT_REQUEST_AND_BODY_ACCOUNTING' and review['physical_http_get_requests']==3
    assert sha(DEST/'hpatches/directory.json')=='fcc34ad30d333e4b985c90cec3a528c6e4adbe4b88212d238aaacbe715108170'
    status={'schema':'0009-qualification-public-interpretation-v1','primary_status':'STRICT_HTTP_ACCOUNTING_FAILED_DATA_NOT_ADMITTED','data_admitted':False,'geometric_performance_tested':False,'physical_http_GET_requests':3,'allowed_maximum_GET_requests':2,'strict_GET_limit_passed':False,'range_operations':2,'HTTPS_302_redirects':1,'verified_206_response_body_increment_bytes':163286,'unmeasured_302_body_bytes':None,'all_HTTP_response_body_total_bytes':None,'all_response_body_one_MiB_limit_verified':False,'historical_asset_ceiling_record_bytes':7532690072,'historical_ceiling_plus_measured_206_only_base_bytes':7532853358,'complete_new_asset_total_verified':False,'complete_updated_conservative_ceiling_bytes':None,'original_ledger_interpretation':'Original ledger/result/metadata-summary fields that label 7532853358 as a ceiling are historical arithmetic claims, not a verified total-response conservative ceiling. They are preserved unchanged and superseded only in interpretation by this record and redirect-accounting-review.json.','directory_entries':len(d['members']),'named_sequence_containers':len(d['sequences']),'scene_independence_established':False,'dataset_payloads_opened':0,'local_headers_read':0,'whole_archive_hash_verified':False,'runtime_availability':'NumPy and SciPy import; scipy.spatial.cKDTree exists. cv2 import fails in both variants; no geometry-performance inference.','new_publication_network_data_requests':0,'scientific_calls':0,'additional_fees':False}
    write(DEST/'admission-status.json',status)
    readme='''# Qualification records: strict HTTP accounting failed; data not admitted

**Primary status: NOT ADMITTED.** Physical HTTP GET count was **3**, exceeding
the authorized maximum of 2. The 302 response body was not metered, so neither
the complete new response-body total nor the strict 1 MiB total-body bound was
verified. Parsing a directory and process exit 0 do not make this qualification
pass. These are AI-assisted runtime/directory records, not a scientific experiment.

## Saved runtime qualification

| Selected path variant | NumPy | SciPy | `scipy.spatial.cKDTree` | OpenCV |
|---|---|---|---|---|
| Existing base interpreter | 2.3.5 | 1.17.0 | exists | `ModuleNotFoundError`, exit 1 |
| Existing runtime-fixed prefix | 2.0.2 | 1.17.0 from base | exists | `ModuleNotFoundError`, exit 1 |

`cv2.USAC_MAGSAC`, `cv2.findHomography` and `cv2.SIFT_create` could not be assessed
because cv2 was absent. No package was installed. No image/matrix function, model,
dataset or geometric benchmark was called. Module entry-file paths and hashes
are retained in [the runtime summary](runtime/summary.json); these are not hashes
of the whole dependency closure. This availability is not geometric performance.

Python 3.12.14 / Linux 6.18.55 / AMD EPYC 9V74, four CPU affinity slots and numeric
threads, 8 GiB AS. Runtime-qualification outer exit/raw wait 0/0, wall 1.211469 s,
peak RSS 53,524 KiB; all six fresh import outcomes, including both cv2 failures,
remain separately recorded. The two qualification directories were not rerun.

## Saved HPatches directory evidence and failed accounting

The pinned author archive declares 1,279,545,572 bytes and whole-file SHA256
`99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670`.
**The whole file and its hash were not verified.** Only ZIP directory metadata
was requested. Normal TLS and the existing runtime route were used.

| Range operation | Requested inclusive byte range | Status | Metered 206 body |
|---|---|---:|---:|
| Tail | 1279480015–1279545571 | 206 | 65,557 B |
| Missing directory prefix | 1279382286–1279480014 | 206 | 97,729 B |

The two ranges are nonoverlapping and were reused to assemble 163,264 central
directory bytes plus the 22-byte EOCD. Both final responses matched the requested
Content-Range and Content-Length before their bodies were explicitly read.
The first operation followed one HTTPS 302; there were therefore **three physical
GET requests**, not two. Default urllib also read an unmeasured redirect body.
No retry or further request was made after discovery. Only final host
`us.aws.cdn.hf.co` is recorded; no signed URL or query token is published.

The old scientific-asset ceiling record **7,532,690,072 B** remains unchanged.
The verified 206-body increment is **163,286 B**. Their sum **7,532,853,358 B** is
only the old ceiling plus the measured increment, **not** a verified updated
total-response conservative ceiling. The 302 transport-body gap remains UNKNOWN;
complete new acquisition and all-body bounds are unverified. Original
[ledger](hpatches/ledger.json), [result](hpatches/result.json), metadata summary
and their historical labels are preserved, with the correction in
[redirect accounting review](hpatches/redirect-accounting-review.json) and
[current interpretation](admission-status.json). No old receipt was edited to pass.

[The complete directory](hpatches/directory.json) has **1,463 members**: 117 directory
entries and 1,346 regular files, grouped into 116 named sequence containers. It
lists 696 expected image names and 580 expected H names, with no missing expected
names, plus 70 unexpected files across 68 containers: `attribs.txt` ×64,
`readme.txt` ×4, `pt.txt` ×1 and `5.png` ×1. None was opened or repaired. Names,
sizes, offsets, CRC fields, flags and compression methods are metadata only.
Different names do not prove independent scenes. Local headers, CRC/payload
integrity, images and H values were never read or decoded. Offsets can inform a
future member-retrieval design; they do not establish data admission or valid geometry.
The parent supplied official-source/license context; this operation did not reread
license pages or independently establish licensing beyond that supplied context.

Directory-operation outer exit/raw wait was **0/0**, wall **11.560743 s**, peak RSS
**19,008 KiB**, four CPU slots/threads, 8 GiB AS and a 120-second external limit.
The original stderr has one regex `SyntaxWarning`, not an execution exception;
it is separate from the real HTTP count/body-accounting failure. Code and warning
bytes were not patched, and the operation was not retried.

## Preservation, privacy and publication scope

All 47 original files across the two directories remain byte-exact, including
two private `.bin` metadata ranges which are **not published**. All 45 original
text/source/command/log files are represented here. The complete directory JSON
is an exact copy, SHA256
`fcc34ad30d333e4b985c90cec3a528c6e4adbe4b88212d238aaacbe715108170`.
Images, H payloads and any other dataset binary are absent.

[Copy manifest](copy-manifest.json) gives original relative source identity,
SHA256/size, public SHA256/size, exact-copy versus redaction and transformation
scope. `<PROJECT>` and `<PYTHON_RUNTIME>` replace private absolute prefixes.
Changed source files are labeled recipes; a recipe is not the original bytes or
an instruction to repeat the operation. Original argv and source digests remain
bound in receipts. No dialogue metadata or signed URLs are included.
[Artifact hashes](artifact-hashes.json) excludes itself; the copy manifest excludes
both inventories to avoid a hash cycle. Publication-only review evidence is kept
under publication/. Any review failure and correction is preserved there.

This publication makes no data request, package import probe, science call,
installation, schedule change or added-fee operation. It modifies no repository
integration document, old freeze or result. Parent review and a separate justified
freeze would be required for any future experiment; none is authorized by these
records. No method-quality, novelty or paper claim follows.
'''
    with (DEST/'README.md').open('x') as f:f.write(readme)
    write(DEST/'copy-manifest.json',{'schema':'0009-qualification-copy-v1','entries':entries,'excluded_private_binaries':excluded,'generated_summaries':{'README.md':'Interpretation of saved runtime/directory/accounting records; no new measurements.','admission-status.json':'Strict failure interpretation, preserving old receipts unchanged.'},'self_exclusion':['copy-manifest.json','artifact-hashes.json'],'private_prefix_placeholders':['<PROJECT>','<PYTHON_RUNTIME>']})
    print(json.dumps({'status':'saved_text_archive_created','copied_original_text_files':len(entries),'original_files_preserved':len(original),'excluded_binaries':len(excluded),'directory_exact_copy_sha256':sha(DEST/'hpatches/directory.json'),'public_file_count':sum(p.is_file() for p in DEST.rglob('*'))}))
if __name__=='__main__':main()
