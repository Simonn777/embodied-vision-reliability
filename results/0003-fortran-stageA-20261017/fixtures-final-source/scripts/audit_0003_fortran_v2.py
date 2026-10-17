"""Future 0003 cached audit. Stage A must NOT invoke main or create its receipt."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile
from evr.motion_design import safe_member, unique_json
from evr.point_geometry_v2 import audit_budget, load_geometry, check_geometry, metadata_json, info_npz_headers, zip_envelope

RECEIPT = 'protocols/0003-fortran-audit-freeze-v2.json'
BRANCH = 'refs/remotes/origin/research/0003-fortran-design-20261017'
INPUTS = frozenset([
    'protocols/0003-fortran-audit-v2.md','configs/0003-fortran-audit-v2.json',
    'src/evr/point_geometry_v2.py','scripts/audit_0003_fortran_v2.py',
    'tests/test_point_geometry_v2.py','results/0003-fortran-stageA-20261017/source-convention.json',
    'protocols/0003-design-freeze-v1.json','protocols/0003-timing-addendum-freeze-v1.json',
    'protocols/0003-resource-addendum-freeze-v1.json'])
PRIOR = {
 'protocols/0003-design-freeze-v1.json':'adb1b58b2fdab0e63899c67e30706f14b710a113bd6892eecdc08657b7826a7a',
 'protocols/0003-timing-addendum-freeze-v1.json':'ef05a121bf5ef0f525367ed7bcf3b4a8cd20df66c2be6bb09194fb44a5f6e2ab',
 'protocols/0003-resource-addendum-freeze-v1.json':'0e1b867fad0a3873ffb8cb6ec64335ce5c2c0dbf3ef779efd8cbe5c553452e04'}


def git(*args):
    p = subprocess.run(['git',*args],capture_output=True)
    if p.returncode: raise ValueError('Git binding unavailable')
    return p.stdout


def digest(path):
    h = hashlib.sha256()
    p = Path(path)
    if p.is_symlink() or any(q.is_symlink() for q in p.parents) or not p.is_file():
        raise ValueError('regular cache/input required')
    with p.open('rb') as f:
        while part := f.read(2**20): h.update(part)
    return h.hexdigest()


def verify_published():
    if Path(RECEIPT).is_symlink() or any(p.is_symlink() for p in Path(RECEIPT).parents): raise ValueError('receipt symlink')
    raw = Path(RECEIPT).read_bytes()  # Missing receipt fails BEFORE cache access.
    r = unique_json(raw)
    if set(r) != {'schema','scope','input_commit','inputs_sha256'} or r['schema'] != 2 or r['scope'] != 'DEVELOPMENT_GEOMETRY_AUDIT_ONLY' or type(r['inputs_sha256']) is not dict or set(r['inputs_sha256']) != INPUTS:
        raise ValueError('canonical complete v2 receipt required')
    commit = r['input_commit']
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('input commit')
    if RECEIPT in git('ls-tree','-r','--name-only',commit).decode().splitlines(): raise ValueError('self reference')
    refs = ['HEAD',BRANCH]
    if Path(RECEIPT).is_symlink() or any(p.is_symlink() for p in Path(RECEIPT).parents): raise ValueError('receipt symlink')
    for ref in refs:
        if git('show',ref+':'+RECEIPT) != raw: raise ValueError('receipt not published/current')
        git('merge-base','--is-ancestor',commit,ref)
    if git('show',':'+RECEIPT) != raw: raise ValueError('unstaged receipt')
    for p,d in r['inputs_sha256'].items():
        if not isinstance(d,str) or not re.fullmatch('[0-9a-f]{64}',d) or digest(p) != d: raise ValueError('input digest')
        for spec in [commit+':'+p,'HEAD:'+p,':'+p,BRANCH+':'+p]:
            if hashlib.sha256(git('show',spec)).hexdigest() != d: raise ValueError('input Git/current binding')
    for p,d in PRIOR.items():
        if digest(p) != d: raise ValueError('unchanged prior receipt binding')
        old = unique_json(Path(p).read_bytes())
        mapping = old.get('inputs_sha256',old.get('input_sha256'))
        if not mapping: raise ValueError('prior mandatory map')
        for name,want in mapping.items():
            if digest(name) != want or hashlib.sha256(git('show',old['input_commit']+':'+name)).hexdigest()!=want:
                raise ValueError('prior frozen input changed')
    return r


def metadata(cache, config, budget, result):
    targets = config['metadata_members']; seen=set(); count=total=0; names=set(); evidence={}
    result['metadata_access']=evidence
    with tarfile.open(cache/'sample-v1.2.tar.gz','r|gz') as tar:
        for member in tar:
            kind='file' if member.isfile() else 'directory' if member.isdir() else 'other'
            name=safe_member(member.name,kind)
            if name in names: raise ValueError('duplicate TAR path')
            names.add(name);count+=1;total+=member.size
            if count>200000 or total>20*1024**3: raise ValueError('TAR limits')
            if name not in targets: continue
            if kind!='file' or member.size!=targets[name] or member.size>16384: raise ValueError('fixed metadata member')
            if sum(targets.values())>65536: raise ValueError('metadata total cap')
            stream=tar.extractfile(member)
            with stream: raw=budget.read(stream,member.size)
            entry={'source_member_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            evidence[name]=entry;seen.add(name)
            if name.endswith('.json'): entry['summary']=metadata_json(raw)
            else: entry['headers']=info_npz_headers(raw,budget)
    result['metadata_access']=evidence
    if seen != set(targets): raise ValueError('metadata member missing')
    result['family_graph']={
      'nodes':['sample/r4_new_f','sample/scene_d78_0318_3rd'],
      'edge':{'type':'UNKNOWN_LINKAGE_CONSERVATIVE_JOIN','session':'unknown','story':'unknown',
              'assets':'unknown','motion':'unknown','multiview':'unknown'},
      'quarantine':'entire sample and unknown linked family DEVELOPMENT',
      'independent_calibration_or_evaluation_families_established':0,
      'interpretation':'different metadata/name/hash cannot establish independence; field presence is not verified linkage'}


def main():
    args=argparse.ArgumentParser();args.add_argument('--cache',type=Path,required=True);args.add_argument('--output',type=Path,required=True);a=args.parse_args()
    r=verify_published();config=json.loads(Path('configs/0003-fortran-audit-v2.json').read_text())
    # Persistent one-shot marker is separate from output and shared across worktrees.
    locks=[]
    for name in ['0003-design.lock','0003-sample.lock','0003-fortran-v2.lock']:
        f=(a.cache.parent/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    if a.output.exists(): raise ValueError('exclusive new output required')
    a.output.mkdir(parents=True,exist_ok=False)
    marker=a.cache/'fortran-v2-started.json'
    with marker.open('x') as f: json.dump({'input_commit':r['input_commit'],'no_automatic_retry':True},f)
    budget=audit_budget()
    result={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running',
            'input_commit':r['input_commit'],'execution_commit':git('rev-parse','HEAD').decode().strip(),
            'receipt_sha256':digest(RECEIPT),'carried_budget_bytes':590976,'point_checks':[],
            'numeric_GT_access':'not started','RGB':0,'model':0,'new_GET':0,'world_units':'unknown','FPS':'unknown'}
    def save():
        result.update(charged_bytes=budget.charged_bytes,returned_bytes_including_carried=budget.returned_bytes,
                      reserves_including_carried=budget.stream_reserve_bytes,
                      ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (a.output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        archive=a.cache/'sample-v1.2.tar.gz';npz=a.cache/'resource-audit-v1/selected-annotation.npz'
        for p,n,h in [(archive,config['archive_bytes'],config['archive_sha256']),(npz,config['annotation_bytes'],config['annotation_sha256'])]:
            if p.stat().st_size!=n or digest(p)!=h: raise ValueError('cached identity mismatch; no download fallback')
        for name,want in [('official-reprojection.body','c226ff24be5037fbce6ed747bf31feb1fea1de17b7a8260317fd84737b5f1c00'),('official-tracking.body','413f680d2dd8b7bf65179bdf305baf94207441dd186c4fbba0d7d9bf66787f2a')]:
            if digest(a.cache/name)!=want: raise ValueError('cached pinned source text mismatch')
        result['cached_pinned_source_hashes_verified']=True
        metadata(a.cache,config,budget,result);save()
        result['numeric_GT_access']='attempting all-points first-three-frame prefixes; includes discarded gaps'
        result['zip_envelope']=zip_envelope(npz,budget)
        ledger={};result['numeric_access']=ledger
        with zipfile.ZipFile(npz) as z: hs,data,ledger=load_geometry(z,budget,config['expected_T_N'],ledger)
        if hs['trajs_2d']['shape'][:2]!=config['expected_T_N']: raise ValueError('observed schema changed')
        result.update(schema=hs,numeric_access=ledger,numeric_GT_access='all required prefixes read; discarded-gap bytes counted')
        save();check_geometry(hs,data,result)
    except Exception as exc:
        result.update(status='stopped',error_type=type(exc).__name__,reason=str(exc).replace(str(Path.cwd()),'<repo>'))
    finally: save()
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['status']=='completed_small_development_check' else 1)


if __name__=='__main__': main()
