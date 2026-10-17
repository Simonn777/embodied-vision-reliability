"""Prospective 0003 timing consistency only; no data/model/experiment entry point."""
import hashlib
from pathlib import Path
import subprocess
from evr.motion_design import unique_json,verify_design,BRANCH

RECEIPT='protocols/0003-timing-addendum-freeze-v1.json'
V1_RECEIPT='protocols/0003-design-freeze-v1.json'
V1_SHA='adb1b58b2fdab0e63899c67e30706f14b710a113bd6892eecdc08657b7826a7a'
MANDATORY=frozenset([V1_RECEIPT,'protocols/0003-motion-design-v1.md','configs/0003-motion-design-v1.json','src/evr/motion_design.py','protocols/0003-timing-addendum-v1.md','configs/0003-timing-addendum-v1.json','src/evr/motion_phases.py','tests/test_motion_phases.py'])

def check_phase(stage,*,schema_identity_checked=False,development_geometry_quarantined=False,allocation_source_manifest_frozen=False,pipeline_binds_v1_and_addendum=False,all_prediction_hashes_frozen=False,frame0_query_GT_accessed=False,heldout_GT_beyond_frame0_accessed=False,GT_adjudication_after_prediction_freeze=False,all_required_family_GT_gates_pass=False):
    flags=locals().copy();flags.pop('stage')
    if stage not in ['pre_cohort','pre_prediction','post_prediction']:raise ValueError('known phase required')
    if any(type(v)is not bool for v in flags.values()):raise ValueError('typed boolean evidence only')
    if stage!='post_prediction'and heldout_GT_beyond_frame0_accessed:raise ValueError('heldout GT timing violation')
    if stage!='post_prediction'and(all_required_family_GT_gates_pass or GT_adjudication_after_prediction_freeze):raise ValueError('quality cannot be prevalidated on heldout GT')
    if stage=='pre_cohort':
        if all_prediction_hashes_frozen:raise ValueError('predictions require frozen cohort first')
        return {'status':'DESIGN_ONLY','GT_quality':'PENDING_POST_PREDICTION','frame0_GT_access':frame0_query_GT_accessed,'execution_authority':False}
    if not all([schema_identity_checked,development_geometry_quarantined,allocation_source_manifest_frozen,pipeline_binds_v1_and_addendum]):raise ValueError('structural cohort/pipeline evidence missing')
    if stage=='pre_prediction':
        if all_prediction_hashes_frozen:raise ValueError('wrong phase')
        return {'status':'STRUCTURAL_READINESS_ONLY','GT_quality':'PENDING_POST_PREDICTION','frame0_GT_access':frame0_query_GT_accessed,'execution_authority':False}
    if not all_prediction_hashes_frozen:raise ValueError('freeze every prescribed prediction before GT adjudication')
    if heldout_GT_beyond_frame0_accessed and not GT_adjudication_after_prediction_freeze:raise ValueError('GT chronology not established')
    quality_pass=GT_adjudication_after_prediction_freeze and all_required_family_GT_gates_pass
    return {'status':'GT_GATE_PASSED_NOT_EFFICACY'if quality_pass else'INCONCLUSIVE','GT_quality':'PASSED'if quality_pass else'UNDEFINED_OR_FAILED','all_allocated_families_retained':True,'primary_definition_unchanged':True,'execution_authority':False}

def verify_timing(repo):
    """Verify additive receipt bytes and published input commit, alongside v1."""
    repo=Path(repo).resolve();verify_design(repo)
    def git(*args):
        p=subprocess.run(['git','-C',str(repo),*args],capture_output=True)
        if p.returncode:raise ValueError('Git timing binding unavailable')
        return p.stdout
    def regular(path):
        p=repo/path
        if not p.is_file()or p.is_symlink()or not p.resolve().is_relative_to(repo)or any(q.is_symlink()for q in p.parents):raise ValueError('regular repository file required')
        return p.read_bytes()
    raw=regular(RECEIPT);r=unique_json(raw)
    if set(r)!={'schema','scope','input_commit','v1_receipt_path','v1_receipt_sha256','inputs_sha256'}or type(r['schema'])is not int or r['schema']!=1 or r['scope']!='PROSPECTIVE_TIMING_ONLY':raise ValueError('timing schema')
    if r['v1_receipt_path']!=V1_RECEIPT or r['v1_receipt_sha256']!=V1_SHA or hashlib.sha256(regular(V1_RECEIPT)).hexdigest()!=V1_SHA:raise ValueError('v1 canonical binding')
    if not isinstance(r['inputs_sha256'],dict)or set(r['inputs_sha256'])!=MANDATORY:raise ValueError('complete eight-file map required')
    commit=r['input_commit']
    if not isinstance(commit,str)or len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit)or git('cat-file','-t',commit).strip()!=b'commit':raise ValueError('full input commit')
    for ref in ['HEAD',':']:
        if git('show',(':'if ref==':'else ref+':')+RECEIPT)!=raw:raise ValueError('uncommitted timing receipt')
    refs=[]
    for ref in ['origin/main',BRANCH]:
        try:
            if git('show',ref+':'+RECEIPT)==raw:refs.append(ref)
        except ValueError:pass
    if not refs:raise ValueError('timing receipt not published')
    for ref in ['HEAD',*refs]:git('merge-base','--is-ancestor',commit,ref)
    for path,digest in r['inputs_sha256'].items():
        if not isinstance(digest,str)or len(digest)!=64 or any(c not in'0123456789abcdef'for c in digest):raise ValueError('SHA256 required')
        for data in [regular(path),git('show',commit+':'+path),git('show','HEAD:'+path),git('show',':'+path),*[git('show',ref+':'+path)for ref in refs]]:
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('timing blob/current/index/published mismatch')
    if RECEIPT in git('ls-tree','-r','--name-only',commit).decode().splitlines():raise ValueError('self-referencing timing inputs')
    return {'status':'TIMING_ADDENDUM_VERIFIED','input_commit':commit,'files':8,'v1_unchanged':True,'scope':'design interpretation only; no final cohort or inference authority'}
