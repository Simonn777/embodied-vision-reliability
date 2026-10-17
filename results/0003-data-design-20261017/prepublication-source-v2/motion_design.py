"""0003 design primitives only. No dataset/model/evaluation runner is provided."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

CONFIG='configs/0003-motion-design-v1.json'
PROTOCOL='protocols/0003-motion-design-v1.md'
RECEIPT='protocols/0003-design-freeze-v1.json'
BRANCH='origin/research/0003-motion-design-20261017'
MANDATORY=frozenset([CONFIG,PROTOCOL,'src/evr/motion_design.py','scripts/check_0003_design.py','tests/test_motion_design.py','docs/0003-data-audit-2026-10-17.md','results/0003-data-design-20261017/sources.json','results/0003-data-design-20261017/readiness.json','results/0003-data-design-20261017/acquisition-ledger.json'])

def finite_vector(v,n):
    if len(v)!=n or not all(type(x)in(int,float)and math.isfinite(x)for x in v):
        raise ValueError('finite numeric vector required')
    return tuple(float(x)for x in v)

def matrix(a,n,m):
    if len(a)!=n:raise ValueError('matrix shape')
    return tuple(finite_vector(row,m)for row in a)

def project(world,K,T_cw):
    """Explicit right-handed pinhole assumption; must be validated on dataset GT.

    T_cw maps world homogeneous column vectors to camera, positive z forward.
    Returns raw raster x,y without clipping; no assumption about upstream verified.
    """
    p=finite_vector(world,3)+(1.,);K=matrix(K,3,3);T=matrix(T_cw,4,4)
    if T[3]!=(0.,0.,0.,1.):raise ValueError('non-affine extrinsic')
    R=[row[:3]for row in T[:3]]
    if any(abs(sum(R[i][k]*R[j][k]for k in range(3))-(i==j))>1e-8 for i in range(3)for j in range(3)):
        raise ValueError('extrinsic rotation not orthonormal')
    det=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0])
    if abs(det-1)>1e-8 or K[2]!=(0.,0.,1.)or K[0][0]<=0 or K[1][1]<=0:raise ValueError('camera convention gate')
    camera=tuple(sum(T[i][j]*p[j]for j in range(4))for i in range(3))
    if camera[2]<=0:raise ValueError('point behind camera')
    h=tuple(sum(K[i][j]*camera[j]for j in range(3))for i in range(3))
    return(h[0]/h[2],h[1]/h[2])

def resize_xy(xy,width,height):
    xy=finite_vector(xy,2)
    if type(width)is not int or type(height)is not int or min(width,height)<=0:raise ValueError('image size')
    return(xy[0]*256/width,xy[1]*256/height)

def innovation(times,positions):
    """Signed current minus two-position causal extrapolation, real time gaps."""
    t=finite_vector(times,3)
    if not t[0]<t[1]<t[2]:raise ValueError('strict timestamps required')
    n=len(positions[0]);p=[finite_vector(v,n)for v in positions];q=(t[2]-t[1])/(t[1]-t[0])
    return tuple(p[2][j]-p[1][j]-(p[1][j]-p[0][j])*q for j in range(n))

def decompose(times,predicted,truth):
    """GT is diagnostic only: exact SAME timestamps/history support in all terms."""
    if len(predicted)!=3 or len(truth)!=3:raise ValueError('three aligned positions required')
    try:
        p=[finite_vector(v,2)for v in predicted];g=[finite_vector(v,2)for v in truth]
    except ValueError:
        return {'defined':False,'reason':'nonfinite anchor; retain primary denominator and original prediction-history anchors'}
    e=[tuple(x-y for x,y in zip(a,b))for a,b in zip(p,g)]
    rp=innovation(times,p);rg=innovation(times,g);re=innovation(times,e)
    difference=max(abs(x-y-z)for x,y,z in zip(rp,rg,re))
    if difference>1e-9:raise ValueError('signed decomposition tolerance exceeded')
    return {'defined':True,'r_pred':rp,'r_gt_image':rg,'r_error':re,'max_identity_absdiff':difference,'d_pred':math.hypot(*rp),'d_gt_image':math.hypot(*rg),'d_error':math.hypot(*re)}

def world_acceleration(times,world):
    """World-coordinate units / timestamp-unit squared, never asserts m/s^2."""
    t=finite_vector(times,3);innovation(t,world);p=[finite_vector(v,3)for v in world]
    return tuple(2*((p[2][i]-p[1][i])/(t[2]-t[1])-(p[1][i]-p[0][i])/(t[1]-t[0]))/(t[2]-t[0])for i in range(3))

def reference_penalties(times,xy):
    """Raw-coordinate history only: deliberately has no GT/logit/acceptance input."""
    if len(times)!=len(xy):raise ValueError('dimensions')
    history=[];out=[];previous=None
    for t,p in zip(times,xy):
        if type(t)not in(int,float)or not math.isfinite(t)or previous is not None and t<=previous:raise ValueError('timestamps')
        previous=t
        if len(p)!=2:raise ValueError('xy dimensions')
        if not all(type(v)in(int,float)and math.isfinite(v)for v in p):out.append(None);continue
        d=0 if len(history)<2 else math.hypot(*innovation([history[-2][0],history[-1][0],t],[history[-2][1],history[-1][1],p]))
        out.append(min(d/8,10));history.append((t,p))
    return out

def safe_member(name,kind='file'):
    if not isinstance(name,str):raise ValueError('path must be text')
    p=PurePosixPath(name)
    if not name or p.is_absolute()or any(x in('..','.')for x in name.split('/'))or '\\'in name or ':'in name or kind not in('file','directory'):
        raise ValueError('unsafe archive member or link/device')
    return name

def allocate(groups,seed=20261020):
    """Whole audited family IDs, not video hashes. No inference-outcome argument."""
    if any(not isinstance(g.get('family_id'),str)or not g['family_id']for g in groups):raise ValueError('nonempty canonical family IDs required')
    if len({g['family_id']for g in groups})!=len(groups):raise ValueError('family groups must be unique')
    result={g['family_id']:'development'for g in groups if g.get('audited')is not False or g.get('independence_verified')is not True}
    remaining=[g['family_id']for g in groups if g['family_id']not in result]
    remaining.sort(key=lambda x:(hashlib.sha256(f'0003|{seed}|{x}'.encode()).hexdigest(),x))
    # Exactly 2 additional untouched development, 10 calibration, 20 evaluation.
    for i,g in enumerate(remaining):result[g]='development'if i<2 else'calibration'if i<12 else'evaluation'if i<32 else'reserve'
    return result

def readiness(evidence):
    reasons=[]
    for name in ['whole_archive_verified','geometry_verified','family_provenance_verified','cohort_frozen','development_precision_plan_passed']:
        if evidence.get(name)is not True:reasons.append(name)
    for name,minimum in [('independent_evaluation_groups',20),('calibration_groups',10),('valid_gt_fraction',.95)]:
        x=evidence.get(name)
        if type(x)not in(int,float)or not math.isfinite(x)or x<minimum or name!='valid_gt_fraction'and type(x)is not int or name=='valid_gt_fraction'and x>1:reasons.append(name)
    return {'ready':not reasons,'status':'READY_FOR_LATER_AUTHORIZED_PREDICTION'if not reasons else'BLOCKED_NO_FINAL_COHORT','missing_evidence':reasons}

def improvement_gate(native_risk,augmented_risk,ci_upper,*,test_groups,error_groups,native_failures,defined_replicates,replicates=2000,reachable=True):
    for x in [test_groups,error_groups,native_failures,defined_replicates,replicates]:
        if type(x)is not int or x<0:raise ValueError('nonnegative integer counts required')
    sufficient=reachable is True and test_groups>=20 and error_groups>=5 and native_failures>=20 and replicates==2000 and defined_replicates>=1900
    finite=all(type(v)in(int,float)and math.isfinite(v)for v in[native_risk,augmented_risk,ci_upper])
    if not sufficient or not finite or not 0<native_risk<=1 or not 0<=augmented_risk<=1 or defined_replicates>replicates:return {'status':'INCONCLUSIVE','C':None,'continuation_signal':False}
    C=augmented_risk-.8*native_risk
    return {'status':'SUFFICIENT_SUPPORT','C':C,'continuation_signal':C<=0 and ci_upper<0}

def unique_json(data):
    def pairs(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(data,object_pairs_hook=pairs)

def verify_design(repo,receipt_path=RECEIPT):
    """Canonical committed/published DESIGN freeze, never authorizes inference.

    Caller must freshly fetch the public branch/main. Local refs are evidence,
    not cryptographic attestation of a remote server. Receipt excludes itself.
    """
    repo=Path(repo).resolve()
    def git(*args):
        p=subprocess.run(['git','-C',str(repo),*args],capture_output=True)
        if p.returncode:raise ValueError('Git binding unavailable')
        return p.stdout
    def regular(path):
        safe_member(path)
        p=repo/path
        if any(x.is_symlink()for x in [p,*p.parents]if x!=repo.parent):raise ValueError('symlink path')
        if not p.is_file()or not p.resolve().is_relative_to(repo):raise ValueError('regular repository file required')
        return p.read_bytes()
    if receipt_path!=RECEIPT:raise ValueError('canonical receipt only')
    raw=regular(RECEIPT);r=unique_json(raw)
    if set(r)!={'schema','freeze_kind','readiness','input_commit','config_path','protocol_path','inputs_sha256'}:raise ValueError('receipt schema')
    if type(r['schema'])is not int or r['schema']!=1 or r['freeze_kind']!='DESIGN_ONLY'or r['readiness']!='BLOCKED_NO_FINAL_COHORT'or r['config_path']!=CONFIG or r['protocol_path']!=PROTOCOL:raise ValueError('receipt binding')
    commit=r['input_commit']
    if not isinstance(commit,str)or not re.fullmatch('[0-9a-f]{40}',commit)or git('cat-file','-t',commit).strip()!=b'commit':raise ValueError('full input commit required')
    if not isinstance(r['inputs_sha256'],dict)or set(r['inputs_sha256'])!=MANDATORY:raise ValueError('complete mandatory map required')
    if git('show','HEAD:'+RECEIPT)!=raw or git('show',':'+RECEIPT)!=raw:raise ValueError('uncommitted/staged/changed receipt')
    refs=[]
    for ref in ['origin/main',BRANCH]:
        try:
            if git('show',ref+':'+RECEIPT)==raw:refs.append(ref)
        except ValueError:pass
    if not refs:raise ValueError('receipt not published in designated ref')
    for ref in ['HEAD',*refs]:git('merge-base','--is-ancestor',commit,ref)
    for path,digest in r['inputs_sha256'].items():
        if not isinstance(digest,str)or not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('SHA256 format')
        for data in [regular(path),git('show',commit+':'+path),git('show','HEAD:'+path),git('show',':'+path),*[git('show',ref+':'+path)for ref in refs]]:
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('input blob/current/index/publication mismatch: '+path)
    if RECEIPT in git('ls-tree','-r','--name-only',commit).decode().splitlines():raise ValueError('receipt self-reference/input commit includes receipt')
    return {'status':'DESIGN_FREEZE_VERIFIED','readiness':'BLOCKED_NO_FINAL_COHORT','input_commit':commit,'files':len(MANDATORY),'published_refs':refs,'scope':'design binding only; no final cohort or experiment authorization'}
