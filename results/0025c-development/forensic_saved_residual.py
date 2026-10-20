"""Post-outcome saved-only arithmetic forensics, not a second audit or solver run.

Reads exactly one saved case, saved audit and CSV/aggregate metadata. It performs
only matrix-vector products, exact Fraction sums and descriptive gate counting;
no decomposition, fit, fixture generation, original source import or data/model.
"""
import csv
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
NOISE=HERE/'noise-01'
OUT=HERE/'residual-forensics-01'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return x if isinstance(x,F) else F.from_float(float(x))
def exact_mv(A,x):return [sum((q(a)*q(v) for a,v in zip(row,x)),F(0)) for row in A]
def vals(x):return [float(v) for v in x]
def errors(value,exact):return vals([abs(q(a)-b) for a,b in zip(value,exact)])

def main():
    OUT.mkdir()
    paths=[NOISE/'case-099.json',NOISE/'all-solves.csv',NOISE/'aggregate.json',NOISE/'receipt.json',NOISE/'seal.json',HERE/'audit-01/audit.json',HERE/'audit-supervision-01/receipt.json']
    bindings={str(p.relative_to(HERE)):sha(p) for p in paths}
    case=json.loads(paths[0].read_text());row=case['solves'][99];s=row['solver'];A=np.asarray(s['A']);x=np.asarray(s['solution'])
    assert (row['case'],row['regime'],row['arm'])==('E','clean','N8')
    raw=np.asarray(s['unnormalized_solution']);norm=math.hypot(*raw[:3]);reconstructed=raw/norm*s['cheirality']['selected_sign']
    saved=np.asarray(s['residuals']['vector']);einsum=np.einsum('ij,j->i',A,x,optimize=False);dot=A@x
    exact=exact_mv(A,x);exact_normal=exact_mv(A.T,exact)
    absolute_products=[sum((abs(q(a)*q(v)) for a,v in zip(r,x)),F(0)) for r in A]
    rows=[]
    for i in range(8):
        rows.append(dict(index=i,saved=float(saved[i]),einsum=float(einsum[i]),dot=float(dot[i]),exact_binary64_product_sum=str(exact[i]),exact_rounded=float(exact[i]),sum_absolute_products=float(absolute_products[i]),cancellation_ratio=float(absolute_products[i]/abs(exact[i])) if exact[i] else None,einsum_absolute_error=errors([einsum[i]],[exact[i]])[0],dot_absolute_error=errors([dot[i]],[exact[i]])[0],saved_dot_absolute_difference=float(abs(saved[i]-dot[i]))))
    saved_normal=np.asarray(s['residuals']['normal_equation_vector']);normal_ein=np.einsum('ij,j->i',A.T,einsum,optimize=False);normal_dot=A.T@dot
    csvrows=list(csv.DictReader(paths[1].open()));assert len(csvrows)==13332
    target=csvrows[99*101+99];assert target['status']=='unavailable' and float(target['error_degrees'])==180 and target['case']=='E' and target['regime']=='clean' and target['arm']=='N8'
    agg=json.loads(paths[2].read_text());decisions=[];headroom=[]
    for motion in ('G','N','C'):
        for regime in ('P','S'):
            group=[r for r in csvrows if (r['case'],r['regime'],r['arm'])==(motion,regime,'D4')]
            useful=sum(r['status']=='valid' and float(r['error_degrees'])<=5 for r in group)
            decisions.append(dict(case=motion,regime=regime,attempted=len(group),useful_count=useful,pass_gate=len(group)==1120 and useful>=144))
        d=next(r for r in decisions if r['case']==motion and r['regime']=='S')
        group=[r for r in csvrows if (r['case'],r['regime'],r['arm'])==(motion,'S','A2')]
        c2=sum(r['status']=='valid' and float(r['error_degrees'])<=5 for r in group)
        headroom.append(dict(case=motion,regime='S',D4_attempted=d['attempted'],D4_useful_count=d['useful_count'],A2_attempted=len(group),A2_useful_count=c2,pass_gate=d['attempted']==1120 and len(group)==448 and d['useful_count']>10*c2))
    assert decisions==agg['decisions'] and headroom==agg['optimistic_A2_headroom']
    target_group=next(g for g in agg['groups'] if (g['case'],g['regime'],g['arm'])==('E','clean','N8'))
    gate=all(d['pass_gate'] for d in decisions) and any(d['pass_gate'] for d in headroom)
    assert gate==agg['utility_gate_pass']==False
    original_audit=json.loads(paths[5].read_text());assert original_audit['status']=='failed' and original_audit['failures']==['99/99.residual','99/99.normal_equation','99/99.residual_norm']
    report=dict(schema='evr0025c.posthoc-residual-forensics.v1',scope='Saved-only exact arithmetic on one already-unavailable E-clean N8 row. No decomposition, refit, original audit rerun, fixture generation or model execution.',source_sha256=sha(Path(__file__)),input_sha256=bindings,numpy=np.__version__,original_audit_status='failed',original_audit_failure_list=original_audit['failures'],target={k:row[k] for k in ['record_index','case','regime','arm','status','reason','rank','weak_rank','raw_angle_degrees','error_degrees']},raw_translation_norm=norm,normalization_factor=1/norm,normalized_max_abs_velocity=float(np.max(np.abs(x[3:]))),saved_solution_exactly_matches_frozen_normalization=bool(np.array_equal(x,reconstructed)),saved_residual_bitwise_matches_einsum=bool(np.array_equal(saved,einsum)),saved_normal_bitwise_matches_einsum=bool(np.array_equal(saved_normal,normal_ein)),residual_rows=rows,residual_norms=dict(saved=s['residuals']['norm'],einsum=math.hypot(*einsum),dot=math.hypot(*dot),exact_rounded=math.hypot(*vals(exact))),normal_equation=dict(saved=vals(saved_normal),einsum=vals(normal_ein),dot=vals(normal_dot),exact_AtAx=vals(exact_normal),saved_absolute_errors_vs_exact_AtAx=errors(saved_normal,exact_normal),dot_absolute_errors_vs_exact_AtAx=errors(normal_dot,exact_normal)),csv_count=len(csvrows),target_group=target_group,decisions=decisions,optimistic_A2_headroom=headroom,recomputed_gate=False,residual_diagnostics_used_by_gate=False,original_output_unchanged=all(sha(HERE/k)==v for k,v in bindings.items()))
    assert report['original_output_unchanged']
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['original_audit_status','raw_translation_norm','normalization_factor','saved_residual_bitwise_matches_einsum','saved_normal_bitwise_matches_einsum','residual_norms','recomputed_gate','original_output_unchanged']}))

if __name__=='__main__':main()
