"""Protocol 0002 only: causal scores, 4-pixel endpoints and paired group inference.

Scoring takes raw predictions/timestamps only. Labels are confined to row building
and metrics; this module is intentionally separate from protocol 0001 metrics.
"""
import hashlib
import math
import numpy as np

CAP = -math.log(1e-8)
METHODS = ('native','augmented','visibility','local','random','oracle')


def prediction_scores(times, xy, occ, expected):
    times=np.asarray(times,dtype=np.float64);xy=np.asarray(xy,dtype=np.float64)
    occ=np.asarray(occ,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    n=len(times)
    if xy.shape!=(n,2) or occ.shape!=(n,) or expected.shape!=(n,) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid prediction dimensions/timestamps')
    coordinate=np.isfinite(xy).all(axis=1);logits=np.isfinite(occ)&np.isfinite(expected)
    available=coordinate&logits
    scores={k:np.full(n,np.nan,np.float64) for k in METHODS[:-1]}
    residual=np.full(n,np.nan,np.float64);history=[]
    for i,t in enumerate(times):
        if coordinate[i]:
            d=0.0
            if len(history)>=2:
                (t0,p0),(t1,p1)=history[-2:]
                prediction=p1+(p1-p0)*((t-t1)/(t1-t0))
                d=float(np.linalg.norm(xy[i]-prediction))
            residual[i]=d
            # Score before append; raw coordinates always enter history, even if
            # logits invalid, a prediction is later rejected, or GT is occluded.
            if available[i]:
                v=min(float(np.logaddexp(0.,occ[i])),CAP)
                l=min(float(np.logaddexp(0.,expected[i])),CAP)
                native=min(v+l,CAP)  # algebraically identical; avoids finite-sum overflow
                scores['native'][i]=native;scores['augmented'][i]=native+min(d/8.,10.)
                scores['visibility'][i]=v;scores['local'][i]=l;scores['random'][i]=0.
            history.append((t,xy[i].copy()))
    return {'coordinate_valid':coordinate,'logit_valid':logits,'available':available,
            'residual_px':residual,'scores':scores}


def original_tie(archive_index,video_hash,point,frame):
    return hashlib.sha256(f'0002|{archive_index}|{video_hash}|{point}|{frame}'.encode()).hexdigest()


def rows_from_arrays(index,video_hash,group_id,raw,point_indices,gt_points,gt_occluded):
    raw=np.asarray(raw,dtype=np.float64);T,Q,C=raw.shape
    if C!=4 or Q!=len(point_indices):raise ValueError('raw shape/query mismatch')
    rows=[]
    for q,p in enumerate(point_indices):
        score=prediction_scores(np.arange(T),raw[:,q,:2],raw[:,q,2],raw[:,q,3])
        for t in range(T):
            truth=np.asarray(gt_points[p,t],np.float64)*np.array([256.,256.])
            visible=not bool(gt_occluded[p,t])
            if visible and not np.isfinite(truth).all():raise ValueError('invalid visible annotation')
            coord=bool(score['coordinate_valid'][t]);available=bool(score['available'][t])
            delta=raw[t,q,:2]-truth
            error=float(np.hypot(delta[0],delta[1])) if coord and visible else None
            s={k:float(v[t]) if np.isfinite(v[t]) else None for k,v in score['scores'].items()}
            s['oracle']=error if available and visible else None
            rows.append({'archive_index':index,'group_id':group_id,'point':p,'frame':t,
              'tie':original_tie(index,video_hash,p,t),'post_query':t>0,'visible':visible,
              'opportunity':visible and t>0,'coordinate_valid':coord,
              'logit_valid':bool(score['logit_valid'][t]),'available':available,
              'error_px':error,'failure':(error>4.) if error is not None else None,
              'occlusion_logit':float(raw[t,q,2]) if np.isfinite(raw[t,q,2]) else None,
              'scores':s})
    return rows


def opportunities(rows):return [r for r in rows if r['opportunity']]


def ranked(rows,method):
    return sorted((r for r in opportunities(rows) if r['available'] and r['scores'][method] is not None and math.isfinite(r['scores'][method])),
                  key=lambda r:(r['scores'][method],r['tie']))


def matched(rows,method):
    eligible=opportunities(rows);n=len(eligible);m=math.ceil(.8*n);order=ranked(rows,method)
    reachable=n>0 and len(order)>=m
    return {'opportunities':n,'target_accepts':m,'available':len(order),'reachable':reachable,
            'attainable_coverage':len(order)/n if n else None,
            'coverage':m/n if reachable else None,
            'risk':sum(bool(r['failure']) for r in order[:m])/m if reachable else None}


def calibrate(rows,method):
    n=len(opportunities(rows));m=math.ceil(.8*n);order=ranked(rows,method)
    threshold=None if not order else [order[min(m,len(order))-1]['scores'][method],order[min(m,len(order))-1]['tie']]
    return {'method':method,'threshold':threshold,'opportunities':n,'target_accepts':m,
            'unreachable':n==0 or len(order)<m,'accepted_calibration':min(m,len(order)),
            'calibration_coverage':min(m,len(order))/n if n else None,
            'calibration_risk':sum(bool(r['failure']) for r in order[:m])/min(m,len(order)) if order and m else None}


def operational(rows,method,calibration):
    eligible=opportunities(rows);n=len(eligible);threshold=calibration['threshold']
    selected=[] if threshold is None else [r for r in ranked(rows,method) if (r['scores'][method],r['tie'])<=tuple(threshold)]
    return {'opportunities':n,'accepted':len(selected),'coverage':len(selected)/n if n else None,
            'risk':sum(bool(r['failure']) for r in selected)/len(selected) if selected else None}


def curve(rows,method):
    n=len(opportunities(rows));order=ranked(rows,method);fail=0;points=[]
    for j,r in enumerate(order,1):
        fail+=bool(r['failure']);points.append([j/n,fail/j])
    return points


def expand_draw(rows,draw):
    groups={}
    for r in rows:groups.setdefault(r['group_id'],[]).append(r)
    # Deliberately retain every sampled copy and original tie; NEVER de-duplicate.
    return [r for group in draw for r in groups.get(group,[])]


def paired_effect(rows,thresholds):
    a=matched(rows,'augmented');n=matched(rows,'native')
    oa=operational(rows,'augmented',thresholds['augmented']);on=operational(rows,'native',thresholds['native'])
    sub=lambda x,y:None if x is None or y is None else x-y
    return {'matched_risk_difference':sub(a['risk'],n['risk']),
            'operational_risk_difference':sub(oa['risk'],on['risk']),
            'operational_coverage_difference':sub(oa['coverage'],on['coverage'])}


def paired_bootstrap(rows,allocated_groups,thresholds,repeats=2000,seed=20261018):
    groups=list(allocated_groups)
    if len(groups)!=len(set(groups)) or not groups:raise ValueError('allocated group frame must be nonempty and unique')
    if not {r['group_id'] for r in rows}<=set(groups):raise ValueError('row outside allocated frame')
    rng=np.random.default_rng(seed);samples={k:[] for k in paired_effect(rows,thresholds)}
    for _ in range(repeats):
        draw=[groups[i] for i in rng.integers(0,len(groups),len(groups))]
        value=paired_effect(expand_draw(rows,draw),thresholds)
        for k,v in value.items():
            if v is not None and np.isfinite(v):samples[k].append(v)
    return {'unit':'allocated exact-duplicate group; empty groups retained','allocated_groups':groups,
      'repeats':repeats,'seed':seed,'intervals':{k:{'defined_replicates':len(v),
        'ci95':np.percentile(v,[2.5,97.5]).tolist() if len(v)>=math.ceil(.95*repeats) else None}
        for k,v in samples.items()}}


def evaluate(calibration_rows,evaluation_rows,allocated_groups,repeats=2000,seed=20261018):
    thresholds={m:calibrate(calibration_rows,m) for m in METHODS[:-1]}
    rows=opportunities(evaluation_rows);n=len(rows)
    coord=[r for r in rows if r['coordinate_valid']]
    allpost=[r for r in evaluation_rows if r['post_query']]
    # Visibility head only: sigmoid(-occ)>0.5 iff finite occ<0. Missing occ is
    # incorrect, included in the all-postquery denominator, even if GT occluded.
    vis_correct=sum(r['occlusion_logit'] is not None and ((r['occlusion_logit']<0)==r['visible']) for r in allpost)
    boot=paired_bootstrap(evaluation_rows,allocated_groups,thresholds,repeats,seed)
    effect=paired_effect(evaluation_rows,thresholds)
    interval=boot['intervals']['matched_risk_difference']['ci95'];point=effect['matched_risk_difference']
    return {'thresholds':thresholds,'matched':{m:matched(evaluation_rows,m) for m in METHODS},
      'operational':{m:operational(evaluation_rows,m,thresholds[m]) for m in METHODS[:-1]},
      'curves':{m:curve(evaluation_rows,m) for m in METHODS},'effect':effect,'bootstrap':boot,
      'primary_opportunities':n,'coordinate_missing_rate':1-len(coord)/n if n else None,
      'joint_score_unavailable_rate':sum(not r['available'] for r in rows)/n if n else None,
      'estimator_failure_including_missing':sum(not r['coordinate_valid'] or bool(r['failure']) for r in rows)/n if n else None,
      'accept_all_coordinates':{'coverage':len(coord)/n if n else None,'risk':sum(bool(r['failure']) for r in coord)/len(coord) if coord else None},
      'visibility_accuracy':vis_correct/len(allpost) if allpost else None,
      'practical_signal':bool(point is not None and point<=-.02 and interval is not None and interval[1]<0),
      'interpretation':'exploratory; unknown scene dependence and upstream exposure; no conformal/general efficacy guarantee'}
