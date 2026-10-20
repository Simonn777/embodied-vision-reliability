"""Qualification controls; no import or execution of a native module at load time.

Scale-only tangent alignment follows Eichhardt and Barath, ECCV2020, Eq.11.
The point control calls the separately pinned, existing PoseLib implementation.
"""
import numpy as np


def _array(value, shape):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError('invalid finite input shape')
    return result


def solve_scale_only(r1,r2,E1,E2,d1,d2,g1,g2,A):
    r1=_array(r1,(3,));r2=_array(r2,(3,))
    E1=_array(E1,(3,2));E2=_array(E2,(3,2));A=_array(A,(2,2))
    g1=_array(g1,(2,));g2=_array(g2,(2,))
    d1=float(d1);d2=float(d2)
    if not np.isfinite([d1,d2]).all():raise ValueError('nonfinite depth')
    U=d1*E1+np.outer(r1,g1);V=(d2*E2+np.outer(r2,g2))@A
    record={'method':'scale_only_tangent','U':U.tolist(),'V':V.tolist(),'candidates':[]}
    if d1<=0 or d2<=0:
        return dict(record,status='unavailable',reason='nonpositive_raw_depth')
    su=np.linalg.svd(U,compute_uv=False);sv=np.linalg.svd(V,compute_uv=False)
    record.update(source_singular_values=su.tolist(),target_singular_values=sv.tolist())
    tolerance=3*np.finfo(float).eps
    if su[-1]<=tolerance*su[0] or sv[-1]<=tolerance*sv[0]:
        return dict(record,status='unavailable',reason='rank_deficient_tangent')
    left,singular,right=np.linalg.svd(V@U.T)
    signs=np.array([1.,1.,1. if np.linalg.det(left@right)>=0 else -1.])
    R=(left*signs)@right
    a=float(np.dot(singular,signs)/np.sum(V*V))
    if not np.isfinite(a) or a<=0:
        return dict(record,status='unavailable',reason='nonpositive_relative_scale')
    t=a*d2*r2-R@(d1*r1)
    usable=bool(np.linalg.norm(t)>0)
    candidate={'b':0.,'a':a,'c':0.,'R':R.tolist(),'t':t.tolist(),
        'translation_direction_defined':usable,'pose_usable':usable,
        'tangent_squared_residual':float(np.sum((R@U-a*V)**2)),
        'rotation_orthogonality_max':float(np.max(np.abs(R.T@R-np.eye(3)))),
        'rotation_determinant':float(np.linalg.det(R))}
    return dict(record,status='completed',candidates=[candidate])


def solve_points_native(x1,x2,d1,d2,native):
    x1=_array(x1,(3,3));x2=_array(x2,(3,3))
    d1=_array(d1,(3,));d2=_array(d2,(3,))
    if not np.array_equal(x1[:,2],np.ones(3)) or not np.array_equal(x2[:,2],np.ones(3)):
        raise ValueError('PoseLib point inputs require axial rays with z exactly one')
    # Unexpected native/ABI exceptions are technical failures, never zero-root cases.
    solutions=native.monodepth_pose_3pt(x1,x2,d1,d2)
    candidates=[]
    for index,s in enumerate(solutions):
        R=np.asarray(s.pose.R,dtype=float);t=np.asarray(s.pose.t,dtype=float).reshape(3)
        a=float(s.scale);b=float(s.shift1);c=a*float(s.shift2)
        if R.shape!=(3,3):raise ValueError('native solver returned malformed geometry')
        if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite([a,b,c]).all():
            safe=lambda x: float(x) if np.isfinite(x) else None
            candidates.append({'native_index':index,'status':'invalid','reason':'nonfinite_native_output',
                'a':safe(a),'b':safe(b),'c':safe(c),'R':[[safe(v) for v in row] for row in R],
                't':[safe(v) for v in t],'pose_usable':False})
            continue
        source=d1+b;target=a*d2+c
        orth=float(np.max(np.abs(R.T@R-np.eye(3))));det=float(np.linalg.det(R))
        physical=bool(a>0 and np.all(source>0) and np.all(target>0))
        proper=bool(orth<=1e-8 and abs(det-1)<=1e-8)
        directed=bool(np.linalg.norm(t)>0)
        candidates.append({'native_index':index,'b':b,'a':a,'c':c,'R':R.tolist(),'t':t.tolist(),
            'source_depths':source.tolist(),'target_depths':target.tolist(),
            'physical':physical,'rotation_orthogonality_max':orth,'rotation_determinant':det,
            'proper_rotation':proper,'translation_direction_defined':directed,
            'pose_usable':physical and proper and directed,
            'point_squared_residual':float(np.sum(((source[:,None]*x1)@R.T+t-target[:,None]*x2)**2))})
    return {'method':'native_three_point_affine_depth','status':'completed','native_returned':len(solutions),
        'candidates':candidates,'usable_count':sum(c['pose_usable'] for c in candidates)}
