"""Qualification controls; no import or execution of a native module at load time.

Scale-only tangent alignment follows Eichhardt and Barath, ECCV2020, Eq.11
and the explicitly second-column-ordered basis of Algorithm 1.
The point control calls the separately pinned, existing PoseLib implementation.
"""
import math
import numpy as np


def _array(value, shape):
    if np.iscomplexobj(value):
        raise ValueError('inputs must be real')
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError('invalid finite input shape')
    return result


def _scalar(value):
    if isinstance(value,(bool,np.bool_)) or np.iscomplexobj(value):
        raise ValueError('depth must be a finite real scalar')
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=() or not np.isfinite(result):
        raise ValueError('depth must be a finite real scalar')
    return float(result)


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError('nonfinite numerical intermediate')
    return value


def _mm(a,b):
    # Avoid the platform BLAS floating-status issue seen in earlier work.
    return np.einsum('ij,jk->ik',a,b,optimize=False)


def _safe(value):
    """Retain numerical failures as null observations, never JSON NaN/Infinity."""
    if isinstance(value,np.ndarray):
        return _safe(value.tolist())
    if isinstance(value,np.generic):
        return _safe(value.item())
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    if isinstance(value,dict):
        return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_safe(v) for v in value]
    if value is None or isinstance(value,(str,bool,int)):
        return value
    raise TypeError('unsupported output observation')


def _unit(vector):
    scale=float(np.max(np.abs(vector)))
    if not math.isfinite(scale) or scale<=0:
        raise FloatingPointError('unresolved ordered-frame direction')
    scaled=vector/scale
    length=float(_finite(np.sqrt(np.sum(scaled*scaled))))
    if length<=0:
        raise FloatingPointError('unresolved ordered-frame norm')
    return _finite(scaled/length)


def _ordered_frame(tangents):
    # Published Alg.1: rx = unit(y1 × y2), rz = unit(y2),
    # frame = [rx, rz × rx, rz]. Pre-normalization preserves the cross direction.
    rz=_unit(tangents[:,1])
    rx=_unit(np.cross(_unit(tangents[:,0]),rz))
    return _finite(np.column_stack((rx,np.cross(rz,rx),rz)))


def _solve_scale_only(r1,r2,E1,E2,d1,d2,g1,g2,A,ordered):
    r1=_array(r1,(3,));r2=_array(r2,(3,))
    E1=_array(E1,(3,2));E2=_array(E2,(3,2));A=_array(A,(2,2))
    g1=_array(g1,(2,));g2=_array(g2,(2,))
    d1=_scalar(d1);d2=_scalar(d2)
    if r1[2]!=1. or r2[2]!=1. or np.any(E1[2]!=0.) or np.any(E2[2]!=0.):
        raise ValueError('axial model requires ray z exactly one and Jacobian z row zero')
    record={'method':'scale_only_tangent_ordered' if ordered else 'scale_only_tangent',
        'rotation_method':'published_second_column_ordered' if ordered else 'proper_svd',
        'candidates':[],
        'inputs':dict(r1=r1,r2=r2,E1=E1,E2=E2,d1=d1,d2=d2,g1=g1,g2=g2,A=A)}
    if d1<=0 or d2<=0:
        return _safe(dict(record,status='unavailable',reason='nonpositive_raw_depth'))
    phase='tangents'
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            U=_finite(d1*E1+np.outer(r1,g1))
            record['U']=U
            V=_finite(_mm(_finite(d2*E2+np.outer(r2,g2)),A))
            record['V']=V
            phase='tangent_rank'
            su=_finite(np.linalg.svd(U,compute_uv=False))
            sv=_finite(np.linalg.svd(V,compute_uv=False))
            record.update(source_singular_values=su,target_singular_values=sv)
            tolerance=3*np.finfo(float).eps
            if su[-1]<=tolerance*su[0] or sv[-1]<=tolerance*sv[0]:
                return _safe(dict(record,status='unavailable',reason='rank_deficient_tangent'))
            phase='proper_rotation_and_scale'
            if ordered:
                R=_finite(_mm(_ordered_frame(V),_ordered_frame(U).T))
                numerator=float(_finite(np.sum(_mm(R,U)*V)))
            else:
                covariance=_finite(_mm(V,U.T))
                left,singular,right=np.linalg.svd(covariance)
                _finite(left);_finite(singular);_finite(right)
                signs=np.array([1.,1.,1. if np.linalg.det(_mm(left,right))>=0 else -1.])
                R=_finite(_mm(left*signs,right))
                numerator=float(_finite(np.sum(singular*signs)))
            denominator=float(_finite(np.sum(V*V)))
            if denominator<=0:
                raise FloatingPointError('unresolved scale denominator')
            a=float(numerator/denominator)
            record['scale']=a
            _finite(a)
            if a<=0:
                return _safe(dict(record,status='unavailable',reason='nonpositive_relative_scale'))
            phase='pose_and_diagnostics'
            target_depth=float(_finite(a*d2))
            t=_finite(target_depth*r2-np.einsum('ij,j->i',R,d1*r1,optimize=False))
            orth=float(_finite(np.max(np.abs(_mm(R.T,R)-np.eye(3)))))
            det=float(_finite(np.linalg.det(R)))
            residual=float(_finite(np.sum((_mm(R,U)-a*V)**2)))
            directed=bool(np.any(t!=0.))  # no squared-norm underflow or near-zero cutoff
            proper=bool(orth<=1e-8 and abs(det-1)<=1e-8)
            physical=bool(target_depth>0.)
            candidate={'b':0.,'a':a,'c':0.,'R':R,'t':t,'physical':physical,
                'proper_rotation':proper,'translation_direction_defined':directed,
                'pose_usable':physical and proper and directed,
                'tangent_squared_residual':residual,'rotation_orthogonality_max':orth,
                'rotation_determinant':det}
            return _safe(dict(record,status='completed',candidates=[candidate]))
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
        return _safe(dict(record,status='numerical_failure',reason='numerical_failure',
            failed_phase=phase,failure={'type':type(error).__name__,'message':str(error)}))


def solve_scale_only(r1,r2,E1,E2,d1,d2,g1,g2,A):
    return _solve_scale_only(r1,r2,E1,E2,d1,d2,g1,g2,A,ordered=False)


def solve_scale_only_ordered(r1,r2,E1,E2,d1,d2,g1,g2,A):
    return _solve_scale_only(r1,r2,E1,E2,d1,d2,g1,g2,A,ordered=True)


def solve_points_native(x1,x2,d1,d2,native):
    x1=_array(x1,(3,3));x2=_array(x2,(3,3))
    d1=_array(d1,(3,));d2=_array(d2,(3,))
    if not np.array_equal(x1[:,2],np.ones(3)) or not np.array_equal(x2[:,2],np.ones(3)):
        raise ValueError('PoseLib point inputs require axial rays with z exactly one')
    # Unexpected native/ABI exceptions are technical failures, never zero-root cases.
    solutions=native.monodepth_pose_3pt(x1,x2,d1,d2)
    candidates=[]
    for index,s in enumerate(solutions):
        if np.iscomplexobj(s.pose.R) or np.iscomplexobj(s.pose.t):
            raise ValueError('native solver returned complex geometry')
        R=np.asarray(s.pose.R,dtype=float);t=np.asarray(s.pose.t,dtype=float).reshape(3)
        a=float(s.scale);b=float(s.shift1);shift2=float(s.shift2);c=a*shift2
        if R.shape!=(3,3):raise ValueError('native solver returned malformed geometry')
        row={'native_index':index,'a':a,'b':b,'c':c,'native_shift2':shift2,
             'R':R,'t':t,'pose_usable':False}
        if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite([a,b,c]).all():
            candidates.append(_safe(dict(row,status='invalid',reason='nonfinite_native_output')))
            continue
        try:
            # Preserve derived nonfinite observations as null, with invalid status.
            # This does not catch unexpected native API/schema exceptions above.
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                source=d1+b;target=a*d2+c
                row.update(source_depths=source,target_depths=target)
                _finite(source);_finite(target)
                orth=float(np.max(np.abs(_mm(R.T,R)-np.eye(3))))
                det=float(np.linalg.det(R))
                residual=float(np.sum((_mm(source[:,None]*x1,R.T)+t-target[:,None]*x2)**2))
                row.update(rotation_orthogonality_max=orth,rotation_determinant=det,
                           point_squared_residual=residual)
                _finite([orth,det,residual])
            physical=bool(a>0 and np.all(source>0) and np.all(target>0))
            proper=bool(orth<=1e-8 and abs(det-1)<=1e-8)
            directed=bool(np.any(t!=0.))
            row.update(status='completed',physical=physical,proper_rotation=proper,
                translation_direction_defined=directed,pose_usable=physical and proper and directed)
        except (FloatingPointError,OverflowError,np.linalg.LinAlgError) as error:
            row.update(status='invalid',reason='nonfinite_or_unresolved_derived_arithmetic',
                failure={'type':type(error).__name__,'message':str(error)})
        candidates.append(_safe(row))
    return {'method':'native_three_point_affine_depth','status':'completed','native_returned':len(solutions),
        'candidates':candidates,'usable_count':sum(c['pose_usable'] for c in candidates)}
