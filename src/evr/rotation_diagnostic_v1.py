"""Failure-follow-up only: exact frame-0 camera extraction and unchanged gate math.

No projection, trajectory, camera repair, transform search, model or NumPy.
"""
import hashlib
import itertools
import math
import struct
import zipfile
from fractions import Fraction
from evr.bounded_point_audit import ReadBudget
from evr.motion_design import safe_member
from evr.point_geometry_v2 import header,offset,zip_envelope

TAU=9.5367431640625e-7  # unchanged max(1e-8,8*2^-23), not a fitted tolerance
CARRIED_RETURNED=20614538
CARRIED_RESERVES=2163798
CARRIED_DIRECTORY=1110


def continued_budget():
    b=ReadBudget();b.returned_bytes=CARRIED_RETURNED;b.stream_reserve_bytes=CARRIED_RESERVES
    b.central_directory_reserved_bytes=CARRIED_DIRECTORY
    return b


def frame0(stream,h,budget):
    """Only n*n frame0 scalars decoded. Every physical gap is read+charged."""
    shape=h['shape']
    if len(shape)!=3 or shape[0]<1 or shape[1]!=shape[2] or shape[1]not in(3,4) or h['dtype']not in('<f4','>f4','=f4'):
        raise ValueError('explicit f32 [T,3,3]/[T,4,4] camera only')
    n=shape[1];requests=sorted((offset(h,(0,i,j)),i,j)for i in range(n)for j in range(n))
    span=requests[-1][0]+4
    if budget.charged_bytes+span>budget.limit:raise ValueError('physical frame0 span exceeds budget BEFORE payload')
    entries=[[None]*n for _ in range(n)];cursor=0;sha=hashlib.sha256()
    for physical,i,j in requests:
        gap=physical-cursor
        while gap:
            raw=budget.read(stream,min(65536,gap));sha.update(raw);cursor+=len(raw);gap-=len(raw)
        raw=budget.read(stream,4);sha.update(raw);cursor+=4
        value=struct.unpack(h['format'],raw)[0]
        entries[i][j]={'byte_offset_in_payload':physical,'stored_bytes_hex':raw.hex(),'float_hex':value.hex(),
                       'exact_rational':str(Fraction.from_float(value)) if math.isfinite(value)else None,'finite':math.isfinite(value)}
    return entries,{'retained_scalars':n*n,'retained_bytes':n*n*4,'numeric_span_bytes':span,
                   'discarded_bytes':span-n*n*4,'span_sha256':sha.hexdigest(),
                   'semantic_access':'frame0 only; discarded gaps and Zip buffering are not decoded; no full CRC/tail claim'}


def read_cameras(path,budget,result,checkpoint):
    result['zip_envelope']=zip_envelope(path,budget)
    with zipfile.ZipFile(path)as z:
        infos=z.infolist();seen=set();byname={}
        if not 0<len(infos)<=64:raise ValueError('NPZ member cap')
        for info in infos:
            if '\0'in info.orig_filename:raise ValueError('NUL ZIP name')
            name=safe_member(info.filename)
            # Central-directory metadata only for OTHER names. Never z.open them.
            import stat
            if name in seen or '/'in name or not name.endswith('.npy')or info.flag_bits&1 or info.compress_type not in(0,8)or stat.S_IFMT(info.external_attr>>16)not in(0,stat.S_IFREG):raise ValueError('unsafe/duplicate ZIP member')
            seen.add(name);byname[name]=info
        result['camera_access']={}
        for key,n in [('intrinsics',3),('extrinsics',4)]:
            name=key+'.npy'
            if name not in byname:raise ValueError('required camera member missing')
            entry={'status':'CAMERA_MEMBER_STARTING','counts_complete':False};result['camera_access'][key]=entry;checkpoint()
            budget.open_stream()
            with z.open(byname[name])as stream:
                h=header(stream,budget)
                if h['shape']!=[849,n,n] or h['dtype']!='<f4' or h['order']!='C':raise ValueError('pinned observed camera schema mismatch')
                if byname[name].file_size!=h['header_bytes']+h['data_bytes']:raise ValueError('shape/member byte mismatch')
                entry.update(header=h,status='NUMERIC_CAMERA_ACCESS_STARTING');checkpoint()
                before=budget.returned_bytes
                try:
                    entries,account=frame0(stream,h,budget);entry.update(values=entries,accounting=account,status='completed')
                finally:
                    entry.update(numeric_bytes_returned=budget.returned_bytes-before,counts_complete=True);checkpoint()
    return result['camera_access']


def _matrix(a,n):
    if len(a)!=n or any(len(r)!=n for r in a):raise ValueError('matrix row/column format')
    for v in itertools.chain.from_iterable(a):
        if type(v)not in(int,float)or not math.isfinite(v):raise ValueError('finite real camera coefficients required')
        try:same=struct.unpack('<f',struct.pack('<f',v))[0]
        except (OverflowError,struct.error)as e:raise ValueError('f32 stored coefficient required')from e
        if same!=v:raise ValueError('coefficient not exactly representable as source f32')


def _max(a):
    values=[(abs(x),i,j)for i,row in enumerate(a)for j,x in enumerate(row)]
    maximum=max(x[0]for x in values)
    return maximum,next([i,j]for x,i,j in values if x==maximum)


def double_statistics(R):
    # EXACT operation order of the old project_v2 rotation checks.
    rr=[[sum(R[i][k]*R[j][k]for k in range(3))-(i==j)for j in range(3)]for i in range(3)]
    rt=[[sum(R[k][i]*R[k][j]for k in range(3))-(i==j)for j in range(3)]for i in range(3)]
    det=sum(R[0][j]*(R[1][(j+1)%3]*R[2][(j+2)%3]-R[1][(j+2)%3]*R[2][(j+1)%3])for j in range(3))
    if any(not math.isfinite(x)for x in list(itertools.chain.from_iterable(rr+rt))+[det]):raise ValueError('double arithmetic nonfinite')
    return rr,rt,det


def exact_statistics(R):
    """Independent exact dyadic arithmetic; no double-derived Gram or determinant."""
    q=[[Fraction.from_float(float(v))for v in row]for row in R]
    columns=list(zip(*q))
    def gram(vectors):
        out=[]
        for i,a in enumerate(vectors):
            row=[]
            for j,b in enumerate(vectors):
                products=[x*y for x,y in zip(a,b)]
                row.append(sum(products,Fraction())-Fraction(int(i==j)))
            out.append(row)
        return out
    determinant=Fraction()
    for permutation in itertools.permutations(range(3)):
        inversions=sum(permutation[i]>permutation[j]for i in range(3)for j in range(i+1,3))
        term=math.prod(q[i][permutation[i]]for i in range(3))
        determinant+=(-1 if inversions%2 else 1)*term
    return gram(q),gram(columns),determinant


def rounding_bounds():
    u=Fraction(1,2**24)
    return {'u':u,'gram':6*u+3*u*u,'determinant':9*u+27*u*u+27*u*u*u}


def _rational(x):
    return {'numerator':str(x.numerator),'denominator':str(x.denominator)}


def binary16_exact(value):
    """Exact signed-bit roundtrip of one original finite f32; never a repair."""
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite original f32 required')
    try:
        original=struct.pack('<f',value)
        if struct.unpack('<f',original)[0]!=value:raise ValueError('original f32 required')
        widened=struct.unpack('<e',struct.pack('<e',value))[0]
    except (OverflowError,struct.error):
        return False
    return original==struct.pack('<f',widened)


def binary16_groups(K,RT):
    """Describe same 25 values only; no converted matrix or gate evaluation."""
    _matrix(K,3);_matrix(RT,4)
    def group(matrix,indices):
        rows=[{'index':[i,j],'original_f32_le_hex':struct.pack('<f',matrix[i][j]).hex(),
               'exactly_binary16':binary16_exact(matrix[i][j])}for i,j in indices]
        return {'total':len(rows),'exactly_representable_count':sum(r['exactly_binary16']for r in rows),
                'all_exactly_representable':all(r['exactly_binary16']for r in rows),'values':rows}
    return {'method':'struct half roundtrip; exact original f32 bits including signed zero; overflow is not representable; no tolerance',
      'groups':{'K9':group(K,[(i,j)for i in range(3)for j in range(3)]),
                'R9':group(RT,[(i,j)for i in range(3)for j in range(3)]),
                'RT16':group(RT,[(i,j)for i in range(4)for j in range(4)]),
                'translation3':group(RT,[(i,3)for i in range(3)]),
                'homogeneous4':group(RT,[(3,j)for j in range(4)])},
      'interpretation':'All-on-half-grid is compatible with direct f16 cast then exact widening to f32, not proof (integers and identity also qualify). Any nonrepresentable value excludes only an all-values direct f16-to-f32 path with no intervening arithmetic; earlier half followed by computation remains possible. No causal probability or actual-pipeline identification.'}


def diagnose(K,RT):
    _matrix(K,3);_matrix(RT,4)
    R=[r[:3]for r in RT[:3]];rr,rt,det=double_statistics(R);err_rr,arg_rr=_max(rr);err_rt,arg_rt=_max(rt)
    checks={'finite':True,'K_exact_bottom_row':list(K[2])==[0,0,1],
            'RT_exact_bottom_row':list(RT[3])==[0,0,0,1],'positive_focal':K[0][0]>0 and K[1][1]>0,
            'RRt':err_rr<=TAU,'RtR':err_rt<=TAU,'determinant':abs(det-1)<=TAU}
    exact_rr,exact_rt,exact_det=exact_statistics(R);exact_tau=Fraction.from_float(TAU)
    exact_err_rr,_=_max(exact_rr);exact_err_rt,_=_max(exact_rt)
    bounds=rounding_bounds()
    violations={name:{'value':value,'threshold':TAU,'excess':max(0.,value-TAU),'ratio':value/TAU,'passes':value<=TAU}
                for name,value in [('RRt',err_rr),('RtR',err_rt),('determinant',abs(det-1))]}
    differences={}
    for name,a,b in [('RRt',rr,exact_rr),('RtR',rt,exact_rt)]:
        differences[name]=[[_rational(Fraction.from_float(float(x))-y)for x,y in zip(row,exactrow)]for row,exactrow in zip(a,b)]
    differences['determinant']=_rational(Fraction.from_float(float(det))-exact_det)
    return {'binary16_exact_representability':binary16_groups(K,RT),
      'original_tau':TAU,'original_gate_checks':checks,'original_gate_pass':all(checks.values()),
      'original_rotation_gate_pass':checks['RRt'] and checks['RtR'] and checks['determinant'],
      'depth_or_projection':'NOT_TESTED: forbidden world/point access',
      'double':{'RRt_minus_I':rr,'RtR_minus_I':rt,'maxabs_RRt':err_rr,'argmax_RRt':arg_rr,'maxabs_RtR':err_rt,'argmax_RtR':arg_rt,
                'det_R':det,'abs_det_minus_one':abs(det-1),'row_squared_norms':[sum(R[i][k]*R[i][k]for k in range(3))for i in range(3)],
                'column_squared_norms':[sum(R[k][i]*R[k][i]for k in range(3))for i in range(3)],
                'row_offdiagonal_dots':[[i,j,rr[i][j]]for i in range(3)for j in range(i+1,3)],
                'column_offdiagonal_dots':[[i,j,rt[i][j]]for i in range(3)for j in range(i+1,3)]},
      'violations':violations,'exact_fraction':{'RRt_minus_I':[[_rational(x)for x in row]for row in exact_rr],
        'RtR_minus_I':[[_rational(x)for x in row]for row in exact_rt],'det_R':_rational(exact_det),
        'rotation_gate_pass':max(exact_err_rr,exact_err_rt)<=exact_tau and abs(exact_det-1)<=exact_tau},
      'double_minus_exact_fraction':differences,
      'single_final_f32_rounding':{'bounds':{k:_rational(v)for k,v in bounds.items()},
        'excluded':max(exact_err_rr,exact_err_rt)>bounds['gram']or abs(exact_det-1)>bounds['determinant'],
        'interpretation':'only necessary bounds for one nearest-f32 rounding of exact SO(3); inside is not causal proof, outside is not root-cause identification'},
      'invariance':'transpose swaps Gram products and preserves determinant; handedness-preserving signed permutations preserve exact max-entry qualification. Orientation-reversing changes can flip determinant and are forbidden repair attempts; no transformed real-data candidates calculated.'}
