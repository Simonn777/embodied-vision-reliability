"""0003 prospective C/F scalar streaming and development geometry; stdlib only.

No NumPy, pickle, network, model, coordinate fitting or implicit axis conversion.
This version is independent of the immutable C-only resource-v1 reader.
"""
import ast
import hashlib
import itertools
import math
import re
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from evr.bounded_point_audit import ReadBudget
from evr.motion_design import safe_member

CAP = 64 * 1024**2
MAX_SCALARS = 1_000_000
KINDS = {('f',4):'f',('f',8):'d',('i',1):'b',('i',2):'h',('i',4):'i',('i',8):'q',
         ('u',1):'B',('u',2):'H',('u',4):'I',('u',8):'Q',('b',1):'B'}


def header(stream, budget, metadata_only=False):
    start = budget.returned_bytes
    if budget.read(stream, 6) != b'\x93NUMPY':
        raise ValueError('NPY magic')
    version = tuple(budget.read(stream, 2))
    if version not in [(1,0),(2,0),(3,0)]:
        raise ValueError('NPY version')
    length = struct.unpack('<H' if version == (1,0) else '<I', budget.read(stream, 2 if version == (1,0) else 4))[0]
    if not 0 < length <= 65536:
        raise ValueError('NPY header cap')
    raw = budget.read(stream, length)
    try:
        node = ast.parse(raw.decode('utf8' if version == (3,0) else 'latin1').strip(), mode='eval').body
        if not isinstance(node, ast.Dict) or len(node.keys) != 3:
            raise ValueError('exact header dictionary')
        keys = [ast.literal_eval(k) for k in node.keys]
        if len(set(keys)) != 3:
            raise ValueError('duplicate header keys')
        data = ast.literal_eval(node)
    except (SyntaxError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError('literal NPY header') from exc
    if set(data) != {'descr','shape','fortran_order'} or type(data['fortran_order']) is not bool:
        raise ValueError('NPY header keys/order')
    shape = data['shape']
    if type(shape) is not tuple or not (0 if metadata_only else 1) <= len(shape) <= 4 or any(type(x) is not int or not 0 < x <= 10_000_000 for x in shape):
        raise ValueError('bounded nonempty shape')
    descr = data['descr']
    if not isinstance(descr, str):
        raise ValueError('structured/object dtype forbidden')
    if metadata_only and re.fullmatch(r'[<>|=]O[0-9]*', descr):
        return {'shape':list(shape),'dtype':descr,'order':'F' if data['fortran_order'] else 'C',
                'header_bytes':budget.returned_bytes-start,'object_payload':'FORBIDDEN_NOT_DECODED'}
    m = re.fullmatch(r'([<>|=])([fiub])([1248])', descr)
    if not m or (m[2], int(m[3])) not in KINDS or m[1] == '|' and m[3] != '1':
        raise ValueError('unsupported real scalar dtype')
    size = int(m[3]); endian = m[1]
    endian = ('<' if sys.byteorder == 'little' else '>') if endian == '=' else '<' if endian == '|' else endian
    total = math.prod(shape) * size
    if total > 2*1024**3:
        raise ValueError('NPY declared data cap')
    return {'shape': list(shape), 'dtype': descr, 'order': 'F' if data['fortran_order'] else 'C',
            'format': endian + KINDS[m[2],size], 'itemsize': size, 'data_bytes': total,
            'header_bytes': budget.returned_bytes-start, 'header_sha256': hashlib.sha256(raw).hexdigest()}


def offset(h, indices):
    shape = h['shape']
    if len(indices) != len(shape) or any(type(i) is not int or not 0 <= i < n for i,n in zip(indices,shape)):
        raise ValueError('scalar indices')
    order = range(len(shape)-1,-1,-1) if h['order'] == 'C' else range(len(shape))
    stride = h['itemsize']; result = 0
    for axis in order:
        result += indices[axis] * stride
        stride *= shape[axis]
    return result


def read_prefix(stream, h, budget, constant=False):
    """Retain all points' first 3 times in C logical order, scan C OR F physical order.

    Every skipped byte is actually read and charged. No seek/decompression shortcut.
    Tail after last requested physical offset is NOT read or claimed CRC-verified.
    """
    shape = h['shape']; saved = shape[:] if constant else [3] + shape[1:]
    if not constant and shape[0] < 3:
        raise ValueError('at least 3 frames')
    count = math.prod(saved)
    if count > MAX_SCALARS:
        raise ValueError('retained scalar cap')
    requests = sorted((offset(h, ix), logical) for logical,ix in enumerate(itertools.product(*(range(x) for x in saved))))
    span = requests[-1][0] + h['itemsize']
    if budget.charged_bytes + span > budget.limit:
        raise ValueError('physical span exceeds cumulative cap before numeric read')
    values = [None] * count; cursor = 0; digest = hashlib.sha256()
    # Stream blocks, not one Python read per discarded scalar. Retain max 64 KiB block.
    request_index = 0
    while cursor < span:
        # all supported item sizes divide 65536, so no scalar straddles a block
        block = budget.read(stream, min(65536, span-cursor)); digest.update(block)
        end = cursor + len(block)
        while request_index < len(requests) and requests[request_index][0] < end:
            physical, logical = requests[request_index]
            values[logical] = struct.unpack_from(h['format'], block, physical-cursor)[0]
            request_index += 1
        cursor = end
    return Prefix(saved, values), {'retained_scalars':count,'numeric_stream_span_bytes':span,
        'discarded_bytes':span-count*h['itemsize'],'returned_span_sha256':digest.hexdigest(),
        'scope':'all points, first three times; discarded physical-gap values not interpreted; no tail/CRC claim'}


@dataclass
class Prefix:
    shape: list
    values: list

    def at(self, *indices):
        # Logical C storage independent of original physical strides.
        if len(indices) != len(self.shape) or any(type(i) is not int or not 0 <= i < n for i,n in zip(indices,self.shape)):
            raise ValueError('saved-prefix indices')
        flat = 0
        for i,n in zip(indices,self.shape): flat = flat*n+i
        return self.values[flat]


def inspect_zip(z, budget):
    infos = z.infolist()
    if not 0 < len(infos) <= 64:
        raise ValueError('NPZ member cap')
    headers = {}; members = {}
    for info in infos:
        if '\0' in info.orig_filename: raise ValueError('NUL ZIP member')
        name = safe_member(info.filename)
        if '/' in name or not name.endswith('.npy') or info.flag_bits & 1 or info.compress_type not in (0,8) or stat.S_IFMT(info.external_attr>>16) not in (0,stat.S_IFREG):
            raise ValueError('regular flat unencrypted NPZ only')
        if name in members:
            raise ValueError('duplicate NPZ member')
        budget.open_stream()
        with z.open(info) as stream: h = header(stream, budget)
        if info.file_size != h['header_bytes'] + h['data_bytes']:
            raise ValueError('NPY shape/member length mismatch')
        members[name] = info; headers[name[:-4]] = h
    return headers, members


def load_geometry(z, budget, expected_TN=None, progress=None):
    hs, members = inspect_zip(z,budget)
    required = ['trajs_2d','trajs_3d','visibs','valids','intrinsics','extrinsics']
    if set(hs) != set(required):
        raise ValueError('exact observed six-array schema required')
    T,N,*rest = hs['trajs_2d']['shape']
    if rest != [2] or T < 3 or hs['trajs_3d']['shape'] != [T,N,3] or any(hs[k]['shape'] != [T,N] for k in ['visibs','valids']):
        raise ValueError('time/point alignment')
    if expected_TN is not None and [T,N] != list(expected_TN):
        raise ValueError('observed T/N schema changed before numeric read')
    for k,n in [('intrinsics',3),('extrinsics',4)]:
        if hs[k]['shape'] not in [[n,n],[1,n,n],[T,n,n]] or hs[k]['dtype'][-2:] not in ['f4','f8']:
            raise ValueError('f32/f64 camera layouts only')
    if any(hs[k]['dtype'][-2:] not in ['f4','f8'] for k in ['trajs_2d','trajs_3d']):
        raise ValueError('f32/f64 trajectories only')
    if any(hs[k]['dtype'][-2:] not in ['b1','u1','i1'] for k in ['visibs','valids']):
        raise ValueError('byte flags only')
    out = {}; ledger = progress if progress is not None else {}
    for key in required:
        budget.open_stream()
        with z.open(members[key+'.npy']) as stream:
            current = header(stream,budget)
            if current != hs[key]: raise ValueError('header changed on reopen')
            n = 3 if key == 'intrinsics' else 4
            constant = key in ['intrinsics','extrinsics'] and current['shape'] in [[n,n],[1,n,n]]
            before = budget.returned_bytes
            ledger[key] = {'status':'attempting','numeric_bytes_returned':0}
            try:
                out[key],detail = read_prefix(stream,current,budget,constant)
                ledger[key].update(detail,status='completed')
            finally:
                ledger[key]['numeric_bytes_returned'] = budget.returned_bytes-before
    return hs,out,ledger


def selected_points(data):
    xy,world,vis = (data[k] for k in ['trajs_2d','trajs_3d','visibs'])
    selected = []
    for p in range(xy.shape[1]):
        flag = vis.at(0,p)
        if flag not in (0,1,2): raise ValueError('visibility enum')
        values = [xy.at(0,p,j) for j in range(2)] + [world.at(0,p,j) for j in range(3)]
        if flag == 1 and all(math.isfinite(x) for x in values): selected.append(p)
        if len(selected) == 2: break
    return selected


def _finite(values):
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('finite real arithmetic required')


def project_v2(world, K, RT, camera_dtype):
    """Pinned upstream RT->z normalization->K; no inverse/axis/transpose search."""
    if camera_dtype not in ('f4','f8'):
        raise ValueError('f32/f64 camera storage only')
    if len(world)!=3 or len(K)!=3 or any(len(r)!=3 for r in K) or len(RT)!=4 or any(len(r)!=4 for r in RT):
        raise ValueError('geometry dimensions')
    _finite(list(world)+sum(K,[])+sum(RT,[]))
    if list(K[2]) != [0,0,1] or list(RT[3]) != [0,0,0,1] or K[0][0]<=0 or K[1][1]<=0:
        raise ValueError('exact homogeneous rows / positive focal')
    eps = 2**-23 if camera_dtype == 'f4' else 2**-52
    tol = max(1e-8,8*eps)
    R = [r[:3] for r in RT[:3]]
    rr = [sum(R[i][k]*R[j][k] for k in range(3))-(i==j) for i in range(3) for j in range(3)]
    rt = [sum(R[k][i]*R[k][j] for k in range(3))-(i==j) for i in range(3) for j in range(3)]
    det = sum(R[0][j]*(R[1][(j+1)%3]*R[2][(j+2)%3]-R[1][(j+2)%3]*R[2][(j+1)%3]) for j in range(3))
    _finite(rr+rt+[det])
    if max(map(abs,rr+rt))>tol or abs(det-1)>tol:
        raise ValueError('storage-aware proper rotation gate')
    cam = [sum(row[j]*world[j] for j in range(3))+row[3] for row in RT[:3]]
    _finite(cam)
    if cam[2] <= 0: raise ValueError('positive camera depth')
    normal = [x/cam[2] for x in cam]; _finite(normal)
    pixel = [sum(K[i][j]*normal[j] for j in range(3)) for i in range(3)]; _finite(pixel)
    return pixel[:2], {'rotation_tolerance':tol,'RRt_max':max(map(abs,rr)),'RtR_max':max(map(abs,rt)),'det_minus_one_abs':abs(det-1)}


def check_geometry(hs, data, result):
    """Mutate supplied receipt per attempt so a later fixed stop preserves prior checks."""
    ids = selected_points(data); result['selected_original_point_indices'] = ids
    result['point_checks'] = []
    def matrix(key,t,n):
        a = data[key]
        return [[a.at(i,j) if len(a.shape)==2 else a.at(0 if a.shape[0]==1 else t,i,j) for j in range(n)] for i in range(n)]
    for t in range(3):
        for p in ids:
            record = {'frame':t,'point_index':p,'status':'attempted'}; result['point_checks'].append(record)
            try:
                flag = data['valids'].at(t,p); visibility = data['visibs'].at(t,p)
                record.update(validity=flag,visibility=visibility)
                if flag != 1 or visibility not in (0,1,2): raise ValueError('invalid selected GT/visibility flag')
                point = [data['trajs_3d'].at(t,p,j) for j in range(3)]
                truth = [data['trajs_2d'].at(t,p,j) for j in range(2)]; _finite(truth)
                # Conservatively use the less precise of K and RT storage.
                dtype = 'f4' if any(hs[k]['dtype'].endswith('f4') for k in ['intrinsics','extrinsics']) else 'f8'
                pixel,gate = project_v2(point,matrix('intrinsics',t,3),matrix('extrinsics',t,4),dtype)
                error = math.hypot(*(a-b for a,b in zip(pixel,truth))); _finite([error])
                record.update(error_original_px=error,rotation=gate)
                if error > .5: raise ValueError('fixed 0.5 original-pixel gate')
                record['status'] = 'passed'
            except (ValueError,OverflowError) as exc:
                record.update(status='failed',reason=str(exc)); raise
    result['status'] = 'completed_small_development_check'
    errors = [r['error_original_px'] for r in result['point_checks']]
    result.update(max_error_original_px=max(errors) if errors else None,
                  rms_error_original_px=math.hypot(*errors)/math.sqrt(len(errors)) if errors else None)
    return result


def audit_budget():
    """Carry forward BOTH old audit and header diagnosis, not a silent reset."""
    b = ReadBudget()
    b.returned_bytes = 1152
    b.stream_reserve_bytes = 589824
    return b


def metadata_json(raw):
    """Parse <=16 KiB pure JSON. Publish key coverage/hashes, never raw metadata."""
    import json
    if len(raw)>16384: raise ValueError('scene JSON cap')
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result: raise ValueError('duplicate metadata JSON key')
            result[k] = v
        return result
    def reject(s): raise ValueError('nonfinite JSON literal')
    data = json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
    if not isinstance(data,dict): raise ValueError('metadata JSON object')
    count = 0; fields = []
    def walk(value,path,depth):
        nonlocal count
        count+=1
        if count>2048 or depth>12: raise ValueError('metadata complexity cap')
        if isinstance(value,dict):
            for key,item in value.items():
                if len(key)>256 or '\0' in key: raise ValueError('metadata key cap')
                fields.append(path+[key]); walk(item,path+[key],depth+1)
        elif isinstance(value,list):
            for item in value: walk(item,path+['[]'],depth+1)
        elif isinstance(value,float) and not math.isfinite(value): raise ValueError('nonfinite metadata number')
    walk(data,[],0)
    # Hash paths rather than publish potentially sensitive metadata keys/values.
    coverage = {}
    for relation in ['asset','motion','session','story','view','camera']:
        covered = [p for p in fields if any(relation in s.lower() for s in p)]
        coverage[relation] = {'matching_key_count':len(covered),
            'key_path_sha256':[hashlib.sha256('/'.join(p).encode()).hexdigest() for p in covered],
            'relationship':'UNKNOWN; key presence alone does not verify identities or cross-scene independence'}
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'field_paths_inspected':len(fields),'coverage':coverage}


def info_npz_headers(raw, budget):
    """Small info container: headers only. Object payloads remain uninterpreted."""
    import io
    if len(raw)>16384: raise ValueError('info NPZ container cap')
    result=[]; seen=set()
    zip_envelope(io.BytesIO(raw),budget,32)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos=z.infolist()
        if not 0<len(infos)<=32: raise ValueError('info member cap')
        for info in infos:
            if '\0' in info.orig_filename: raise ValueError('NUL info member')
            name=safe_member(info.filename)
            if '/' in name or not name.endswith('.npy') or name in seen or info.flag_bits&1 or info.compress_type not in (0,8) or stat.S_IFMT(info.external_attr>>16)not in(0,stat.S_IFREG):
                raise ValueError('info NPZ unsafe member')
            seen.add(name);budget.open_stream()
            with z.open(info) as stream:
                h=header(stream,budget,metadata_only=True)
            result.append({'name':name,'header':h,'numeric_payload_interpreted':False,'object_payload_decoded':False})
    return result


def zip_envelope(path_or_stream, budget, max_members=64):
    """Bound central-directory allocation before ZipFile parses; no ZIP64/comments."""
    own = not hasattr(path_or_stream,'read')
    f = open(path_or_stream,'rb') if own else path_or_stream
    try:
        f.seek(0,2); size=f.tell()
        if size<22: raise ValueError('truncated ZIP envelope')
        f.seek(-22,2); raw=budget.read(f,22)
        sig,disk,cd_disk,n_disk,n,size_cd,off,comment=struct.unpack('<4s4H2IH',raw)
        if sig!=b'PK\x05\x06' or disk or cd_disk or n_disk!=n or not 0<n<=max_members or size_cd>65536 or off+size_cd!=size-22 or comment:
            raise ValueError('bounded non-ZIP64 single-disk uncommented envelope')
        return {'members':n,'central_directory_bytes':size_cd}
    finally:
        if own:f.close()
        else:f.seek(0)
