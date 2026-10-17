"""Resource-only 0003 audit primitives: bounded NPY prefixes, never whole arrays."""
import ast
import hashlib
import math
import re
import struct
import sys

MAX_READ=64*1024**2
class ReadBudget:
    def __init__(self,limit=MAX_READ):
        if type(limit)is not int or not 0<limit<=MAX_READ:raise ValueError('bounded positive byte budget required')
        self.limit=limit;self.returned_bytes=0;self.stream_reserve_bytes=0
    @property
    def charged_bytes(self):return self.returned_bytes+self.stream_reserve_bytes
    def open_stream(self):
        # ZipExtFile may internally buffer >=4 KiB beyond a tiny header read;
        # charge a conservative 64 KiB per open in addition to every returned byte.
        if self.charged_bytes+65536>self.limit:raise ValueError('cumulative stream reserve cap')
        self.stream_reserve_bytes+=65536
    def read(self,stream,n):
        if type(n)is not int or n<0 or self.charged_bytes+n>self.limit:raise ValueError('cumulative uncompressed read cap')
        out=bytearray()
        while len(out)<n:
            part=stream.read(min(1024**2,n-len(out)));self.returned_bytes+=len(part)
            if not part:raise ValueError('truncated bounded stream')
            out.extend(part)
        return bytes(out)

def npy_header(stream,budget):
    start=budget.returned_bytes
    if budget.read(stream,6)!=b'\x93NUMPY':raise ValueError('NPY magic')
    version=tuple(budget.read(stream,2))
    if version not in[(1,0),(2,0),(3,0)]:raise ValueError('NPY version')
    n=struct.unpack('<H'if version==(1,0)else'<I',budget.read(stream,2 if version==(1,0)else 4))[0]
    if not 0<n<=65536:raise ValueError('NPY header cap')
    raw=budget.read(stream,n)
    try:header=ast.literal_eval(raw.decode('utf-8'if version==(3,0)else'latin1').strip())
    except Exception as exc:raise ValueError('NPY literal header')from exc
    if not isinstance(header,dict)or set(header)!={'descr','fortran_order','shape'}or header['fortran_order']is not False:raise ValueError('C order only')
    shape=header['shape'];descr=header['descr']
    if not isinstance(shape,tuple)or not 1<=len(shape)<=4 or any(type(x)is not int or not 0<x<=10000000 for x in shape):raise ValueError('bounded nonempty shape')
    if not isinstance(descr,str):raise ValueError('structured/object dtype forbidden')
    m=re.fullmatch(r'([<>|=])([fiub])([1248])',descr)
    if not m:raise ValueError('real numeric dtype only; pickle/object/complex forbidden')
    endian,kind,size=m.group(1),m.group(2),int(m.group(3))
    types={('f',2):'e',('f',4):'f',('f',8):'d',('i',1):'b',('i',2):'h',('i',4):'i',('i',8):'q',('u',1):'B',('u',2):'H',('u',4):'I',('u',8):'Q',('b',1):'?'}
    if(kind,size)not in types or endian=='|'and size!=1:raise ValueError('dtype layout')
    fmt=('<'if endian=='|'else'<'if endian=='='and sys.byteorder=='little'else'>'if endian=='='else endian)+types[kind,size]
    return {'shape':list(shape),'dtype':descr,'itemsize':size,'format':fmt,'header_bytes':budget.returned_bytes-start,'data_bytes':math.prod(shape)*size,'allow_pickle':False,'fortran_order':False}

def prefix(stream,header,budget,frames=3):
    if frames!=3:raise ValueError('exactly first three frames')
    shape=header['shape']
    if shape[0]<3:raise ValueError('short trajectory')
    n=3*math.prod(shape[1:])*header['itemsize']
    raw=budget.read(stream,n)
    return raw,hashlib.sha256(raw).hexdigest()

def scalar(raw,header,*indices):
    shape=header['shape']
    if len(indices)!=len(shape)or any(type(x)is not int or not 0<=x<s for x,s in zip(indices,shape)):raise ValueError('scalar indices')
    i=0
    for x,s in zip(indices,shape):i=i*s+x
    offset=i*header['itemsize']
    if offset+header['itemsize']>len(raw):raise ValueError('outside saved prefix')
    return struct.unpack_from(header['format'],raw,offset)[0]

def first_frame_points(xy_raw,xy_header,xyz_raw,xyz_header,vis_raw,vis_header):
    if xy_header['shape'][2:]!=[2]or xyz_header['shape'][2:]!=[3]or xy_header['shape'][:2]!=xyz_header['shape'][:2]or vis_header['shape']!=xy_header['shape'][:2]:raise ValueError('aligned time-major layouts required')
    out=[]
    for p in range(xy_header['shape'][1]):
        v=scalar(vis_raw,vis_header,0,p)
        if v not in(0,1,2):raise ValueError('unknown visibility enum')
        values=[scalar(xy_raw,xy_header,0,p,j)for j in range(2)]+[scalar(xyz_raw,xyz_header,0,p,j)for j in range(3)]
        if v==1 and all(math.isfinite(x)for x in values):out.append(p)
        if len(out)==2:break
    return out
