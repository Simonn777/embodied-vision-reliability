"""Fixed sparse development annotation check, not rigid-camera validation.

Only the standalone synthetic tests may call this before a published receipt.
No model, camera repair, search, compatibility witness or external source imports.
"""
import hashlib,itertools,math,stat,struct,zipfile
from evr.bounded_point_audit import ReadBudget
from evr.motion_design import safe_member
from evr.point_geometry_v2 import header,offset,zip_envelope
from evr.rotation_diagnostic_v1 import double_statistics,TAU

KEYS=('trajs_2d','trajs_3d','visibs','valids','intrinsics','extrinsics')
SCHEMA={'trajs_2d':([849,11721,2],'<f4','C'),'trajs_3d':([849,11721,3],'<f4','C'),
 'visibs':([849,11721],'|b1','F'),'valids':([849,11721],'|b1','F'),
 'intrinsics':([849,3,3],'<f4','C'),'extrinsics':([849,4,4],'<f4','C')}
SPANS={'trajs_2d':187552,'trajs_3d':281328,'visibs':852,'valids':852,'intrinsics':108,'extrinsics':192}


def continued_budget():
 b=ReadBudget();b.returned_bytes=20614916;b.stream_reserve_bytes=2295240;b.central_directory_reserved_bytes=1480
 return b


def requests(key):
 if key in ('trajs_2d','trajs_3d'):
  return list(itertools.product(range(3),(0,1),range(2 if key=='trajs_2d'else 3)))
 if key in ('visibs','valids'):return list(itertools.product(range(3),(0,1)))
 n=3 if key=='intrinsics'else 4
 return list(itertools.product(range(3),range(n),range(n)))


def sparse_values(stream,h,indices,budget):
 """Charge maximum physical span including all gaps; interpret only requested scalars."""
 if not indices or len(indices)>117 or len(set(indices))!=len(indices):raise ValueError('bounded unique scalar requests')
 req=sorted((offset(h,ix),ix)for ix in indices);span=req[-1][0]+h['itemsize']
 if budget.charged_bytes+span>budget.limit:raise ValueError('sparse span cap BEFORE payload')
 values={};cursor=0;j=0;sha=hashlib.sha256()
 while cursor<span:
  raw=budget.read(stream,min(65536,span-cursor));sha.update(raw);end=cursor+len(raw)
  while j<len(req)and req[j][0]<end:
   at,ix=req[j];values[ix]=struct.unpack_from(h['format'],raw,at-cursor)[0];j+=1
  cursor=end
 return values,{'numeric_span_bytes':span,'retained_scalars':len(values),'retained_bytes':len(values)*h['itemsize'],
  'discarded_bytes':span-len(values)*h['itemsize'],'span_sha256':sha.hexdigest()}


def read_inputs(path,budget,progress,checkpoint):
 progress['zip_envelope']=zip_envelope(path,budget);checkpoint()
 with zipfile.ZipFile(path)as z:
  members={}
  for info in z.infolist():
   if '\0'in info.orig_filename:raise ValueError('NUL ZIP name')
   name=safe_member(info.filename)
   if name in members or '/'in name or info.flag_bits&1 or info.compress_type not in (0,8)or stat.S_IFMT(info.external_attr>>16)not in (0,stat.S_IFREG):raise ValueError('safe unique flat regular unencrypted members')
   members[name]=info
  if set(members)!={k+'.npy'for k in KEYS}:raise ValueError('only six fixed members')
  data={};progress['arrays']={}
  for key in KEYS:
   item={'state':'HEADER_STARTING','counts_complete':False};progress['arrays'][key]=item;checkpoint()
   before=budget.returned_bytes;numeric_start=None
   try:
    budget.open_stream()
    with z.open(members[key+'.npy'])as stream:
     h=header(stream,budget);shape,dtype,order=SCHEMA[key]
     if (h['shape'],h['dtype'],h['order'])!=(shape,dtype,order)or h['header_bytes']!=128 or members[key+'.npy'].file_size!=h['header_bytes']+h['data_bytes']:raise ValueError('fixed observed schema/length')
     item.update(state='NUMERIC_ACCESS_STARTING',header_sha256=h['header_sha256']);checkpoint()
     numeric_start=budget.returned_bytes
     data[key],account=sparse_values(stream,h,requests(key),budget)
     if account['numeric_span_bytes']!=SPANS[key]:raise ValueError('fixed sparse span')
     item.update(account,state='COMPLETE')
   except Exception:
    item['state']='FAILED';raise
   finally:
    item.update(returned_bytes=budget.returned_bytes-before,returned_numeric_bytes=budget.returned_bytes-numeric_start if numeric_start is not None else 0,counts_complete=True);checkpoint()

 return data


def finite(values):
 if any(type(x)not in (int,float)or not math.isfinite(x)for x in values):raise ValueError('finite required arithmetic')


def camera(data,key,t,n):return [[data[key][(t,i,j)]for j in range(n)]for i in range(n)]


def camera_bytes(matrix):return b''.join(struct.pack('<f',v)for row in matrix for v in row)


def gate(K,RT):
 if len(K)!=3 or any(len(r)!=3 for r in K)or len(RT)!=4 or any(len(r)!=4 for r in RT):raise ValueError('camera shape')
 finite(sum(K,[])+sum(RT,[]))
 if K[2]!=[0,0,1]or RT[3]!=[0,0,0,1]or K[0][0]<=0 or K[1][1]<=0:raise ValueError('exact homogeneous rows and positive focal')
 rr,rt,det=double_statistics([row[:3]for row in RT[:3]])
 a=max(map(abs,sum(rr,[])));b=max(map(abs,sum(rt,[])));d=abs(det-1)
 return {'tau':TAU,'RRt_max':a,'RtR_max':b,'det_minus_one_abs':d,'proper_rotation_pass':max(a,b,d)<=TAU}


def project_descriptive(world,K,RT):
 if len(world)!=3:raise ValueError('world shape')
 finite(world)
 c=[sum(row[j]*world[j]for j in range(3))+row[3]for row in RT[:3]];finite(c)
 if c[2]<=0:raise ValueError('positive depth required')
 normal=[v/c[2]for v in c];finite(normal)
 pixel=[sum(K[i][j]*normal[j]for j in range(3))for i in range(3)];finite(pixel)
 return pixel[:2]  # only internal; never publish original/projected coordinates


def blank_checks():
 return [{'frame':t,'point_id':p,'state':'UNATTEMPTED','projection_consistent':None,
  'valid_visible_annotation_support':False,'rigid_qualified_support':False}for t in range(3)for p in (0,1)]


def summarize(result):
 rows=result['checks'];errors=[r['error_original_px']for r in rows if 'error_original_px'in r]
 maximum=max(errors)if errors else None
 rms=maximum*math.sqrt(sum((e/maximum)**2 for e in errors)/len(errors))if maximum else maximum
 result.update(planned=6,attempted=sum(r['state']!='UNATTEMPTED'for r in rows),unattempted=sum(r['state']=='UNATTEMPTED'for r in rows),
  finite_error_count=len(errors),max_error_original_px=maximum,rms_error_original_px=rms,
  projection_consistent_count=sum(r['projection_consistent']is True for r in rows),
  valid_visible_annotation_support_count=sum(r['valid_visible_annotation_support']for r in rows),
  rigid_qualified_support_count=sum(r['rigid_qualified_support']for r in rows))


def check_points(data,expected_bytes,old_gate,result,checkpoint=lambda:None):
 result.setdefault('checks',blank_checks());result['frame_camera_gates']={};result['outcome']='STOPPED'
 try:
  for row in result['checks']:
   t,p=row['frame'],row['point_id'];row['state']='ATTEMPTING';checkpoint()
   try:
    K=camera(data,'intrinsics',t,3);RT=camera(data,'extrinsics',t,4)
    g=gate(K,RT)
    if t==0:
     if camera_bytes(K)!=expected_bytes['K']or camera_bytes(RT)!=expected_bytes['RT']:raise ValueError('frame0 saved camera byte identity')
     if g['proper_rotation_pass']or g!=old_gate:raise ValueError('frame0 original failure/statistics changed')
    result['frame_camera_gates'][str(t)]=g
    validity=data['valids'][(t,p)];visibility=data['visibs'][(t,p)]
    row.update(validity_supported=validity==1,visibility_visible=visibility==1)
    if type(validity)is not int or validity!=1 or type(visibility)is not int or visibility not in (0,1,2):raise ValueError('required validity or visibility enum')
    truth=[data['trajs_2d'][(t,p,c)]for c in range(2)];finite(truth)
    xy=project_descriptive([data['trajs_3d'][(t,p,c)]for c in range(3)],K,RT)
    dx,dy=(xy[i]-truth[i]for i in range(2));error=math.hypot(dx,dy);finite([dx,dy,error])
    passed=error<=.5
    row.update(dx_original_px=dx,dy_original_px=dy,error_original_px=error,positive_depth=True,
     projection_consistent=passed,valid_visible_annotation_support=passed and visibility==1,
     rigid_qualified_support=passed and visibility==1 and g['proper_rotation_pass'])
    if not passed:raise ValueError('fixed 0.5 original-pixel criterion')
    row['state']='COMPLETE'
   except (ValueError,OverflowError,ZeroDivisionError)as exc:
    row.update(state='FAILED',reason=str(exc));raise
   finally:summarize(result);checkpoint()
  result['outcome']='COMPLETED_DESCRIPTIVE_WITH_ORIGINAL_RIGID_FAILURE'
  return result
 finally:summarize(result)
