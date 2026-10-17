"""ONE resource-frozen cached PointOdyssey development audit. No network/model."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
from evr.bounded_point_audit import ReadBudget,npy_header,prefix,scalar,first_frame_points
from evr.motion_design import safe_member,project,unique_json,BRANCH
from evr.motion_phases import verify_timing

B=Path('results/0003-resource-audit-20261017');C=Path('../bootstapir-cache/0003-pointodyssey');RECEIPT='protocols/0003-resource-addendum-freeze-v1.json'
INPUTS=frozenset(['protocols/0003-resource-addendum-v1.md','configs/0003-resource-addendum-v1.json','src/evr/bounded_point_audit.py','scripts/audit_point_sample.py','tests/test_bounded_point_audit.py','protocols/0003-design-freeze-v1.json','protocols/0003-timing-addendum-freeze-v1.json','results/0003-network-followup-20261017/pinned-source-http.json','results/0003-network-followup-20261017/geometry-source-http.json','results/0003-network-followup-20261017/export-source-http.json'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):
    p=subprocess.run(['git',*args],capture_output=True)
    if p.returncode:raise ValueError('resource Git binding unavailable')
    return p.stdout

def frozen():
    verify_timing(Path.cwd());raw=Path(RECEIPT).read_bytes();r=unique_json(raw)
    if set(r)!={'schema','scope','input_commit','inputs_sha256'}or r['schema']!=1 or r['scope']!='RESOURCE_ONLY_CACHED_DEVELOPMENT_AUDIT'or set(r['inputs_sha256'])!=INPUTS:raise ValueError('canonical complete resource receipt required')
    commit=r['input_commit']
    if not isinstance(commit,str)or len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('resource input commit')
    for ref in ['HEAD',BRANCH]:
        if git('show',ref+':'+RECEIPT)!=raw:raise ValueError('unpublished or changed resource receipt')
        git('merge-base','--is-ancestor',commit,ref)
    if git('show',':'+RECEIPT)!=raw or Path(RECEIPT).is_symlink():raise ValueError('staged/symlink receipt')
    if RECEIPT in git('ls-tree','-r','--name-only',commit).decode().splitlines():raise ValueError('self-referencing inputs')
    for path,digest in r['inputs_sha256'].items():
        p=Path(path)
        if p.is_symlink()or not p.resolve().is_relative_to(Path.cwd())or any(q.is_symlink()for q in p.parents):raise ValueError('regular input path required')
        for data in [p.read_bytes(),git('show',commit+':'+path),git('show','HEAD:'+path),git('show',':'+path),git('show',BRANCH+':'+path)]:
            if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('resource input binding')
    return r

def main():
    receipt=frozen();archive=C/'sample-v1.2.tar.gz';work=C/'resource-audit-v1';work.mkdir(mode=0o700,exist_ok=False)
    with(B/'audit-reservation.json').open('x')as f:json.dump({'one_cached_audit_only':True,'input_commit':receipt['input_commit']},f)
    r={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_commit':git('rev-parse','HEAD').decode().strip(),'resource_input_commit':receipt['input_commit'],'receipt_sha256':sha(RECEIPT),'numeric_GT_values_accessed':False,'numeric_data_bytes_returned':0,'RGB_frames_decoded':0,'model_inferences':0,'archive_GETs':0,'inventory_complete':False,'annotation_saved_count':0,'read_budget_limit':64*1024**2,'point_checks':[],'quarantine':'entire sample and unknown linked family permanently audit/development; no independent cohort claim'}
    t=time.perf_counter();budget=ReadBudget();h=hashlib.sha256()
    def save():
        r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t,uncompressed_bytes_returned=budget.returned_bytes,stream_buffer_reserve_bytes=budget.stream_reserve_bytes,total_read_budget_charged=budget.charged_bytes)
        (B/'audit.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    try:
        with archive.open('rb')as f:
            while part:=f.read(2**20):h.update(part)
        if archive.stat().st_size!=3324284510 or h.hexdigest()!='436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f':raise ValueError('cached archive identity')
        r['archive_sha256']=h.hexdigest();r['archive_bytes']=archive.stat().st_size
        seen=set();annotations=[];total=0;count=0;invhash=hashlib.sha256();chunk=[];chunk_index=0;scenes={};r['previous_header_2756_new_observation']=None
        def flush():
            nonlocal chunk_index,chunk
            if not chunk:return
            with(B/f'inventory-{chunk_index:03d}.csv').open('x',newline='')as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(['ordinal','canonical_path','type','declared_bytes']);w.writerows(chunk)
            chunk=[];chunk_index+=1
        with tarfile.open(archive,'r|gz')as tar:
            for m in tar:
                kind='file'if m.isfile()else'directory'if m.isdir()else'other';name=safe_member(m.name,kind)
                if name in seen:raise ValueError('duplicate canonical TAR path')
                seen.add(name);count+=1;total+=m.size
                if count>200000 or total>20*1024**3:raise ValueError('unchanged TAR count/total-size cap')
                row=[count,name,kind,m.size];invhash.update((json.dumps(row,separators=(',',':'))+'\n').encode());chunk.append(row)
                if len(chunk)>=2048:flush()
                if count==2756:r['previous_header_2756_new_observation']={'path':name,'declared_bytes':m.size,'observation':'new complete-resource-audit metadata, not recovered original log'}
                parent=str(Path(name).parent);s=scenes.setdefault(parent,{'files':0,'declared_bytes':0})
                if kind=='file':s['files']+=1;s['declared_bytes']+=m.size
                if kind=='file'and Path(name).name in['anno.npz','annot.npz','annotations.npz']:annotations.append({'path':name,'declared_bytes':m.size,'scene_parent':parent})
        flush();r.update(inventory_complete=True,inventory_members=count,inventory_declared_bytes=total,inventory_sha256=invhash.hexdigest(),inventory_csv_files=chunk_index,annotation_candidates=annotations)
        (B/'scene-metadata.json').write_text(json.dumps({'parents':scenes,'independence':'unknown; path names are not verified scene/story/session/asset families'},indent=2)+'\n');save()
        if not annotations:raise ValueError('no canonical annotation candidate')
        annotations.sort(key=lambda x:(x['scene_parent'],x['path']));selected=annotations[0]
        if sum(x['scene_parent']==selected['scene_parent']for x in annotations)!=1:raise ValueError('ambiguous first scene annotation')
        r['selected_annotation']=selected;r['free_disk_before_bytes']=shutil.disk_usage(C).free
        if selected['declared_bytes']>2*1024**3 or r['free_disk_before_bytes']<selected['declared_bytes']+512*1024**2:raise ValueError('2 GiB disk cap or free-disk reserve')
        annotation=work/'selected-annotation.npz';digest=hashlib.sha256();copied=0
        with tarfile.open(archive,'r|gz')as tar:
            for m in tar:
                if m.name.rstrip('/')!=selected['path']:continue
                if not m.isfile()or m.size!=selected['declared_bytes']:raise ValueError('selected regular identity')
                fd=os.open(annotation,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
                with os.fdopen(fd,'wb')as out,tar.extractfile(m)as source:
                    while part:=source.read(2**20):
                        copied+=len(part)
                        if copied>selected['declared_bytes']or copied>2*1024**3:raise ValueError('annotation copy cap')
                        digest.update(part);out.write(part)
                break
        if copied!=selected['declared_bytes']:raise ValueError('annotation copy incomplete')
        r.update(annotation_saved_count=1,annotation_bytes=copied,annotation_sha256=digest.hexdigest());save()
        with zipfile.ZipFile(annotation)as z:
            infos=z.infolist();headers={};bykey={}
            if not 0<len(infos)<=64:raise ValueError('NPZ member count')
            for info in infos:
                name=safe_member(info.filename)
                if '/'in name or not name.endswith('.npy')or info.flag_bits&1 or stat.S_IFMT(info.external_attr>>16)not in(0,stat.S_IFREG)or info.compress_type not in(0,8):raise ValueError('NPZ regular unencrypted numeric members only')
                key=name[:-4]
                if key in headers:raise ValueError('duplicate NPZ key')
                budget.open_stream()
                with z.open(info)as f:header=npy_header(f,budget)
                if info.file_size!=header['header_bytes']+header['data_bytes']:raise ValueError('NPY shape/size mismatch')
                headers[key]=header;bykey[key]=info
            r['annotation_schema']=headers;save()
            required=['trajs_2d','trajs_3d','intrinsics','extrinsics']
            if any(x not in headers for x in required):raise ValueError('missing exact geometry keys')
            aliases=[k for k in ['visibs','visibilities','visibility']if k in headers]
            if len(aliases)!=1:raise ValueError('one supported visibility field required')
            vis=aliases[0];valid=[k for k in ['valids','validities']if k in headers]
            if len(valid)>1:raise ValueError('ambiguous validity field')
            T,N,*rest=headers['trajs_2d']['shape']
            if rest!=[2]or T<3 or headers['trajs_3d']['shape']!=[T,N,3]or headers[vis]['shape']!=[T,N]or valid and headers[valid[0]]['shape']!=[T,N]:raise ValueError('unsupported aligned time-major trajectory layout')
            for key,size in [('intrinsics',3),('extrinsics',4)]:
                if headers[key]['shape']not in[[size,size],[1,size,size],[T,size,size]]:raise ValueError('unsupported camera layout')
            data={}
            for key in required+[vis]+valid:
                budget.open_stream()
                with z.open(bykey[key])as f:
                    repeated=npy_header(f,budget)
                    if repeated!=headers[key]:raise ValueError('NPY header changed')
                    is_constant=key in['intrinsics','extrinsics']and repeated['shape']in[[3,3],[4,4],[1,3,3],[1,4,4]]
                    amount=repeated['data_bytes']if is_constant else 3*math.prod(repeated['shape'][1:])*repeated['itemsize']
                    before=budget.returned_bytes
                    try:raw=budget.read(f,amount)
                    finally:
                        r['numeric_data_bytes_returned']+=budget.returned_bytes-before;r['numeric_GT_values_accessed']=r['numeric_data_bytes_returned']>0;save()
                    data[key]=raw;headers[key]['prefix_sha256']=hashlib.sha256(raw).hexdigest();headers[key]['prefix_bytes']=len(raw)
            points=first_frame_points(data['trajs_2d'],headers['trajs_2d'],data['trajs_3d'],headers['trajs_3d'],data[vis],headers[vis]);r['selected_original_point_indices']=points;r['point_selection']='frame0 visibility==1 and finite GT2D/GT3D only; no future filtering or replacement'
            def camera(key,t,size):
                shape=headers[key]['shape'];indices=lambda i,j:(i,j)if len(shape)==2 else(0 if shape[0]==1 else t,i,j)
                return[[scalar(data[key],headers[key],*indices(i,j))for j in range(size)]for i in range(size)]
            errors=[]
            for t0 in range(3):
                K=camera('intrinsics',t0,3);RT=camera('extrinsics',t0,4)
                for p in points:
                    record={'frame':t0,'original_point_index':p,'status':'invalid'}
                    try:
                        world=[scalar(data['trajs_3d'],headers['trajs_3d'],t0,p,j)for j in range(3)];truth=[scalar(data['trajs_2d'],headers['trajs_2d'],t0,p,j)for j in range(2)]
                        if not all(math.isfinite(v)for v in world+truth):raise ValueError('nonfinite selected GT')
                        if valid:record['provided_validity']=scalar(data[valid[0]],headers[valid[0]],t0,p)
                        record['source_visibility']=scalar(data[vis],headers[vis],t0,p)
                        record['rotation_orthogonality_max_absdiff']=max(abs(sum(RT[i][k]*RT[j][k]for k in range(3))-(i==j))for i in range(3)for j in range(3))
                        projected=project(world,K,RT);error=math.hypot(*(a-b for a,b in zip(projected,truth)));record.update(status='valid',reprojection_error_original_px=error)
                        if error>.5:raise ValueError('fixed original-pixel reprojection tolerance')
                        errors.append(error)
                    except Exception as exc:
                        record.update(status='invalid',error_type=type(exc).__name__,reason=str(exc));r['point_checks'].append(record);save();raise ValueError('fixed-stop selected geometry failure')from exc
                    r['point_checks'].append(record)
            r.update(status='completed_small_development_diagnostic',valid_checks=len(errors),invalid_checks=0,max_reprojection_error_original_px=max(errors)if errors else None,RMS_reprojection_error_original_px=math.sqrt(sum(x*x for x in errors)/len(errors))if errors else None,world_units='not verified',FPS='not verified',scope='up to six fixed point/frame checks only; not whole dataset GT gate or model result')
    except Exception as exc:
        r.update(status='stopped',error_type=type(exc).__name__,reason=str(exc).replace(str(Path.cwd()),'<repo>'),valid_checks=sum(x['status']=='valid'for x in r['point_checks']),invalid_checks=sum(x['status']=='invalid'for x in r['point_checks']))
    finally:
        computed=[x['reprojection_error_original_px']for x in r['point_checks']if 'reprojection_error_original_px'in x and math.isfinite(x['reprojection_error_original_px'])]
        r.update(projection_checks_computed=len(computed),max_reprojection_error_original_px=max(computed)if computed else None,RMS_reprojection_error_original_px=math.sqrt(sum(x*x for x in computed)/len(computed))if computed else None,point_checks_attempted=len(r['point_checks']),world_units='not verified',FPS='not verified')
        save()
    print(json.dumps({k:v for k,v in r.items()if k not in['annotation_schema','annotation_candidates']},indent=2));raise SystemExit(0 if r['status']=='completed_small_development_diagnostic'else 1)
if __name__=='__main__':main()
