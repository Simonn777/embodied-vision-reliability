"""One bounded ZIP central-directory qualification; never inflates payloads."""
import datetime,hashlib,json,os,re,signal,struct,sys,time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler,Request,build_opener

HERE=Path(__file__).resolve().parent
URL='https://huggingface.co/datasets/vbalnt/hpatches/resolve/594f211e80b75081ad84a396c93960c8653235f8/hpatches-sequences-release.zip'
TOTAL=1279545572;TAIL=65557;CAP=1024**2
WHOLE_SHA='99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670'
LEDGER={'schema':'0009-range-metadata-ledger-v1','starting_asset_ceiling_bytes':7532690072,'body_hard_cap_bytes':CAP,'maximum_range_get_operations':2,'range_get_operations':[],'actual_response_body_bytes_read':0,'whole_archive_downloaded':False,'whole_archive_sha256_verified':False,'declared_whole_archive_bytes':TOTAL,'declared_whole_archive_sha256':WHOLE_SHA,'redirect_accounting':'HTTPS redirect hops are recorded separately within each range GET operation; they are not retries. A second range operation reuses the first final URL privately.'}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(name,value):
    p=HERE/name;tmp=HERE/(name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def checkpoint():
    LEDGER['updated_utc']=utc();LEDGER['ending_asset_ceiling_bytes']=LEDGER['starting_asset_ceiling_bytes']+LEDGER['actual_response_body_bytes_read'];atomic('ledger.json',LEDGER)
class Redirects(HTTPRedirectHandler):
    def __init__(self,row):super().__init__();self.row=row
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urlsplit(newurl).scheme.lower()!='https':raise ValueError('non-HTTPS redirect rejected')
        self.row['https_redirect_statuses'].append(code)
        if len(self.row['https_redirect_statuses'])>5:raise ValueError('redirect hop bound exceeded')
        checkpoint()
        return super().redirect_request(req,fp,code,msg,headers,newurl)
def ranged_get(url,start,end,label):
    if len(LEDGER['range_get_operations'])>=2:raise ValueError('range GET budget exhausted')
    expected=end-start+1
    if not 0<=start<=end<TOTAL or LEDGER['actual_response_body_bytes_read']+expected>CAP:raise ValueError('range/body budget rejected before request')
    row={'operation':len(LEDGER['range_get_operations'])+1,'label':label,'range_start':start,'range_end':end,'expected_body_bytes':expected,'utc_started':utc(),'https_redirect_statuses':[],'actual_body_bytes_read':0,'status':'REQUEST_STARTING'}
    LEDGER['range_get_operations'].append(row);checkpoint();response=None;data=bytearray()
    try:
        request=Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'bounded-research-directory-audit/1.0'},method='GET')
        response=build_opener(Redirects(row)).open(request,timeout=20)
        final=response.geturl();row['final_host']=urlsplit(final).hostname
        if urlsplit(final).scheme.lower()!='https':raise ValueError('non-HTTPS final response rejected')
        row['http_status']=response.status
        selected=('Content-Range','Content-Length','Accept-Ranges','Content-Type','Content-Encoding','ETag')
        row['headers']={k:response.headers.get(k) for k in selected}
        lengths=response.headers.get_all('Content-Length') or [];ranges=response.headers.get_all('Content-Range') or []
        expected_range=f'bytes {start}-{end}/{TOTAL}'
        if response.status!=206 or len(ranges)!=1 or ranges[0]!=expected_range or len(lengths)!=1 or not lengths[0].isdigit() or int(lengths[0])!=expected:raise ValueError('status/Content-Range/Content-Length rejected before body read')
        if response.headers.get('Content-Encoding','identity').lower() not in ('identity',''):raise ValueError('encoded body rejected before read')
        row['status']='HEADERS_VERIFIED_BODY_STARTING';checkpoint()
        with (HERE/(label+'.bin')).open('xb') as out:
            while len(data)<expected:
                block=response.read(min(65536,expected-len(data)))
                if not block:raise ValueError('truncated range body')
                out.write(block);data.extend(block)
                row['actual_body_bytes_read']+=len(block);LEDGER['actual_response_body_bytes_read']+=len(block)
                if LEDGER['actual_response_body_bytes_read']>CAP:raise ValueError('body cap exceeded')
                checkpoint()
            out.flush();os.fsync(out.fileno())
        row['body_sha256']=sha(data);row['status']='COMPLETED';row['utc_completed']=utc();checkpoint()
        return bytes(data),final
    except Exception as exc:
        row['status']='FAILED';row['error_type']=type(exc).__name__
        row['error']=re.sub(r'https?://[^\s]+','<redacted-url>',str(exc))
        if isinstance(exc,HTTPError):row['http_status']=exc.code;row['final_host']=urlsplit(exc.geturl()).hostname
        row['partial_body_sha256']=sha(data);row['utc_completed']=utc();checkpoint();raise
    finally:
        if response is not None:response.close()
def eocd(tail,start):
    candidates=[]
    for at in range(len(tail)-22,-1,-1):
        if tail[at:at+4]!=b'PK\x05\x06':continue
        fields=struct.unpack_from('<4s4H2IH',tail,at)
        if at+22+fields[-1]==len(tail):candidates.append((at,fields))
    if len(candidates)!=1:raise ValueError('missing/ambiguous EOCD ending at exact EOF')
    at,f=candidates[0];_,disk,cd_disk,disk_entries,entries,cd_size,cd_offset,comment=f
    if disk or cd_disk or disk_entries!=entries:raise ValueError('multi-disk archive rejected')
    if entries==65535 or cd_size==0xffffffff or cd_offset==0xffffffff or (at>=20 and tail[at-20:at-16]==b'PK\x06\x07'):raise ValueError('ZIP64 rejected')
    if not 0<entries<=5000 or not 0<cd_size<=CAP or cd_offset+cd_size!=start+at:raise ValueError('central directory count/boundaries rejected')
    return {'entries':entries,'central_directory_start':cd_offset,'central_directory_bytes':cd_size,'eocd_start':start+at,'comment_bytes':comment,'single_disk':True,'zip64':False}
def safe_name(raw,flags):
    name=raw.decode('utf-8' if flags&0x800 else 'cp437')
    if not name or '\x00' in name or '\\' in name or name.startswith('/'):raise ValueError('unsafe member name')
    trimmed=name[:-1] if name.endswith('/') else name
    parts=trimmed.split('/')
    if not trimmed or any(p in ('','.','..') or ':' in p for p in parts):raise ValueError('unsafe member components')
    return name,parts
def parse_directory(cd,meta):
    members=[];names=set();offsets=set();at=0
    for _ in range(meta['entries']):
        if at+46>len(cd):raise ValueError('truncated central header')
        values=struct.unpack_from('<4s6H3I5H2I',cd,at)
        sig,made,needed,flags,method,mtime,mdate,crc,compressed,expanded,nlen,xlen,clen,disk,internal,external,local=values
        if sig!=b'PK\x01\x02' or not nlen:raise ValueError('invalid central header')
        if disk or flags&(1|64|8192):raise ValueError('multi-disk/encrypted member rejected')
        if compressed==0xffffffff or expanded==0xffffffff or local==0xffffffff or disk==65535 or needed>=45:raise ValueError('ZIP64/unsupported required-version member rejected')
        end=at+46+nlen+xlen+clen
        if end>len(cd):raise ValueError('truncated central variable fields')
        raw=cd[at+46:at+46+nlen];extra=cd[at+46+nlen:at+46+nlen+xlen]
        off=0
        while off<len(extra):
            if off+4>len(extra):raise ValueError('truncated central extra')
            kind,length=struct.unpack_from('<HH',extra,off);off+=4
            if off+length>len(extra) or kind==1:raise ValueError('invalid/ZIP64 central extra')
            off+=length
        name,parts=safe_name(raw,flags)
        if name in names or local in offsets:raise ValueError('duplicate filename/local header offset')
        names.add(name);offsets.add(local)
        kind=(external>>16)&0o170000
        if kind not in (0,0o100000,0o040000):raise ValueError('nonregular member type rejected')
        directory=name.endswith('/')
        if (kind==0o040000 or external&0x10) and not directory:raise ValueError('directory attribute/name mismatch')
        if not local<meta['central_directory_start'] or local+30+compressed>meta['central_directory_start']:raise ValueError('local-header/payload minimal bounds rejected')
        members.append({'name':name,'kind':'directory' if directory else 'file','compressed_bytes':compressed,'uncompressed_bytes':expanded,'crc32':f'{crc:08x}','flags':flags,'compression_method':method,'local_header_offset':local,'filename_raw_sha256':sha(raw),'required_version':needed})
        at=end
    if at!=len(cd):raise ValueError('unparsed central directory bytes')
    sequences={};unexpected=[]
    for m in members:
        if m['kind']=='directory':continue
        parts=m['name'].split('/');candidates=[x for x in parts[:-1] if re.fullmatch('[iv]_[^/]+',x)]
        if len(candidates)!=1:unexpected.append(m['name']);continue
        seq=candidates[0];leaf=parts[-1]
        s=sequences.setdefault(seq,{'members':[],'compressed_bytes':0,'uncompressed_bytes':0,'image_filenames':[],'homography_filenames':[],'unexpected_filenames':[]})
        s['members'].append(m['name']);s['compressed_bytes']+=m['compressed_bytes'];s['uncompressed_bytes']+=m['uncompressed_bytes']
        if re.fullmatch('[1-6]\.ppm',leaf):s['image_filenames'].append(leaf)
        elif re.fullmatch('H_1_[2-6]',leaf):s['homography_filenames'].append(leaf)
        else:s['unexpected_filenames'].append(leaf)
    anomalies=[]
    for seq,s in sequences.items():
        s['image_filenames'].sort();s['homography_filenames'].sort()
        s['image_file_count']=len(s['image_filenames']);s['homography_file_count']=len(s['homography_filenames'])
        s['missing_expected_images']=sorted({f'{i}.ppm' for i in range(1,7)}-set(s['image_filenames']))
        s['missing_expected_homographies']=sorted({f'H_1_{i}' for i in range(2,7)}-set(s['homography_filenames']))
        if s['missing_expected_images'] or s['missing_expected_homographies'] or s['unexpected_filenames']:anomalies.append(seq)
    return members,sequences,{'nonsequence_files':unexpected,'sequences_with_missing_or_unexpected_names':anomalies}
def main():
    start=time.monotonic();atomic('started.json',{'utc':utc(),'scope':'Directory metadata only; no local headers, decompression, image/H values or science.','source_sha256':sha(Path(__file__).read_bytes()),'request_source':URL,'whole_archive_sha256_not_verified':WHOLE_SHA})
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('operation alarm before outer hard timeout')));signal.alarm(110)
    result={'status':'FAILED','whole_file_sha256_verified':False,'payload_members_opened':0,'local_headers_read':0,'new_asset_bytes':0}
    try:
        first=TOTAL-TAIL;tail,final=ranged_get(URL,first,TOTAL-1,'tail')
        meta=eocd(tail,first);atomic('eocd.json',meta)
        if meta['central_directory_start']<first:
            prefix,_=ranged_get(final,meta['central_directory_start'],first-1,'central-prefix')
            cd=(prefix+tail)[:meta['central_directory_bytes']]
        else:cd=tail[meta['central_directory_start']-first:meta['central_directory_start']-first+meta['central_directory_bytes']]
        if len(cd)!=meta['central_directory_bytes']:raise ValueError('incomplete assembled directory')
        members,sequences,anomalies=parse_directory(cd,meta)
        directory={'schema':'0009-hpatches-central-directory-v1','upstream_url':URL,'declared_whole_bytes':TOTAL,'declared_whole_sha256':WHOLE_SHA,'whole_sha256_verified':False,'central_directory_sha256':sha(cd),'eocd':meta,'members':members,'sequences':sequences,'anomalies':anomalies,'limitations':'Central directory metadata only. CRC, local header/name, compressed payload, images and homography values were not read or verified. Different names do not establish scene independence. No subset or performance label selection. Dataset-card MIT claim was supplied by parent; license not independently reread in this operation.'}
        atomic('directory.json',directory)
        result.update(status='COMPLETED_DIRECTORY_ONLY',entries=len(members),sequences=len(sequences),image_filenames=sum(s['image_file_count'] for s in sequences.values()),homography_filenames=sum(s['homography_file_count'] for s in sequences.values()),anomalies=anomalies,central_directory_sha256=sha(cd),directory_json_sha256=sha((HERE/'directory.json').read_bytes()),central_directory_bytes=len(cd),compressed_payload_sum_bytes=sum(m['compressed_bytes'] for m in members),subset_range_evidence='Central records supply per-member offsets, methods and sizes, supporting a future bounded local-header/member retrieval design. Payload integrity, server ranges at those offsets and useful geometry are untested.')
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=re.sub(r'https?://[^\s]+','<redacted-url>',str(exc)),restriction='Stopped after first failure; no retry, fallback full body, image/H access or extra GET.')
    finally:
        signal.alarm(0);result.update(utc_completed=utc(),wall_seconds=time.monotonic()-start,new_asset_bytes=LEDGER['actual_response_body_bytes_read'],range_get_operations=len(LEDGER['range_get_operations']),final_hosts=sorted({r['final_host'] for r in LEDGER['range_get_operations'] if r.get('final_host')}),ending_asset_ceiling_bytes=7532690072+LEDGER['actual_response_body_bytes_read']);checkpoint();atomic('result.json',result)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result['status']=='COMPLETED_DIRECTORY_ONLY' else 1
if __name__=='__main__':raise SystemExit(main())
