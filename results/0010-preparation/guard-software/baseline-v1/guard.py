"""Read-only cached-probe guard. No download, deserialization, repair or probe."""
import argparse, errno, hashlib, json, os, stat, subprocess
from pathlib import Path

MAIN='e9e7c7609e53e1bc1b3228096f9b36008cb78d6b'; SOURCE='ececb253bfa00acffdb8d84c91d848c2802444a0'
FREEZE='93629d385b286b9471167e0c15c49bf29120096e'; RECEIPT='protocols/0010-clip-capability-freeze.json'
RECEIPT_SHA='8dfec7362d01fac80c46f9871fe4e8696c90d12ba07038c2635a6f8dd947b29f'
PINS_SHA='20a8aabd3f501d2475610ef8d8fd3b51e135acd74eaa2226426a1a2cbc906258'
PARTIAL=(343605248,'97127aac2ade9ec552c1828df711e2c40aeb5e50ddc08a44959734cb0d238e01')
TEMPS={'.EuroSAT.zip.91Kpzc':24117248,'.ViT-B-32.pt.partial.hvJJcZ':197132288,'.ViT-B-32.pt.partial.hvJJcZ.dBQxjR':185073664}
ABSENT=('clip-0010-assets/0010-probe-started.json','scratch/0010-capability-output','scratch/0010-capability-control/probe')

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def snapshot(path):
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink path')
    s=path.lstat(); require(stat.S_ISREG(s.st_mode),'not regular')
    return dict(dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def inspect_file(path, expected=None):
    before=snapshot(path); h=hashlib.sha256(); g=hashlib.sha1(b'blob '+str(before['size']).encode()+b'\0'); total=0
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        require((os.fstat(f.fileno()).st_dev,os.fstat(f.fileno()).st_ino)==(before['dev'],before['inode']),'opened inode changed')
        for block in iter(lambda:f.read(1<<20),b''): h.update(block); g.update(block); total+=len(block)
    after=snapshot(path); require(before==after and total==before['size'],'file changed while reading')
    got=dict(bytes=total,sha256=h.hexdigest(),git_blob_sha1=g.hexdigest())
    if expected: require(all(v is None or got[k]==v for k,v in expected.items()),'file identity mismatch')
    return dict(before=before,after=after,**got)

def prefix_equal(partial, final):
    before={str(p):snapshot(p) for p in (partial,final)}; compared=0
    with os.fdopen(os.open(partial,os.O_RDONLY|os.O_NOFOLLOW),'rb') as a, os.fdopen(os.open(final,os.O_RDONLY|os.O_NOFOLLOW),'rb') as b:
        for raw in iter(lambda:a.read(1<<20),b''):
            require(raw==b.read(len(raw)),'partial is not a final-file prefix'); compared+=len(raw)
    require(all(snapshot(Path(p))==v for p,v in before.items()),'prefix input changed')
    return dict(bytes_compared=compared,exact_prefix=True,origin='UNKNOWN')

def process_snapshot(proc, targets):
    counts={}; errors=[]; related=[]; holders=[]; fd_count=0; pid_count=0
    def observe(operation,path,call):
        c=counts.setdefault(operation,dict(success=0,enoent=0,permission=0,other=0))
        try: value=call(); c['success']+=1; return value
        except OSError as exc:
            kind='enoent' if exc.errno==errno.ENOENT else 'permission' if exc.errno in (errno.EACCES,errno.EPERM) else 'other'
            c[kind]+=1; errors.append(dict(operation=operation,path=str(path.relative_to(proc)),kind=kind,errno=exc.errno)); return None
    pids=observe('proc_list',proc,lambda:list(proc.iterdir())) or []
    require(len(pids)<=10000,'process enumeration cap')
    for d in pids:
        if not d.name.isdecimal(): continue
        pid_count+=1
        raw=observe('cmdline',d/'cmdline',lambda:(d/'cmdline').read_bytes())
        if raw is not None:
            args=[a.decode(errors='replace') for a in raw.split(bytes([0])) if a]
            scripts=[Path(a).name for a in args if '\n' not in a and (Path(a).name in ('run_capability.py','run_antithetic_training.py','run_antithetic_gradient_screen.py','run_geometric_stage0.py') or (a.endswith('/cli.py') and '0010' in a))]
            if scripts: related.append(dict(pid=int(d.name),script_basenames=scripts))
        entries=observe('fd_list',d/'fd',lambda:list((d/'fd').iterdir())) or []
        for fd in entries:
            fd_count+=1; require(fd_count<=100000,'fd enumeration cap')
            st=observe('fd_stat',fd,fd.stat); link=observe('fd_readlink',fd,lambda:os.readlink(fd))
            matches=[] if st is None else [k for k,v in targets.items() if (st.st_dev,st.st_ino)==(v['dev'],v['inode'])]
            if matches:
                info=observe('fdinfo',d/'fdinfo'/fd.name,lambda:(d/'fdinfo'/fd.name).read_text())
                flags=None
                if info is not None:
                    vals=[x.split()[1] for x in info.splitlines() if x.startswith('flags:')]
                    require(len(vals)==1,'fdinfo flags missing/ambiguous'); flags=int(vals[0],8)
                holders.append(dict(pid=int(d.name),fd=fd.name,matches=matches,link_basename=Path(link).name if link else None,
                                    flags_octal=oct(flags) if flags is not None else None,writable=bool(flags & os.O_ACCMODE) if flags is not None else None))
    gaps=sum(c['permission']+c['other'] for c in counts.values())
    return dict(counts=counts,errors=errors,related=related,matching_fds=holders,fd_entries=fd_count,
                visible_processes=pid_count,admissible=pid_count>0 and not gaps and not related and not holders,cmdline_separator='bytes([0])',
                scope='Visible /proc snapshot only. ENOENT is a disappearing-entry race; hidden/prior/future writers and global absence remain unproven.',absence_proven=False)

def git_gate(repo):
    def git(*args): return subprocess.check_output(['git','--no-optional-locks',*args],cwd=repo)
    require(git('rev-parse','HEAD').decode().strip()==MAIN and not git('status','--porcelain'),'old worktree changed')
    require(git('rev-list','--parents','-n','1',FREEZE).decode().split()==[FREEZE,SOURCE],'freeze parent')
    require(git('diff-tree','--no-commit-id','--name-only','-r',SOURCE,FREEZE).decode().splitlines()==[RECEIPT],'freeze delta')
    subprocess.run(['git','merge-base','--is-ancestor',FREEZE,MAIN],cwd=repo,check=True)
    raw=(repo/RECEIPT).read_bytes(); require(sha(raw)==RECEIPT_SHA,'canonical freeze hash')
    require(raw==git('show',FREEZE+':'+RECEIPT)==git('show',MAIN+':'+RECEIPT)==git('show',':'+RECEIPT),'receipt tree/index')
    f=json.loads(raw); require(f['source_commit']==SOURCE and len(f['inputs'])==25,'25 frozen inputs')
    records={RECEIPT:inspect_file(repo/RECEIPT,{'sha256':RECEIPT_SHA})}
    for name,spec in f['inputs'].items():
        records[name]=inspect_file(repo/name,spec); data=(repo/name).read_bytes()
        require(all(data==git('show',ref+':'+name) for ref in (SOURCE,MAIN,'')),'frozen tree/index mismatch')
    return records

def run(base,repo,out,*,proc=Path('/proc'),cgroup=Path('/sys/fs/cgroup')):
    base,repo,out=Path(base).resolve(),Path(repo).resolve(),Path(out).resolve()
    protected=[repo]+[base/n for n in ('clip-0010-assets','clip-0010-downloads','clip-0010-wheels','scratch/0010-capability-control','scratch/0010-partial-review')]
    require(out!=base and out not in base.parents and all(out!=p and out not in p.parents and p not in out.parents for p in protected),'output overlaps preserved inputs')
    out.mkdir(parents=True,exist_ok=False); r=dict(status='started',scope='cached guard only; no model/download/repair',cause='UNKNOWN'); observed={}; sources={}
    def save(): (out/'guard.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    try:
        r['cgroup_before']={n:(cgroup/n).read_text() for n in ('memory.max','memory.current','memory.stat','memory.events')}
        r['available_cpus']=sorted(os.sched_getaffinity(0)); save()
        require(r['cgroup_before']['memory.max'].strip()=='8589934592' and {0,1,2,3}<=set(r['available_cpus']),'cgroup/CPU slots')
        r['absent_before']={n:not os.path.lexists(base/n) for n in ABSENT}; require(all(r['absent_before'].values()),'prior probe marker/output/supervision')
        sources=git_gate(repo); r['source_checks']=sources
        pinpath=Path(__file__).with_name('preservation-pins.json'); require(sha(pinpath.read_bytes())==PINS_SHA,'preservation pins')
        pins=json.loads(pinpath.read_text())['files']
        for name,spec in pins.items(): observed[name]=inspect_file(base/name,spec)
        r['preserved']=observed; control=base/'scratch/0010-capability-control'
        for label,code in [('wheels',0),('offline-install',0),('imports',0),('assets',0),('before-probe',1)]:
            e=json.loads((control/(label+'.execution.json')).read_text()); require(e['actual_exit']==code and e['raw_wait_status']==code*256,'original stage exit changed')
            for suffix,key in [('.command.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')]:
                require(observed['scratch/0010-capability-control/'+label+suffix]['sha256']==e[key],'stage log identity')
        cfg=json.loads((repo/'experiments/0010/config.json').read_text()); require(len(cfg['assets'])==12,'12 full assets')
        r['assets']={n:inspect_file(base/'clip-0010-assets'/n,s) for n,s in cfg['assets'].items()}
        checkpoint=cfg['assets']['weights/ViT-B-32.pt']; download='clip-0010-downloads/ViT-B-32.pt'
        observed[download]=inspect_file(base/download,checkpoint)
        partial=download+'.partial'; observed[partial]=inspect_file(base/partial,dict(zip(('bytes','sha256'),PARTIAL)))
        p=observed[partial]['before']; finals=[observed[download]['before'],r['assets']['weights/ViT-B-32.pt']['before']]
        require(all((p['dev'],p['inode'])!=(f['dev'],f['inode']) for f in finals),'partial shares final inode')
        r['prefix']=prefix_equal(base/partial,base/download); r['excluded_temporary_copies']={}
        for name,size in TEMPS.items():
            rel='clip-0010-downloads/'+name; observed[rel]=inspect_file(base/rel,{'bytes':size}); r['excluded_temporary_copies'][name]=observed[rel]
        targets={k:v['before'] for k,v in observed.items()}; targets.update({'asset/'+k:v['before'] for k,v in r['assets'].items()})
        r['processes']=process_snapshot(proc,targets); save(); require(r['processes']['admissible'],'processes/holders/inspection gaps')
        r['absent']={n:not os.path.lexists(base/n) for n in ABSENT}; require(all(r['absent'].values()),'prior probe marker/output/supervision')
        require(all(snapshot(base/n)==v['before'] for n,v in observed.items()),'preserved file changed')
        require(all(snapshot(base/'clip-0010-assets'/n)==v['before'] for n,v in r['assets'].items()),'asset changed')
        require(all(snapshot(repo/n)==v['before'] for n,v in sources.items()),'source changed')
        r.update(status='PASS_CACHED_GUARD_ONLY',stat_unchanged=True,old_failure_exit=1,first_probe_executed=False)
    except Exception as exc: r.update(status='BLOCKED',error_type=type(exc).__name__,reason=str(exc))
    finally: save()
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','repo','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args(); result=run(a.base,a.repo,a.output); print(json.dumps({'status':result['status'],'first_probe_executed':False}))
    raise SystemExit(0 if result['status']=='PASS_CACHED_GUARD_ONLY' else 1)
