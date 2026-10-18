"""One qualification: six isolated imports, within an external 60-second cap."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]/'bootstapir-cache/runtime-fixed'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    began=time.monotonic();rows=[]
    for variant in ('base','runtime-fixed'):
        for module in ('numpy','scipy','cv2'):
            if time.monotonic()-began>54:raise RuntimeError('qualification elapsed guard; no retries')
            label=variant+'-'+module;env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
            if variant=='runtime-fixed':env['PYTHONPATH']=str(RUNTIME)
            command=['timeout','--kill-after=1s','8',sys.executable,str(HERE/'probe.py'),module]
            write(HERE/(label+'.command.json'),{'argv':command,'cwd':str(HERE),'PYTHONPATH':env.get('PYTHONPATH'),'PYTHONNOUSERSITE':'1','thread_variables':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}})
            start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
            with (HERE/(label+'.stdout.log')).open('xb') as out,(HERE/(label+'.stderr.log')).open('xb') as err:
                child=subprocess.Popen(command,cwd=HERE,env=env,stdout=out,stderr=err)
                _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
                out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
            text=(HERE/(label+'.stdout.log')).read_text()
            try:result=json.loads(text)
            except ValueError:result={'status':'no_valid_probe_json'}
            record={'variant':variant,'module':module,'utc_started':start,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exit':child.returncode,'raw_wait_status':status,'wall_seconds':time.monotonic()-clock,'peak_rss_kib':usage.ru_maxrss,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'stdout_sha256':sha(HERE/(label+'.stdout.log')),'stderr_sha256':sha(HERE/(label+'.stderr.log')),'command_sha256':sha(HERE/(label+'.command.json')),'probe':result}
            write(HERE/(label+'.execution.json'),record);rows.append(record)
    write(HERE/'summary.json',{'status':'completed_import_qualification_only','probes':rows,'wall_seconds':time.monotonic()-began,'qualification_source_sha256':sha(__file__),'new_asset_bytes':0,'geometry_calls':0,'model_calls':0,'installations':0,'scope':'Availability only. A new experiment requires independent design/freeze; this does not establish geometric accuracy or runtime feasibility.'})
    print(json.dumps({'status':'qualification_complete','probes':len(rows),'availability':[{'variant':r['variant'],'module':r['module'],'status':r['probe']['status'],'actual_exit':r['actual_exit']} for r in rows]},sort_keys=True))
if __name__=='__main__':main()
