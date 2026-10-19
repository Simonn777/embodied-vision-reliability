"""Isolated official build, bounded time and summed process RSS."""
from pathlib import Path
import json,os,shutil,signal,subprocess,sys,time
P=Path(__file__).resolve().parent
C=P/'inputs/cmake/data/bin/cmake'
E=P/'inputs/eigen-3.4.0'
S=P/'inputs/PoseLib-6a636b1d84273d5e3ba97676463476569611572a'
env=dict(os.environ,PYTHONPATH=str(P/'inputs'),CMAKE_BUILD_PARALLEL_LEVEL='2')
commands=[
 ['eigen-config',str(C),'-S',str(E),'-B',str(P/'eigen-build'),'-DCMAKE_INSTALL_PREFIX='+str(P/'eigen-install'),'-DBUILD_TESTING=OFF'],
 ['eigen-install',str(C),'--install',str(P/'eigen-build')],
 ['pose-config',str(C),'-S',str(S),'-B',str(P/'pose-build'),'-DCMAKE_BUILD_TYPE=Release','-DPYTHON_PACKAGE=ON','-DGENERATE_STUBS=OFF','-DBUILD_SHARED_LIBS=OFF','-DBUILD_TESTS=OFF','-DMARCH_NATIVE=OFF','-DEigen3_DIR='+str(P/'eigen-install/share/eigen3/cmake'),'-DPython_EXECUTABLE='+sys.executable],
 ['pose-build',str(C),'--build',str(P/'pose-build'),'-j','2']]
with (P/'build-started.json').open('x') as f: json.dump(commands,f,indent=2)
outer=time.monotonic()
for name,*command in commands:
 begin=time.monotonic(); peak=0; termination=None
 with (P/(name+'.log')).open('xb') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while child.poll() is None:
    raw=subprocess.run(['/bin/ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,timeout=3)
    rows=[list(map(int,s.split())) for s in raw.stdout.splitlines() if len(s.split())==3]
    descendants={child.pid}
    for _ in range(10):
     new=descendants|{pid for pid,ppid,rss in rows if ppid in descendants}
     if new==descendants: break
     descendants=new
    peak=max(peak,sum(rss for pid,ppid,rss in rows if pid in descendants))
    if peak>8*1024*1024: termination='rss_limit'; break
    if time.monotonic()-outer>1200: termination='total_timeout'; break
    time.sleep(.25)
  finally:
   if child.poll() is None: os.killpg(child.pid,signal.SIGKILL)
   code=child.wait(timeout=5)
  result={'stage':name,'exit_code':code,'termination':termination,'seconds':time.monotonic()-begin,'peak_group_rss_kib':peak}
  with (P/(name+'.receipt.json')).open('x') as f: json.dump(result,f,indent=2)
  print(json.dumps(result),flush=True)
  if code!=0 or termination: raise SystemExit(1)
dest=P/'runtime/poselib'; dest.mkdir(parents=True)
for src in (S/'pyposelib').iterdir():
 if src.is_file(): shutil.copy2(src,dest/src.name)
binary=list((P/'pose-build/pybind').glob('_core*.so'))
if len(binary)!=1: raise ValueError('expected one binary')
shutil.copy2(binary[0],dest/binary[0].name)
print('Isolated runtime ready; no benchmark operation',flush=True)
