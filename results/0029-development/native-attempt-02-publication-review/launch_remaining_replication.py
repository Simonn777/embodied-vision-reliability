"""Dispatch the remaining fixed videos once; stop immediately on any failure."""
import hashlib,json,subprocess,sys
from pathlib import Path
prep=Path(__file__).resolve().parent
r=prep.parent/'embodied-vision-reliability'
b=prep.parent/'0029-native-development-02'
commit='098dee64878b5ef33edbac12223f86da176e091e'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()==commit
assert not subprocess.check_output(['git','status','--porcelain'],cwd=r,text=True).strip()
ci=json.loads((b/'source-ci.json').read_text())
assert (ci['headSha'],ci['status'],ci['conclusion'])==(commit,'completed','success')
manifest=json.loads((b/'specifications-sha256.json').read_text())
# The reviewed metadata is read before any remaining scientific dispatch.
assert hashlib.sha256((b/'specifications-sha256.json').read_bytes()).hexdigest()=='ff62ec81b628784c8b8988e9b669bae1625cd8070bb316ea044567a41173bb71'
for relative,digest in manifest.items():
    assert hashlib.sha256((b/relative).read_bytes()).hexdigest()==digest
supervisor=r/'experiments/0029-memory-information/supervise_exit_checked.py'
assert hashlib.sha256(supervisor.read_bytes()).hexdigest()=='8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8'
def require_completed(name):
    receipt=json.loads((b/name/'supervision/receipt.json').read_text())
    assert receipt['status']=='completed' and receipt['exit_code']==0 and receipt['termination'] is None
    assert not receipt['remaining_descendant_pids'] and not receipt['cleanup_errors'] and not receipt['signal_attempts']
    worker=json.loads((b/name/'video/receipt.json').read_text())
    assert worker['status']=='completed' and worker['eligible'] is True
    assert (b/name/'video/seal.json').is_file()
    return receipt
for name in ('dogs-scale','upside-down'):
    require_completed(name)
for name in ('dog-agility','cat-girl','planes-water','dog-gooses'):
    directory=b/name
    spec=directory/'spec.json'
    assert not (directory/'video').exists() and not (directory/'supervision').exists()
    print(json.dumps({'sequence':name,'state':'launching'}),flush=True)
    with (directory/'launcher.stdout.log').open('xb') as out,(directory/'launcher.stderr.log').open('xb') as err:
        result=subprocess.run(['/opt/homebrew/bin/python3.12','-B',str(supervisor),str(spec)],cwd=r,stdout=out,stderr=err)
    if result.returncode:
        print(json.dumps({'sequence':name,'state':'failed','launcher_returncode':result.returncode}),flush=True)
        sys.exit(1)
    receipt=require_completed(name)
    print(json.dumps({'sequence':name,'state':'completed','seconds':receipt['wall_seconds'],'peak_kib':receipt['peak_group_rss_kib']}),flush=True)
