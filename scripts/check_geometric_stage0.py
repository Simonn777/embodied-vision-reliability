#!/usr/bin/env python3
"""Independent saved-coordinate arithmetic audit; never imports production EVR.

No RNG draws or tail DP. Stored tail counts are bounded and ranked, not recounted.
Only NumPy + stdlib. Floating arithmetic tolerance: atol=1e-10, rtol=1e-12.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import numpy as np

FAMILIES = ('uniform', 'clustered_outlier', 'compact_true', 'sparse_isolated')
CONTROLS = ('raw', 'msac', 'cell', 'excess', 'ratio')
METHODS = CONTROLS + ('tail',)
STREAMS = ('global-shift','inlier-xy','inlier-noise','outlier-source','outlier-target',
           'outlier-pairing','wrong-noise','clone-base','priority','candidate-order')
FREEZE = 'protocols/0009-geometric-stage0-freeze.json'
CHECKS = NUMERIC = 0
MAX_DIFFERENCE = 0.0


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    global NUMERIC, MAX_DIFFERENCE
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': dictionary')
        for k,v in expected.items():
            require(k in actual, label + ': missing ' + k)
            same(actual[k], v, label + '.' + k)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for i,(a,b) in enumerate(zip(actual, expected)):
            same(a,b,label + '['+str(i)+']')
    elif isinstance(expected, float):
        require(type(actual) in (int,float) and math.isfinite(actual), label + ': finite numeric')
        difference = abs(actual-expected); NUMERIC += 1
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, difference)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), label + ': numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact mismatch')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def ah(value, kind='f'):
    array = np.asarray(value, dtype={'f':'<f8','i':'<i8','b':np.bool_}[kind], order='C')
    header = json.dumps({'dtype':array.dtype.str,'shape':list(array.shape)},sort_keys=True,separators=(',',':')).encode('ascii')
    return digest(header+b'\n'+array.tobytes(order='C'))


def pool():
    small = [(x,y) for x in (-8,0,8) for y in (-8,0,8)]
    offsets = small + [(x+128,y+128) for x,y in small] + [(-24,0),(24,0),(0,-24),(0,24),(-24,-24),(-24,24),(24,-24),(24,24),(32,0),(-32,0),(0,32),(0,-32),(32,32),(-32,-32)]
    return [[[1.,0.,float(x)],[0.,1.,float(y)],[0.,0.,1.]] for x,y in offsets]


def order():
    return [f'{FAMILIES[(s+j)%4]}-{s:02d}' for s in range(64) for j in range(4)]


def dedup(x,y,priorities):
    retained, decisions = [], []
    for i in sorted(range(len(x)), key=lambda k:(priorities[k],k)):
        sx = any(math.hypot(*(x[i]-x[j])) <= 1 for j in retained)
        sy = any(math.hypot(*(y[i]-y[j])) <= 1 for j in retained)
        keep = not (sx or sy)
        decisions.append({'raw_id':i,'accepted':keep,'reason':'both' if sx and sy else 'source' if sx else 'target' if sy else None})
        if keep: retained.append(i)
    return retained, decisions


def components(a):
    # Independent union-find over the two vertex sets, without rook computations.
    n = len(a); parent = list(range(2*n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    edges = list(zip(*np.nonzero(a)))
    for i,j in edges: parent[root(n+int(j))] = root(int(i))
    groups = {}
    for i,j in edges:
        rows,cols,count = groups.setdefault(root(int(i)), [set(),set(),0])
        rows.add(int(i)); cols.add(int(j)); groups[root(int(i))][2] += 1
    return [{'rows':len(r),'columns':len(c),'edges':e} for r,c,e in sorted(groups.values(),key=lambda g:min(g[0]))]


def scalar_rows(x,y,candidates):
    n = len(x)
    cell = lambda p: tuple(np.clip(np.floor(p/64),0,7).astype(int))
    cx,cy = Counter(map(cell,x)),Counter(map(cell,y))
    cs,ct = [cx[cell(p)] for p in x],[cy[cell(p)] for p in y]
    weights = np.array([1/math.sqrt(a*b) for a,b in zip(cs,ct)])
    result = []
    for cid,h in enumerate(candidates):
        # Fixed translations, but retain the frozen normalized-homogeneous
        # floating convention: direct x+t can flip an inclusive radius bit.
        offset=np.array([h[0][2],h[1][2]])
        scale=max(1.,float(np.max(np.abs(offset)))); unit=1./scale
        projected=(x*unit+offset/scale)/unit
        difference=projected-y
        d = np.hypot(difference[:,0],difference[:,1])
        dx=projected[:,None,0]-y[None,:,0]
        dy=projected[:,None,1]-y[None,:,1]
        a = np.hypot(dx,dy) <= 3
        inside = d <= 3; count = int(inside.sum()); edges = int(a.sum())
        require(int(np.trace(a)) == count, 'diagonal support')
        key = [count*n,edges] if edges else [0,1]
        result.append({'canonical_id':cid,'n':n,'N':count,'S':edges,
            'H_sha256':ah(h),'A_sha256':ah(a,'b'),'graph_components':components(a),
            'raw_key':count,'msac_key':float(np.maximum(0,1-(d[inside]/3)**2).sum()),
            'cell_key':float(weights[inside].sum()),'excess_key':n*count-edges,
            'excess':count-edges/n,'mu':edges/n,'ratio_key':key,'ratio':float(Fraction(*key))})
    return result, {'source':cs,'target':ct}


def case_audit(rec, label, *, fixture=False):
    status = rec['status']; require(status in ('completed','tail_failed','incomplete','invalid','unattempted'),'case status')
    complete = status == 'completed'; controls = rec.get('controls_complete') is True
    require(not complete or controls, 'completed case lacks controls')
    hypotheses = rec.get('hypotheses',[]); selections = rec.get('selections',{})
    if not complete:
        require(selections.get('tail',{}).get('selected_id') is None,'failed tail selection retained')
    if status == 'unattempted':
        require(not hypotheses and not controls and not selections,'unattempted has computation')
        return False
    inputs = rec.get('inputs')
    if inputs is None:
        require(not complete and not controls and not hypotheses,'completed missing input')
        return False
    require(set(inputs)=={'raw_x','raw_y','priorities','candidate_order','candidates'},'input field set')
    x,y = np.asarray(inputs['raw_x'],float), np.asarray(inputs['raw_y'],float)
    priorities = inputs['priorities']; candidates = inputs['candidates']; shuffled = inputs['candidate_order']
    require(x.ndim==2 and x.shape[1]==2 and x.shape==y.shape and len(priorities)==len(x),'input dimensions')
    require(fixture or len(x)==160,'raw pair count')
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(priorities).all(),'input finite')
    require(((0<=x)&(x<=512)).all() and ((0<=y)&(y<=512)).all(),'coordinate range')
    require(all(0<=p<1 for p in priorities),'priorities range')
    same(candidates,pool(),'fixed candidate pool')
    require(all(type(i) is int for i in shuffled) and sorted(shuffled)==list(range(32)),'saved permutation')
    for name,value in inputs.items():
        if complete or name in rec.get('input_hashes',{}):
            same(rec['input_hashes'][name],ah(value,'i' if name=='candidate_order' else 'f'),'input hash '+name)
    kept,decisions = dedup(x,y,priorities)
    for key,value in [('retained_ids',kept),('dedup_decisions',decisions),('retained_x',x[kept].tolist()),('retained_y',y[kept].tolist()),('retained_hashes',{'x':ah(x[kept]),'y':ah(y[kept])})]:
        if complete or key in rec: same(rec[key],value,key)
    expected,counts = scalar_rows(x[kept],y[kept],candidates)
    if complete or 'cell_counts' in rec: same(rec['cell_counts'],counts,'cell counts')
    require(len(hypotheses)<=32 and (not controls or len(hypotheses)==32),'hypothesis completeness')
    require([h['canonical_id'] for h in hypotheses]==shuffled[:len(hypotheses)],'hypothesis prefix/order')
    tail_keys = []; attempted = 0; terminated = False
    for row in hypotheses:
        cid = row['canonical_id']; same(row,expected[cid],f'H{cid}')
        tail = row['tail']; state = tail['status']
        require(state in ('not_attempted','completed','unsupported','timeout','running'),'tail state')
        if state != 'not_attempted':
            require(not terminated,'tail attempts after failure/gap'); attempted += 1
        if state == 'completed':
            numerator,denominator = tail['numerator'],tail['denominator']
            require(isinstance(numerator,str) and re.fullmatch(r'[1-9][0-9]*',numerator) is not None,'tail numerator string')
            require(isinstance(denominator,str) and re.fullmatch(r'[1-9][0-9]*',denominator) is not None,'tail denominator string')
            p,q = int(numerator),int(denominator)
            require(q==math.factorial(len(kept)) and 1<=p<=q,'observed factorial tail bounds')
            if row['N']==0 or row['S']==len(kept)**2:
                require(p==q,'certain-event tail must equal one')
            require(type(tail['operations']) is int and 0<=tail['operations']<=2000000,'tail operation bound')
            comps = expected[cid]['graph_components']
            metadata = [[c['rows'],c['columns'],c['edges'],'complete-biclique' if c['edges']==c['rows']*c['columns'] else 'subset-dp'] for c in comps]
            same(tail['components'],metadata,'tail component metadata')
            require(all(c['edges']==c['rows']*c['columns'] or min(c['rows'],c['columns'])<=12 for c in comps),'completed unsupported component')
            tail_keys.append(Fraction(p,q))
        else:
            terminated = True
            if state == 'unsupported': require(isinstance(tail.get('reason'),str) and tail['reason'],'unsupported reason')
    recorded_attempts=rec.get('tail_attempted',0)
    boundary_interrupt=(not complete and rec.get('deadline_expired') is True and
                        recorded_attempts==attempted+1 and attempted<32)
    require(recorded_attempts==attempted or boundary_interrupt,'tail attempted/counter transition')
    for method in CONTROLS:
        if controls or method in selections:
            require(len(hypotheses)==32,'selection before all controls computed')
            # Independently recomputed values; exact rational keys for ratio.
            keys = [Fraction(*expected[c]['ratio_key']) if method=='ratio' else expected[c][method+'_key'] for c in shuffled]
            best = max(keys); ties = [c for c,k in zip(shuffled,keys) if k==best]
            same(selections[method],{'selected_id':ties[0],'tied_ids':ties},'control selection '+method)
    if complete:
        require(len(tail_keys)==32 and attempted==32,'complete tail count')
        best=min(tail_keys); ties=[c for c,k in zip(shuffled,tail_keys) if k==best]
        same(selections['tail'],{'selected_id':ties[0],'tied_ids':ties},'tail selection')
    if label is not None:
        same(label['H_star'],np.eye(3).tolist(),'GT identity')
        same(label['true_candidate_id'],4,'GT ID')
        if not fixture:
            base=label['base_ids']; same(base[:128],list(range(128)),'original base IDs')
            require(len(base)==160 and all(type(b) is int and 0<=b<128 for b in base),'base IDs')
            require(len(set(base[128:]))==16 and all(v==2 for v in Counter(base[128:]).values()),'clone counts')
            require(all(base[i]==base[i+1] for i in range(128,160,2)),'clone repeat order')
            k=8 if rec['seed_index']<32 else 16
            same(label['true_flags'],[b<k for b in base],'true flags')
            same(label['clone_flags'],[i>=128 for i in range(160)],'clone flags')
            require(np.array_equal(x,x[base]) and np.array_equal(y,y[base]),'clone coordinates')
    for value in rec.get('timings',{}).values(): require(type(value) in (int,float) and math.isfinite(value) and value>=0,'phase time')
    if 'active_seconds' in rec: require(math.isfinite(rec['active_seconds']) and rec['active_seconds']>=0,'active time')
    return complete


def aggregate(records):
    counts={f:{m:0 for m in METHODS} for f in FAMILIES}; totals=Counter(); rows=[]
    paired={f:{m:{k:0 for k in ('tail_success_baseline_failure','baseline_success_tail_failure','both_success','both_failure')} for m in CONTROLS} for f in FAMILIES}
    complete=True
    for cid in order():
        r=records[cid]; f=r['family']; totals[f]+=1
        ok=(r['status']=='completed' and r.get('controls_complete') is True and len(r.get('hypotheses',[]))==32 and r.get('tail_attempted')==32 and all(h['tail']['status']=='completed' for h in r['hypotheses']))
        complete &= ok; result={}
        for m in METHODS:
            selected=r.get('selections',{}).get(m,{}).get('selected_id')
            offset=None if selected is None else [pool()[selected][0][2],pool()[selected][1][2]]
            error=None if offset is None else math.hypot(*offset)
            success=error is not None and error<=3 and (m!='tail' or ok)
            counts[f][m]+=int(success)
            result[m]={'selected_id':selected,'offset':offset,'transfer_error':error,'success':bool(success)}
        for m in CONTROLS:
            a,b=result['tail']['success'],result[m]['success']
            key='both_success' if a and b else 'tail_success_baseline_failure' if a else 'baseline_success_tail_failure' if b else 'both_failure'
            paired[f][m][key]+=1
        rows.append({'case_id':cid,'family':f,'status':r['status'],'methods':result})
    return {'metric':'mean_one_way_euclidean_81_grid_0_512','success_pixels':3.,'family_totals':dict(totals),'success_counts':counts,'paired':paired,'cases':rows,'gates':gates(counts,paired,complete)}


def gates(counts,paired,complete):
    primary=counts['clustered_outlier']; best=max(CONTROLS,key=lambda m:primary[m]); gain=primary['tail']-primary[best]
    harms={f:{m:paired[f][m]['baseline_success_tail_failure'] for m in CONTROLS} for f in ('uniform','compact_true','sparse_isolated')}
    harmless=all(n<=2 for values in harms.values() for n in values.values())
    return {'decision':'ELIGIBLE_PENDING_EXTERNAL_REVIEW' if complete and gain>=8 and harmless else 'NO_GO','coverage_complete':bool(complete),'primary_strongest_control':best,'primary_net_gain':gain,'primary_gain_pass':gain>=8,'paired_harms_15':harms,'paired_harm_pass':harmless,'insufficient_primary_headroom':primary[best]>56,'independent_resource_and_saved_result_review':'not_attested_by_this_evaluator'}


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)


def binding_audit(repo,binding):
    commit=binding['freeze_commit']; require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'freeze ID')
    raw=git(repo,'show',commit+':'+FREEZE); canonical=json.loads(raw)
    require(canonical['schema']=='evr0009.freeze.v1','freeze schema')
    require(binding['files']==canonical['files'],'exact frozen file map')
    require(digest(Path(__file__).read_bytes()) in canonical['files'].values(),'reviewer must itself be frozen')
    require(digest(raw)==binding['receipt_sha256'],'freeze receipt hash')
    same(binding['files'],canonical['files'],'binding files'); same(binding['source_commit'],canonical['source_commit'],'binding source')
    same(git(repo,'rev-list','--parents','-n','1',commit).decode().split(),[commit,canonical['source_commit']],'freeze parent')
    same(git(repo,'diff-tree','--no-commit-id','--name-status','-r',commit).decode().splitlines(),['A\t'+FREEZE],'freeze delta')
    git(repo,'merge-base','--is-ancestor',commit,'HEAD')
    for relative,expected in {**canonical['files'],FREEZE:digest(raw)}.items():
        path=repo/relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'safe frozen path')
        require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular frozen file')
        for data in (path.read_bytes(),git(repo,'show',':'+relative),git(repo,'show','HEAD:'+relative)):
            require(digest(data)==expected,'current/index/HEAD frozen bytes '+relative)
        if relative!=FREEZE: require(digest(git(repo,'show',canonical['source_commit']+':'+relative))==expected,'source frozen bytes')
    return digest(raw)


def outer_audit(folder,raw_lock):
    execution=(folder/'execution.json').read_bytes(); e=json.loads(execution); b=load(folder/'collection-binding.json')
    same(b['schema'],'evr0009.collection-binding.v1','outer binding schema')
    same(b['execution_sha256'],digest(execution),'outer execution binding')
    same(b['selection_lock'],None if raw_lock is None else {'bytes':len(raw_lock),'sha256':digest(raw_lock)},'outer selection-lock binding')
    same(b['lock_status'],'missing_or_unsafe' if raw_lock is None else 'present','outer lock status')
    same(e,load(folder/'command.json'),'outer command fields')
    for name,meta in e['logs'].items():
        if meta is not None:
            data=(folder/name).read_bytes(); same(meta,{'bytes':len(data),'sha256':digest(data)},'outer log')
    require(e['wall_limit_seconds']==900 and e['address_space_bytes']==8589934592,'outer limits')
    require(1<=len(e['cpu_affinity'])<=4 and len(set(e['cpu_affinity']))==len(e['cpu_affinity']),'outer affinity')
    require(e['numeric_threads']==4,'outer thread limit')
    expected_exit=125 if e['child_exit_code']==0 and raw_lock is None else e['supervisor_exit_code']
    same(b['effective_exit_code'],expected_exit,'effective outer exit')
    good=(e['child_exit_code']==0 and e['raw_wait_status']==0 and not e['timed_out'] and not e['supervisor_interrupted'] and e['supervisor_exit_code']==0 and b['effective_exit_code']==0 and e.get('failure_type') is None and e.get('reap_error') is None and 0<=e['wall_seconds']<=900 and 0<=e['max_rss_kib']<=8*1024**2)
    return bool(good),b['effective_exit_code']


def command_audit(execution,repo,folder,freeze):
    argv=execution['argv']; cwd=Path(execution['cwd'])
    require(isinstance(argv,list) and len(argv)==7 and all(isinstance(a,str) for a in argv),'collector argv shape')
    require(bool(argv[0]) and Path(argv[0]).is_absolute() and cwd.is_absolute(),'recorded interpreter/cwd paths')
    same(argv[2:6],['collect','--freeze-commit',freeze,'--output'],'collector command/freeze')
    require((cwd/argv[1]).resolve()==(repo/'scripts/run_geometric_stage0.py').resolve(),'collector script path')
    require((cwd/argv[6]).resolve()==folder.resolve(),'collector scientific output path')


def review(repo,folder,evaluation=None,supervision=None):
    require(not any(p.is_symlink() for p in (folder/'selection-lock.json',folder,*folder.parents)),'regular lock path')
    raw_lock=(folder/'selection-lock.json').read_bytes() if (folder/'selection-lock.json').is_file() else None
    outer_ok,outer_code=outer_audit(supervision,raw_lock) if supervision else (False,None)
    if raw_lock is None:
        require(supervision is not None and outer_code!=0,'missing lock needs failed outer evidence')
        return {'decision':'NO_GO','review_status':'INCOMPLETE_NO_LOCK','outer_effective_exit':outer_code,'arithmetic_replayed':False}
    lock=json.loads(raw_lock); require(lock['schema']=='evr0009.selection-lock.v1','lock schema')
    files=lock['files']; actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual==set(files)|{'selection-lock.json'},'lock exact inventory')
    for rel,h in files.items():
        path=folder/rel; require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'safe lock path')
        require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'regular locked file')
        require(digest(path.read_bytes())==h,'locked hash '+rel)
    collection=load(folder/'collection.json'); binding=collection['binding']; binding_audit(repo,binding)
    if supervision: command_audit(load(supervision/'execution.json'),repo,folder,binding['freeze_commit'])
    same(lock['freeze_commit'],binding['freeze_commit'],'lock freeze'); same(collection['case_order'],order(),'case order')
    same(load(folder/'planned-cases.json')['case_ids'],order(),'planned IDs')
    records={}; inventory=[]
    for cid in order():
        require('cases/'+cid+'.json' in files,'locked case missing')
        rec=load(folder/'cases'/f'{cid}.json'); family,s=cid.rsplit('-',1)
        same(rec['case_id'],cid,'case ID'); same(rec['family'],family,'case family'); same(rec['seed_index'],int(s),'case seed')
        if rec['status']!='unattempted':
            seeds={name:int.from_bytes(hashlib.sha256(f'evr0009|stage0|{family}|{int(s)}|{name}'.encode()).digest()[:8],'little') for name in STREAMS}
            # Only hash-derived metadata: no PCG64 state or random draw is generated.
            require(rec.get('rng_seeds')==seeds if 'inputs' in rec else rec.get('rng_seeds') in ({},seeds),'stream seed metadata')
            same(rec['case_limit_seconds'],2.5,'case time cap')
        label_path=folder/'labels'/f'{cid}.json'; label=load(label_path) if label_path.is_file() else None
        if any(v.get('selected_id') is not None for v in rec.get('selections',{}).values()): require(label is not None,'selection without labels')
        case_audit(rec,label); records[cid]=rec; inventory.append({'case_id':cid,'status':rec['status']})
    same(collection['inventory'],inventory,'collection inventory')
    complete=all(r['status']=='completed' for r in records.values())
    same(collection['selection_complete'],complete,'collection complete'); same(lock['selection_complete'],complete,'lock complete')
    outcome=aggregate(records)
    case_resources=all(r['status']!='completed' or 0<=r['active_seconds']<=2.5 for r in records.values())
    if evaluation:
        saved=load(evaluation/'evaluation.json'); same(saved,outcome,'evaluation arithmetic')
        same(saved['selection_lock_sha256'],digest(raw_lock),'evaluation lock hash')
        left,right=dict(saved['binding']),dict(binding); left.pop('execution_head',None);right.pop('execution_head',None)
        same(left,right,'evaluation binding')
    return {'decision':outcome['gates']['decision'] if outer_ok and case_resources else 'NO_GO','review_status':'PASS','arithmetic_replayed':True,'selection_lock_sha256':digest(raw_lock),'outer_resource_evidence_pass':outer_ok,'case_resource_evidence_pass':case_resources,'gates':outcome['gates'],'success_counts':outcome['success_counts'],'status_counts':dict(Counter(r['status'] for r in records.values()))}


def artificial_fixture(failed=False):
    # Three hand-written observations: two distinct exact matches and one clone.
    inputs={'raw_x':[[10.,10.],[50.,50.],[10.,10.]],'raw_y':[[10.,10.],[50.,50.],[10.,10.]],
            'priorities':[.2,.1,.3],'candidates':pool(),'candidate_order':list(range(31,-1,-1))}
    rec={'status':'completed','controls_complete':True,'tail_attempted':32,'inputs':inputs,
         'input_hashes':{k:ah(v,'i' if k=='candidate_order' else 'f') for k,v in inputs.items()},
         'retained_ids':[1,0],'retained_x':[[50.,50.],[10.,10.]],'retained_y':[[50.,50.],[10.,10.]],
         'retained_hashes':{'x':ah([[50.,50.],[10.,10.]]),'y':ah([[50.,50.],[10.,10.]])},
         'dedup_decisions':[{'raw_id':1,'accepted':True,'reason':None},{'raw_id':0,'accepted':True,'reason':None},{'raw_id':2,'accepted':False,'reason':'both'}],
         'cell_counts':{'source':[2,2],'target':[2,2]},'hypotheses':[],
         'selections':{m:{'selected_id':4,'tied_ids':[4]} for m in METHODS}}
    for cid in inputs['candidate_order']:
        hit=cid==4; a=np.eye(2,dtype=bool) if hit else np.zeros((2,2),bool)
        rec['hypotheses'].append({'canonical_id':cid,'n':2,'N':2 if hit else 0,'S':2 if hit else 0,
            'H_sha256':ah(pool()[cid]),'A_sha256':ah(a,'b'),
            'graph_components':[{'rows':1,'columns':1,'edges':1}]*2 if hit else [],
            'raw_key':2 if hit else 0,'msac_key':2. if hit else 0.,'cell_key':1. if hit else 0.,
            'excess_key':2 if hit else 0,'excess':1. if hit else 0.,'mu':1. if hit else 0.,
            'ratio_key':[4,2] if hit else [0,1],'ratio':2. if hit else 0.,
            'tail':{'status':'completed','numerator':'1' if hit else '2','denominator':'2','operations':1,
                    'components':[[1,1,1,'complete-biclique']]*2 if hit else []}})
    if failed:
        rec.update(status='tail_failed',reason='artificial unsupported',tail_attempted=1)
        for i,h in enumerate(rec['hypotheses']):
            h['tail']={'status':'unsupported','reason':'artificial component cap','operations':1} if i==0 else {'status':'not_attempted'}
        rec['selections']['tail']={'selected_id':None,'tied_ids':[]}
    return rec,{'H_star':np.eye(3).tolist(),'true_candidate_id':4}


class SavedFixtureTests(unittest.TestCase):
    def stored(self,rec):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        p=Path(temp.name)/'case.json'; p.write_text(json.dumps(rec)); return load(p)

    def test_handwritten_complete_dedup_and_ranking(self):
        rec,label=artificial_fixture()
        self.assertTrue(case_audit(self.stored(rec),label,fixture=True))
        self.assertEqual(dedup(np.array([[0.,0.],[1.,0.],[2.01,0.]]),np.array([[0.,0.],[10.,0.],[20.,0.]]),[.1,.2,.3])[0],[0,2])
        # Literal operational rounding boundary; direct x+t would accept A00.
        rows,_=scalar_rows(np.array([[50.7,50.7],[300.,300.]]),np.array([[167.7,170.7],[20.,20.]]),pool())
        self.assertEqual(rows[9]['N'],0)  # Translation (120,120).
        self.assertEqual(rows[9]['A_sha256'],ah(np.zeros((2,2),bool),'b'))

    def test_handwritten_unsupported_is_valid_failure_and_gates(self):
        rec,label=artificial_fixture(True)
        self.assertFalse(case_audit(self.stored(rec),label,fixture=True))
        counts={f:{m:40 for m in METHODS} for f in FAMILIES}
        counts['clustered_outlier']['tail']=48
        paired={f:{m:{'baseline_success_tail_failure':0} for m in CONTROLS} for f in FAMILIES}
        self.assertEqual(gates(counts,paired,True)['decision'],'ELIGIBLE_PENDING_EXTERNAL_REVIEW')
        self.assertEqual(gates(counts,paired,False)['decision'],'NO_GO')
        paired['compact_true']['cell']['baseline_success_tail_failure']=3
        self.assertEqual(gates(counts,paired,True)['decision'],'NO_GO')

    def test_handwritten_tampered_saved_values_rejected(self):
        for mutation in ('score','hash','selection','zero-tail'):
            rec,label=artificial_fixture()
            if mutation=='score': rec['hypotheses'][0]['excess_key']+=1
            if mutation=='hash': rec['hypotheses'][0]['A_sha256']='0'*64
            if mutation=='selection': rec['selections']['raw']['selected_id']=0
            if mutation=='zero-tail': rec['hypotheses'][0]['tail']['numerator']='0'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                case_audit(self.stored(rec),label,fixture=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); folder=root/'collection'; freeze='a'*40
            e={'cwd':str(root),'argv':['/usr/bin/python3',str(root/'scripts/run_geometric_stage0.py'),'collect','--freeze-commit',freeze,'--output',str(folder)]}
            command_audit(e,root,folder,freeze)
            for i,value in ((1,str(root/'different.py')),(4,'b'*40),(6,str(root/'other-output'))):
                bad=json.loads(json.dumps(e));bad['argv'][i]=value
                with self.subTest(argv_index=i),self.assertRaises(ValueError):
                    command_audit(bad,root,folder,freeze)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--repo',type=Path); parser.add_argument('--input',type=Path)
    parser.add_argument('--evaluation',type=Path); parser.add_argument('--supervision',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SavedFixtureTests))
        print(json.dumps({'self_test_groups':result.testsRun,'passed':result.wasSuccessful(),'scope':'handwritten artificial saved records only'}))
        return 0 if result.wasSuccessful() else 1
    if args.repo is None or args.input is None or args.report is None:
        parser.error('--repo, --input and --report are required')
    for directory in (args.input,args.evaluation,args.supervision):
        if directory is not None:
            require(directory.resolve()!=args.report.resolve() and directory.resolve() not in args.report.resolve().parents,'report cannot alter reviewed artifacts')
    try:
        result=review(args.repo.resolve(),args.input.resolve(),args.evaluation.resolve() if args.evaluation else None,args.supervision.resolve() if args.supervision else None)
        code=0
    except Exception as exc:
        result={'review_status':'MISMATCH','error_type':type(exc).__name__,'reason':str(exc)}; code=1
    result.update(checks=CHECKS,numeric_comparisons=NUMERIC,max_absolute_difference=MAX_DIFFERENCE,
        atol=1e-10,rtol=1e-12,reviewer_sha256=digest(Path(__file__).read_bytes()),
        tail_counts_recomputed=False,rng_draws_replayed=False,
        scope='Saved-coordinate/control/selection/gate arithmetic only; exact tail counts bounded and ranked, not independently recounted; external exits are archived evidence, not new tool observations.')
    with args.report.open('x') as stream: stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'review_status':result['review_status'],'decision':result.get('decision'),'checks':CHECKS}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
