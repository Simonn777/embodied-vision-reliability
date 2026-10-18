"""Publication-only text archive from immutable saved outputs; no science imports."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'research-0008-training';STUDY=ROOT/'antithetic-training-0008';CONTROL=ROOT/'scratch/0008-control';PUB=Path(__file__).parent;DEST=REPO/'results/0008-antithetic-training'
PYTHON='<PYTHON>'
BASE='74f9330014aa8a13a8c55785992e115f045f2fa9'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    assert not DEST.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    pins={STUDY/'scores.json':'810cfd9384c475dc0df3d048bcd98a5bd33f96993e70e15d6ef69d68e40943e7',CONTROL/'independent-replay.json':'75247b4e4d723ab8f7bc4a68d0a3ec3ba7f3641a6aa5733eb97e53128b24d716',CONTROL/'postcheck.json':'34c911ea286e765ed579d7078dd59dd0a3b2c33c2e58c7f08b120c797db0009c'}
    for p,h in pins.items():assert sha(p)==h,p.name
    before=read(CONTROL/'pre-execution.json');score=read(STUDY/'scores.json');post=read(CONTROL/'postcheck.json')
    assert score['status']=='stop_this_candidate' and post['status']=='PASS'
    # Snapshot ORIGINAL control bytes before any public transformations.
    original={str(p.relative_to(ROOT)):sha(p) for p in CONTROL.iterdir() if p.is_file()}
    original.update({str(p.relative_to(ROOT)):sha(p) for p in STUDY.rglob('*') if p.is_file()})
    write(PUB/'original-inventory.json',{'files':original,'scope':'Saved 0008 controls and all study outputs, including private binaries; bytes hashed only, no arrays decoded.'})
    for n,h in before['tracked'].items():assert sha(REPO/n)==h,n
    for tree,files in before['historical_trees'].items():
        for n,h in files.items():assert sha(ROOT/tree/n)==h,(tree,n)
    for n,h in before['markers'].items():assert sha(ROOT/n)==h,n
    DEST.mkdir();entries=[]
    def archive(src,relative,role='saved_evidence',force_recipe=False):
        raw=src.read_bytes();text=raw.decode('utf-8')
        public=text.replace(PYTHON,'<PYTHON>').replace(str(ROOT),'<PROJECT_ROOT>')
        dst=DEST/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(public.encode('utf-8'))
        scopes=[]
        if PYTHON in text:scopes.append('existing interpreter path replaced by <PYTHON>')
        if str(ROOT) in text:scopes.append('original project absolute prefix replaced by <PROJECT_ROOT>')
        entries.append({'original_source_relative':str(src.relative_to(ROOT)),'original_sha256':sha(src),'public_path':str(dst.relative_to(REPO)),'public_sha256':sha(dst),'original_bytes':len(raw),'public_bytes':dst.stat().st_size,'mode':'redacted_copy' if scopes else 'byte_exact_copy','role':role,'redaction_scope':scopes,'recipe_not_private_argv':force_recipe})
    for p in sorted(STUDY.glob('*.json')):archive(p,Path('study')/p.name)
    for p in sorted(STUDY.glob('cases/*/*.json')):archive(p,Path('trajectories')/p.parent.name/p.name)
    for p in sorted((STUDY/'control').iterdir()):
        if not p.is_file():continue
        if p.name.endswith('.command.private.json'):archive(p,Path('trajectory-control')/p.name.replace('.command.private.json','.command.recipe.json'),'redacted_command_recipe',True)
        elif p.suffix in ('.json','.log'):archive(p,Path('trajectory-control')/p.name)
    for p in sorted(CONTROL.iterdir()):
        if not p.is_file():continue
        if p.name.endswith('.command.private.json'):archive(p,Path('outer-control')/p.name.replace('.command.private.json','.command.recipe.json'),'redacted_command_recipe',True)
        elif p.name in ('outer.py','preflight.py','postcheck.py'):archive(p,Path('outer-control')/p.name.replace('.py','.recipe.py'),'execution_source_recipe')
        elif p.suffix in ('.json','.log'):archive(p,Path('outer-control')/p.name)
    compact=CONTROL/'compact-execution.json';raw=compact.read_bytes()
    parsed=json.loads(raw) # Original is valid; no repair or normalization.
    write(DEST/'summary-format-review.json',{'original_source_relative':str(compact.relative_to(ROOT)),'original_sha256':sha(compact),'original_bytes':len(raw),'original_json_parse':'PASS','tail_hex':raw[-12:].hex(),'ends_in_literal_backslash_n':raw.endswith(b'\\n'),'normalization_performed':False,'scope':'Only this saved summary was suspected of formatting damage. No claim is made about unselected private files. Public path redactions are tracked separately.'})
    assert len(post['saved_output_hashes'])==128
    for rel,h in post['saved_output_hashes'].items():assert sha(STUDY/rel)==h,rel
    write(DEST/'original-output-hashes.json',{'source_relative':'scratch/0008-control/postcheck.json','source_sha256':sha(CONTROL/'postcheck.json'),'kind':'derived_hash_inventory_no_binary_payload','files':post['saved_output_hashes'],'file_count':128,'array_decodes':0})
    freeze=read(REPO/'protocols/0008-training-freeze.json')
    write(DEST/'source-bindings.json',{'execution_main':BASE,'scientific_source_commit':freeze['source_commit'],'canonical_freeze_commit':'5680e6dae68ed102147ca89eeb115010cfe13687','canonical_freeze_relative':'protocols/0008-training-freeze.json','canonical_sha256':sha(REPO/'protocols/0008-training-freeze.json'),'required_inputs':freeze['files'],'original_control_source_hashes':{n:sha(CONTROL/n) for n in ('outer.py','preflight.py','postcheck.py')},'postcheck_sha256':sha(CONTROL/'postcheck.json'),'scores_sha256':sha(STUDY/'scores.json'),'independent_report_sha256':sha(CONTROL/'independent-replay.json'),'new_asset_bytes':0,'historical_asset_ceiling_bytes':7532690072,'publication_branch':'research/0008-training-results','actual_tool_observation':'Parent retrieved execution-turn tool records and independently confirmed all execution phases exit 0; archived wait4/log evidence retains its own scope.'})
    # All numerical rendering below reads SAVED JSON only; no scoring is called.
    md=['# 0008: mean pairing failed the frozen training-quality screen','',
        '**Decision: `stop_this_candidate`.** All 32 trajectories and 64 predictions completed once. Both frozen quality gates failed. This exact mean-sort candidate is stopped; the broader research objective remains active. No manuscript, new method, novelty or quality-success claim follows. AI-assisted execution and reporting.','',
        '## Fixed results, including harms','',
        'The controls are independent, random antithetic and same-patch. The comparator is the per-case PSNR-best control (an oracle, not a deployable selector). All four images, both corruption seeds, methods and endpoints are retained below. Values are rendered from the unchanged [formal scores](study/scores.json), never recomputed by a model or scorer during publication.','']
    methods=score['methods'];rows=score['cases']
    for endpoint in ('updates500','wall5s'):
        md+=['### '+endpoint,'','| Case | Independent PSNR | Random PSNR | Mean PSNR | Same-patch PSNR | Oracle control | Mean − oracle dB |','|---|---:|---:|---:|---:|---|---:|']
        for pair in score['endpoints'][endpoint]['pairs']:
            g,seed=pair['group'],pair['seed'];ps={r['method']:r['score']['psnr_db'] for r in rows if r['endpoint']==endpoint and r['group']==g and r['seed']==seed}
            md.append('| %s seed %d | %s | %s | %.9f |'%(g,seed,' | '.join('%.9f'%ps[m] for m in methods),pair['oracle_best_control'],pair['gain_db']))
        md+=['']
    md+=['| Image token | 500-update mean gain dB | 5-second mean gain dB |','|---|---:|---:|']
    for group in score['groups']:md.append('| %s | %.9f | %.9f |'%(group,score['endpoints']['updates500']['group_mean_gain_db'][group],score['endpoints']['wall5s']['group_mean_gain_db'][group]))
    md+=['','| Frozen gate | 500 updates | 5 seconds |','|---|---|---|']
    for key in score['endpoints']['updates500']['gates']:
        vals=[score['endpoints'][e]['gates'][key] for e in ('updates500','wall5s')]
        md.append('| `%s` | %s | %s |'%(key,*('PASS' if v else 'FAIL' for v in vals)))
    md+=['','Mean gain is **−0.0009555460164949459 dB** at 500 updates and **−0.3300652878587962 dB** at 5 seconds, below the unchanged +0.5 dB requirement. Positive image means are 1/4 and 0/4; the minimum image means are −0.4196137493 and −1.0892817462 dB. The ≥3-positive-images and every-image ≥−0.2 dB gates fail at both endpoints. Texture ratios are %.9f and %.9f, both ≤1.05. The wall5s total mean/independent allocated outer-cost ratio is **%.16f**, passing ≤1.10. Passing texture/cost cannot rescue failed quality.'%(score['endpoints']['updates500']['mean_candidate_first_difference_mse']/score['endpoints']['updates500']['mean_corresponding_control_first_difference_mse'],score['endpoints']['wall5s']['mean_candidate_first_difference_mse']/score['endpoints']['wall5s']['mean_corresponding_control_first_difference_mse'],score['wall5s_candidate_vs_independent_pipeline_ratio']),'',
        '## Real execution and cost accounting','',
        'Execution main `74f9330014aa8a13a8c55785992e115f045f2fa9`; source `829688df750964d796417da569f0710453f62ca7`; canonical freeze `5680e6dae68ed102147ca89eeb115010cfe13687`. All 13 required inputs remain frozen. Publication uses this same saved execution, not the later result commit.','',
        'Pinned cloud software tests: **51 passed, zero skips** (7.049 seconds unittest; 7.754978 seconds outer), plus **16 independent artificial self-test groups**. CI Torch skips were not counted as runtime execution. The scientific outer wait4 returned exit 0 / raw status 0, **260.1861582160018 seconds**, peak RSS **304,904 KiB**. Actual training across shared trajectories used **98,994 optimizer updates**. All 64 MC50 predictions account for **48,000 forwards and 2,333,811,200 normal scalar draws**. No warmup or repeated trajectory occurred. All wall captures were between 5.000254871 and 5.003422856 seconds, within the original 5.05-second capture bound.','',
        'CPU Python 3.12.14, NumPy 2.0.2 and Torch 2.10.0+cpu on AMD EPYC 9V74 / Linux 6.18.55; four affinity slots [0,1,2,3], numeric threads 4, Torch interop 1, deterministic CPU and 8 GiB address space. Whole-study hard cap 900 seconds, case external 120 / inner training 90 seconds, audit/score/checker outer caps 120 seconds. All nine actual outer stage exit/raw wait records are 0.','',
        'The full dual-endpoint trajectory wall includes runtime/source checks, shared setup, training, both inference calls, I/O and shutdown. Endpoint **allocated outer costs** subtract only measured extra training after that endpoint and the other endpoint’s weight-write/restore/infer/save work. They are not independently timed standalone runs. Both endpoint costs are correlated allocations from one trajectory, and earlier snapshot overhead remains charged. Full case wall and all recorded components are preserved in the 32 receipts and [trajectory cost table](trajectory-costs.json).','',
        '| Allocated outer seconds, summed over eight cases | Independent | Random | Mean | Same-patch |','|---|---:|---:|---:|---:|']
    for endpoint in ('updates500','wall5s'):
        costs=score['endpoints'][endpoint]['pipeline_seconds_by_method'];md.append('| %s | %s |'%(endpoint,' | '.join('%.9f'%costs[m] for m in methods)))
    md+=['','Audit passed after all 64 predictions locked and before the isolated scorer first opened four pinned clean references. Formal score ran once, outer exit 0, 0.324616 seconds, peak RSS 34,080 KiB. Independent saved-array replay used no production metric/Torch import or model: **45,770 checks / 5,287 numeric**, max absolute difference **7.105427357601002e-15**, fixed rtol 1e-10 / atol 1e-12, exit 0, 0.505234 seconds outer, peak RSS 32,444 KiB. It reproduces `stop_this_candidate`.','',
        '## Evidence, privacy and limitations','',
        '- [Copy/redaction manifest](copy-manifest.json): original project-relative source identity, original SHA/size, public SHA/size and exact transformation scope for every archived file. Command recipes never claim to be original private argv files.','- [Source/freeze bindings](source-bindings.json), [128 original output hashes](original-output-hashes.json), [before inventory](outer-control/pre-execution.json), [after check](outer-control/postcheck.json), [independent report](outer-control/independent-replay.json) and [nine outer stage records](outer-control/compact-execution.json). Full case receipts, started markers, closed stdout/stderr and wait records are in trajectories/ and trajectory-control/.','- The original compact summary **parses as valid JSON**, ends in a real newline (`0a`), and required no normalization. [Exact formatting review](summary-format-review.json) records its original hash; its public path substitutions are distinct from normalization. No claim is made that every unselected private artifact is valid JSON.','- All 2,249 prior public result files and 2,477 original tracked files were preserved except the four explicitly authorized README/state/note edits. Old protocols/source freezes remain unchanged. The 496 non-reference 0006 cache files, 84 original 0007 output files, all 407 failed-cache files and 22 old marker files remain byte-exact; four references were first checked/opened at the permitted scoring phase. 0006’s stdout gap remains **OPEN**, its candidate stays failed, and 0003 remains deferred.','- Raw images/references, noisy arrays, full predictions and trained weights remain private. Only non-reconstructable block SSE/counts, scalar metrics, logs, hashes and receipts are public. Original private absolute paths and command files stay private; `<PROJECT_ROOT>` / `<PYTHON>` are explicit placeholders.','- No scientific run, inference, formal scorer or independent replay was repeated for publication. New acquired asset bytes **0**; historical ceiling **7,532,690,072 bytes**, no added fees, package installation or schedule change.','- Four reused development image substrates, known synthetic covariance and a tiny MLP cannot establish independent-scene or real-camera efficacy, competitiveness against modern strong baselines, or population uncertainty. Generic antithetic novelty is already rejected. 0007’s variance pass did not establish a quality benefit.','- Future distinct hypotheses need a prospective rationale and untouched validation. Do not rescue this exact rule through hyperparameter, pairing-feature or budget tuning on these images or old tests. No manuscript is warranted.','',
        'Execution parents observe their timeout children via actual wait4; the parent itself has no recursively added outer observer. The parent task separately inspected the original tool returns. Publication-only validations do not replace original run evidence. Any publication-only exception is preserved separately.','']
    with (DEST/'README.md').open('x') as f:f.write('\n'.join(md))
    costrows=[]
    for item in read(STUDY/'study.completed.json')['cases']:
        r=read(STUDY/'cases'/item['case']/'receipt.json')
        costrows.append({'trajectory':item['case'],'actual_outer_wall_seconds':item['wall_seconds'],'peak_rss_kib':item['peak_rss_kib'],'actual_exit':item['actual_exit'],'input_seconds':r['input_seconds'],'shared_setup_seconds':r['fit']['shared_setup_seconds'],'training_wall_seconds':r['fit']['training_wall_seconds'],'completed_updates':r['fit']['completed_updates'],'endpoints':{e:{'allocated_outer_pipeline_seconds':next(v['pipeline_seconds'] for v in score['cases'] if v['group']==r['identity']['group'] and v['seed']==r['identity']['seed'] and v['method']==r['method'] and v['endpoint']==e),'training':d['training'],'weights_write_seconds':d['weights_write_seconds'],'restore_infer_save_seconds':d['restore_infer_save_seconds'],'algorithm_pipeline_seconds':d['algorithm_pipeline_seconds']} for e,d in r['endpoints'].items()}})
    write(DEST/'trajectory-costs.json',{'source_relative':'antithetic-training-0008/study.completed.json','source_sha256':sha(STUDY/'study.completed.json'),'kind':'saved_JSON_cost_summary_not_new_measurement','actual_case_outer_wall_sum_seconds':sum(r['actual_outer_wall_seconds'] for r in costrows),'study_outer_wall_seconds':260.1861582160018,'trajectories':costrows})
    write(DEST/'copy-manifest.json',{'schema':'0008-original-to-public-v1','entries':entries,'generated_derivatives':{'README.md':'Saved metrics/receipts rendered in English; no new score or model call.','trajectory-costs.json':'Saved scores and trajectory timing receipts joined by fixed identity.','original-output-hashes.json':'Original postcheck file-hash map, actual bytes rehashed only.','source-bindings.json':'Original canonical and saved-control identities.','summary-format-review.json':'Original summary parsed without modification; format observation only.'},'manifest_self_reference':'This manifest excludes itself. Public artifact inventory is written separately.','private_source_root':'<PROJECT_ROOT>','interpreter':'<PYTHON> is the existing pinned Python3.12.14 interpreter.'})
    print(json.dumps({'status':'archived_saved_text_only','archived_entries':len(entries),'public_files':sum(p.is_file() for p in DEST.rglob('*')),'public_bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'compact_original_json_valid':True,'normalization_performed':False,'scientific_calls':0}))
if __name__=='__main__':main()
