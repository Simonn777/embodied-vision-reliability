"""Read-only publication/provenance audit; no NumPy, Torch, production imports or science.

Only public text/metadata, Git objects and explicitly supplied parent tool records
are read. Private absolute paths are used in memory solely to check redactions.
They and original command text are never exported in the report.
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True, help='Local repository clone')
parser.add_argument('--output', type=Path, required=True, help='Review output directory outside the repository')
parser.add_argument('--observations', type=Path, required=True, help='Published redacted parent tool observations')
parser.add_argument('--private-execution', type=Path, required=True, help='Original private execution-tool retrieval; never published')
parser.add_argument('--prior-inventory', type=Path, required=True, help='Prior public result/protocol Git-blob inventory')
args = parser.parse_args()
REPO = args.repo.resolve()
OUT = args.output.resolve()
if OUT == REPO or REPO in OUT.parents:
    parser.error('--output must be outside the repository')
BASE = REPO / 'results/0008-antithetic-training'
EXPECTED_HEAD = '70bc57fd83634a8df66aecf072ea9b526e3af043'
OBS = args.observations
PRIVATE = args.private_execution
PRIOR = args.prior_inventory
checks = collections.Counter()
failures = []
notes = []

def check(ok, group, label):
    checks[group] += 1
    if not ok:
        failures.append({'group': group, 'check': label})

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p): return digest(p.read_bytes())
def git(*args): return subprocess.check_output(['git', *args], cwd=REPO)
def reject_constant(v): raise ValueError('nonfinite JSON constant')
def unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise ValueError('duplicate JSON key')
        out[k] = v
    return out
def loads(s): return json.loads(s, object_pairs_hook=unique_object, parse_constant=reject_constant)
def load(p): return loads(p.read_text())
def dt(s): return datetime.datetime.fromisoformat(s)

head = git('rev-parse', 'HEAD').decode().strip()
check(head == EXPECTED_HEAD, 'checkout', 'requested head')
tracked = git('status', '--porcelain').decode()
check(not tracked, 'checkout', 'clean starting checkout')
before = load(BASE/'outer-control/pre-execution.json')
post = load(BASE/'outer-control/postcheck.json')
bindings = load(BASE/'source-bindings.json')
freeze = load(REPO/'protocols/0008-training-freeze.json')
manifest = load(BASE/'copy-manifest.json')
hashes = load(BASE/'artifact-hashes.json')['files']
inventory = load(BASE/'publication-control/original-inventory.json')['files']
observations = load(OBS)
private = load(PRIVATE)
prior = load(PRIOR)

# Verify the caller's extracted observations against original commandExecution
# records without exporting private command strings or thread identifiers.
check(sha(PRIVATE) == observations['private_retrieval_sha256'], 'tool_records', 'private retrieval digest')
items = {x['id']: x for x in private['turn']['items'] if x['type'] == 'commandExecution'}
check(len(items) == 18 == len(observations['commands']), 'tool_records', '18 commands')
mapping_values = collections.defaultdict(set)
observed_stages = {}
for c in observations['commands']:
    raw = items[c['tool_item_id']]
    check(digest(raw['command'].encode()) == c['original_command_text_sha256'], 'tool_records', 'original command digest')
    check(raw['status'] == c['status'] and raw['exitCode'] == c['exit_code'] and raw['durationMs'] == c['tool_duration_ms'], 'tool_records', 'tool status exit duration')
    check(bool(raw.get('output', {}).get('truncated', False)) == c['output_truncated'], 'tool_records', 'truncation disclosure')
    pieces = re.split(r'(<PROJECT_ROOT>|<PROJECT>|<PYTHON>)', c['command_recipe_redacted'])
    pattern = ''.join('(.*?)' if x in ('<PROJECT_ROOT>', '<PROJECT>', '<PYTHON>') else re.escape(x) for x in pieces)
    match = re.fullmatch(pattern, raw['command'], re.DOTALL)
    check(match is not None, 'redaction', 'tool command reversible placeholders')
    if match:
        for token, value in zip([x for x in pieces if x in ('<PROJECT_ROOT>', '<PROJECT>', '<PYTHON>')], match.groups()):
            mapping_values['<PROJECT_ROOT>' if token == '<PROJECT>' else token].add(value)
    actual_records = []
    if not raw.get('output', {}).get('truncated', False):
        for line in raw.get('output', {}).get('text', '').splitlines():
            try: obj = loads(line)
            except (ValueError, TypeError): continue
            if isinstance(obj, dict) and {'label', 'actual_exit', 'raw_wait_status', 'outer_source_sha256'} <= set(obj): actual_records.append(obj)
    check(actual_records == c['observed_child_records'], 'tool_records', 'child records extracted from complete original output')
    for rec in actual_records:
        check(rec['label'] not in observed_stages, 'tool_records', 'unique observed stage')
        observed_stages[rec['label']] = rec
mapping = {k: next(iter(v)) for k, v in mapping_values.items() if len(v) == 1}
check(set(mapping) == {'<PROJECT_ROOT>', '<PYTHON>'}, 'redaction', 'two unambiguous private substitutions')
check(set(observed_stages) == set(observations['observed_stages']) and len(observed_stages) == 9, 'tool_records', 'nine complete observed stages')

# Hash every public artifact and every declared entry, including publication-only
# failures. Derivatives are identified rather than called original copies.
files = {str(p.relative_to(BASE)): p for p in BASE.rglob('*') if p.is_file()}
check(len(files) == 265, 'archive', '265 files')
check(set(hashes) == set(files)-{'artifact-hashes.json'}, 'archive', 'complete artifact hash coverage except self')
for n, h in hashes.items(): check(sha(files[n]) == h, 'archive', 'artifact hash '+n)
entries = {e['public_path']: e for e in manifest['entries']}
check(len(entries) == len(manifest['entries']) == 263, 'archive', '263 unique copy or derivative entries')
check(set(entries) == {str(p.relative_to(REPO)) for n,p in files.items() if n not in ('artifact-hashes.json','copy-manifest.json')}, 'archive', 'manifest coverage')
reversed_count = 0
selective_source_reversals = 0
redaction_not_reversed = []
for rel, e in entries.items():
    p = REPO/rel
    check(sha(p) == e['public_sha256'] and p.stat().st_size == e['public_bytes'], 'manifest', 'public hash length '+rel)
    if e['original_source_relative'] in inventory:
        check(inventory[e['original_source_relative']] == e['original_sha256'], 'manifest', 'original inventory link '+rel)
    if e['mode'] == 'byte_exact_copy':
        check(e['public_sha256'] == e['original_sha256'] and e['public_bytes'] == e['original_bytes'] and not e['redaction_scope'], 'manifest', 'byte-exact declaration '+rel)
    elif e['mode'] == 'redacted_copy':
        check(bool(e['redaction_scope']), 'manifest', 'redaction disclosed '+rel)
        text = p.read_text()
        original_candidate = text
        for placeholder, original in mapping.items(): original_candidate = original_candidate.replace(placeholder, original)
        candidate_bytes = original_candidate.encode()
        if digest(candidate_bytes) == e['original_sha256'] and len(candidate_bytes) == e['original_bytes']:
            reversed_count += 1
        else:
            # Publication scripts contain literal placeholder destinations in
            # addition to their redacted PYTHON assignment. Reverse only that
            # one assignment; never rewrite those original destination strings.
            selective = text.replace("PYTHON='<PYTHON>'", "PYTHON='"+mapping['<PYTHON>']+"'", 1).encode()
            if digest(selective) == e['original_sha256'] and len(selective) == e['original_bytes']:
                reversed_count += 1
                selective_source_reversals += 1
            else:
                redaction_not_reversed.append(str(p.relative_to(BASE)))
    else:
        check(e['mode'] == 'derived_summary' and bool(e.get('original_sources')), 'manifest', 'identified derived summary '+rel)
    if rel.endswith('.command.recipe.json'):
        check(e['recipe_not_private_argv'] is True, 'manifest', 'recipe label '+rel)

# All prior public result/protocol Git blobs and their working tree bytes.
tree = {}
for row in git('ls-tree','-rz','HEAD').split(b'\0'):
    if row:
        metadata, path = row.split(b'\t',1)
        tree[path.decode()] = metadata.split()[2].decode()
check(len(prior['files']) == 2285, 'prior_blobs', '2285 prior records')
for path, blob in prior['files'].items():
    check(tree.get(path) == blob, 'prior_blobs', 'Git object '+path)
    raw = (REPO/path).read_bytes()
    check(hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest() == blob, 'prior_blobs', 'worktree blob '+path)
mutable = {'README.md','CURRENT_STATE.md','NEXT_ACTIONS.md','notes/2026-10-18.md'}
check(len(before['tracked']) == 2477, 'history', '2477 original tracked files')
for path,h in before['tracked'].items():
    if path not in mutable: check(sha(REPO/path) == h, 'history', 'immutable tracked bytes '+path)
changed = set(git('diff','--name-only',bindings['execution_main'],'HEAD').decode().splitlines())
check(changed-set(str(p.relative_to(REPO)) for p in files.values()) == mutable, 'history', 'only four existing files modified')
baseline_note = git('show',bindings['execution_main']+':notes/2026-10-18.md').decode()
current_note = (REPO/'notes/2026-10-18.md').read_text()
check(current_note.startswith(baseline_note), 'history', 'date note append-only')
bad_link = '../../results/0007-gradient-screen/README.md'
check(bad_link in baseline_note and bad_link in current_note, 'history', 'disclosed unresolved historical link retained')

check(len(freeze['files']) == 13 and freeze['files'] == bindings['required_inputs'] == before['frozen_files'], 'freeze', '13 matching frozen inputs')
for path,h in freeze['files'].items():
    check(sha(REPO/path) == h, 'freeze', 'worktree frozen '+path)
    check(digest(git('show',freeze['source_commit']+':'+path)) == h, 'freeze', 'source commit '+path)
canonical = 'protocols/0008-training-freeze.json'
check(sha(REPO/canonical) == bindings['canonical_sha256'] == before['canonical_sha256'] == post['canonical_sha256'], 'freeze', 'canonical digest cross-bindings')
check(git('show',bindings['canonical_freeze_commit']+':'+canonical) == (REPO/canonical).read_bytes(), 'freeze', 'original canonical bytes')

# Actual nine outer tool records, original log hashes and recipes. Hash references
# in wait records refer to original private files, not placeholder recipes.
stage_order = observations['observed_stages']
last_time = None
for label in stage_order:
    rec = load(BASE/f'outer-control/{label}.execution.json')
    check(rec == observed_stages[label], 'outer_stages', 'direct tool equality '+label)
    check(rec['actual_exit'] == rec['raw_wait_status'] == 0, 'outer_stages', 'exit zero '+label)
    check(rec['execution_main'] == bindings['execution_main'], 'outer_stages', 'execution source '+label)
    check(rec['outer_source_sha256'] == bindings['original_control_source_hashes']['outer.py'], 'outer_stages', 'outer source '+label)
    for kind in ('stdout','stderr'):
        p = BASE/f'outer-control/{label}.{kind}.log'
        e = entries[str(p.relative_to(REPO))]
        check(rec[kind+'_sha256'] == e['original_sha256'], 'outer_stages', 'closed original log '+label+' '+kind)
    recipe = BASE/f'outer-control/{label}.command.recipe.json'
    check(rec['command_file_sha256'] == entries[str(recipe.relative_to(REPO))]['original_sha256'], 'outer_stages', 'private argv hash '+label)
    check(rec['cpu_affinity'] == [0,1,2,3] and rec['address_space_bytes'] == 8589934592, 'outer_stages', 'recorded resource bounds '+label)
    if last_time: check(dt(rec['utc_started']) >= last_time, 'outer_stages', 'serial stage order '+label)
    last_time = dt(rec['utc_completed'])

completed = load(BASE/'study/study.completed.json')
started = load(BASE/'study/study.started.json')
names = [r['case'] for r in completed['cases']]
check(len(names) == len(set(names)) == 32 and names == started['order'], 'trajectories', '32 unique trajectories in frozen runtime order')
check({p.name for p in (BASE/'trajectories').iterdir()} == set(names), 'trajectories', 'no omitted or extra receipt directories')
saved_outputs = load(BASE/'original-output-hashes.json')
check(saved_outputs['files'] == post['saved_output_hashes'] and saved_outputs['file_count'] == 128, 'trajectories', '128 output identities preserved')
output_refs = {}
last_end = None
for listed in completed['cases']:
    name = listed['case']
    execution = load(BASE/f'trajectory-control/{name}.execution.json')
    check(execution == {k:v for k,v in listed.items() if k!='case'}, 'trajectories', 'completion wait record '+name)
    check(execution['actual_exit'] == execution['raw_wait_status'] == 0 and not execution['hard_timeout'], 'trajectories', 'actual case completion '+name)
    if last_end: check(dt(execution['utc_started']) >= last_end, 'trajectories', 'sequential trajectory '+name)
    last_end = dt(execution['utc_completed'])
    receiptp = BASE/f'trajectories/{name}/receipt.json'
    receipt = load(receiptp)
    mark = load(BASE/f'trajectories/{name}/started.json')
    check(execution['receipt_sha256'] == sha(receiptp), 'trajectories', 'receipt file identity '+name)
    check(receipt['status'] == 'completed' and receipt['source_commit'] == freeze['source_commit'] and receipt['freeze_sha256'] == bindings['canonical_sha256'], 'trajectories', 'frozen case identity '+name)
    check(mark['case'] == receipt['identity'] and mark['method'] == receipt['method'], 'trajectories', 'one-shot marker '+name)
    for path,h in receipt['source_hashes'].items(): check(freeze['files'][path] == h, 'trajectories', 'case source pin '+name+' '+path)
    for kind in ('stdout','stderr'):
        p = BASE/f'trajectory-control/{name}.{kind}.log'
        check(execution[kind+'_sha256'] == entries[str(p.relative_to(REPO))]['original_sha256'], 'trajectories', 'closed case log '+name+' '+kind)
    recipep = BASE/f'trajectory-control/{name}.command.recipe.json'
    check(execution['command_file_sha256'] == entries[str(recipep.relative_to(REPO))]['original_sha256'], 'trajectories', 'case private command hash '+name)
    recipe = load(recipep)
    argv = recipe['argv']
    check(recipe['cwd'] == '<PROJECT_ROOT>/research-0008-training' and argv[0] == '<PYTHON>' and argv[2] == 'case', 'recipes', 'case command shape '+name)
    for flag, value in [('--case-index',str(receipt['identity']['index'])),('--method',receipt['method']),('--freeze-commit',bindings['canonical_freeze_commit']),('--cache','<PROJECT_ROOT>/conditional-residual-0006'),('--output','<PROJECT_ROOT>/antithetic-training-0008'),('--runtime-root','<PROJECT_ROOT>/bootstapir-cache/runtime-fixed')]:
        check(argv.count(flag)==1 and argv[argv.index(flag)+1] == value, 'recipes', 'case argument '+name+' '+flag)
    for n,h in receipt['files'].items(): output_refs['cases/'+name+'/'+n] = h
check(output_refs == saved_outputs['files'], 'trajectories', 'all endpoint weight and prediction file hashes bound to receipts')
format_review = load(BASE/'summary-format-review.json')
compact_text = (BASE/'outer-control/compact-execution.json').read_text()
for placeholder, original in mapping.items(): compact_text = compact_text.replace(placeholder,original)
compact_bytes = compact_text.encode()
loads(compact_text)
check(digest(compact_bytes)==format_review['original_sha256'] and len(compact_bytes)==format_review['original_bytes'], 'format', 'original compact summary hash and length')
check(compact_bytes[-12:].hex()==format_review['tail_hex'] and compact_bytes.endswith(b'\n') and not compact_bytes.endswith(b'\\n'), 'format', 'original compact exact terminal newline')
check(format_review['normalization_performed'] is False, 'format', 'no normalization claimed')

# Publication-only failed and successful validations remain distinct from the
# nine scientific/preparation stages and cannot be described as science reruns.
v1 = load(BASE/'publication-control/verify.execution.json')
v2 = load(BASE/'publication-control/verify-v2.execution.json')
check(v1['actual_exit']==1 and v1['raw_wait_status']==256 and v2['actual_exit']==v2['raw_wait_status']==0, 'publication', 'failed v1 and successful v2 retained')
check(bad_link in (BASE/'publication-control/verify.stderr.log').read_text(), 'publication', 'actual failed historical-link exception')
check('publication-only issue' in (BASE/'README.md').read_text(), 'publication', 'README distinguishes publication exception')

# Scope-limited privacy review, not a general guarantee. Check exact private path
# strings and known conversation identifiers without emitting those identifiers.
private_tokens = list(mapping.values()) + [str(private['thread'].get('id','')), str(private['turn'].get('id',''))]
json_count = 0
for n,p in files.items():
    raw = p.read_bytes()
    text = raw.decode('utf-8')
    check(b'\0' not in raw and p.suffix in {'.json','.md','.py','.log'}, 'privacy', 'text-only allowed extension '+n)
    check(not n.endswith('.private.json') and not any(token and token in text for token in private_tokens), 'privacy', 'no original paths or conversation IDs '+n)
    check(not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',text), 'privacy', 'credential pattern '+n)
    if p.suffix=='.json': load(p); json_count += 1

check(git('rev-parse','HEAD').decode().strip()==head, 'checkout', 'head unchanged during audit')
check(git('status','--porcelain').decode()==tracked, 'checkout', 'worktree unchanged during audit')
report = {
 'schema':'0008-independent-parent-provenance-v1',
 'status':'PASS' if not failures else 'BLOCKED',
 'head':head,
 'scope':'Local public-file and Git preservation, saved receipt consistency, and independent comparison with original parent-retrieved execution tool records. No scientific recomputation or cloud access.',
 'auditor_source_sha256':sha(Path(__file__)),
 'input_evidence_sha256':{'parent_tool_observations':sha(OBS),'prior_public_inventory':sha(PRIOR),'private_execution_retrieval':sha(PRIVATE),'public_copy_manifest':sha(BASE/'copy-manifest.json'),'artifact_hashes':sha(BASE/'artifact-hashes.json')},
 'checks_by_group':dict(checks),'total_checks':sum(checks.values()),'failures':failures,
 'counts':{'archive_files':len(files),'artifact_hash_entries':len(hashes),'manifest_entries':len(entries),'manifest_modes':dict(collections.Counter(e['mode'] for e in entries.values())),'redacted_copies_inverse_hash_verified':reversed_count,'selective_source_assignment_inversions':selective_source_reversals,'redacted_copies_not_unambiguously_inverted':len(redaction_not_reversed),'public_json_validated':json_count,'prior_result_protocol_Git_blobs':len(prior['files']),'old_tracked_worktree_files_checked_excluding_four_mutable':len(before['tracked'])-4,'frozen_inputs':13,'canonical_freeze_receipt':1,'original_command_tool_records':len(items),'complete_observed_stages':len(observed_stages),'trajectories':len(names),'closed_trajectory_logs':2*len(names),'private_output_file_hash_references':len(output_refs)},
 'redaction_inverse_limit_files':redaction_not_reversed,
 'original_9_stage_records_equal_to_archived':True if all(load(BASE/f'outer-control/{n}.execution.json')==r for n,r in observed_stages.items()) else False,
 'historical_private_cache_counts_claimed_by_cloud_postcheck':post['historical_tree_counts_preserved'],
 'historical_private_marker_count_claimed_by_cloud_postcheck':post['old_markers_preserved'],
 'limitations':[
 'No private cache/data/weight/prediction bytes were accessed or rehashed locally. Their preservation is the scope of the observed cloud postcheck and its saved inventory, not fresh parent verification.',
 'Output hashes establish identity links but do not expose private preimages. No score, gradient, PSNR, bootstrap, model inference or training was rerun.',
 'Original command tool records independently support nine stage outcomes. They do not instrument every library syscall or prove absence of arbitrary hidden activity.',
 'All 52 redacted copies were restored in memory to their declared original hash and length. Four publication scripts required restoring only their PYTHON assignment while retaining original literal placeholder destinations. No restored private bytes were written or exported.',
 'The published verification-v2 summary describes its then-259-file snapshot. The final archive has 265 files and was independently checked here; the older snapshot is not silently rewritten.',
 'Historical note link remains unresolved by design. The original publication-only failure is retained; it is not a scientific failure or repeated experiment.'
 ],
 'network_requests':0,'scientific_calls':0,'repository_writes':0
}
OUT.mkdir(exist_ok=True)
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'checks':report['total_checks'],'failures':failures,'counts':report['counts'],'redaction_inverse_limit_files':redaction_not_reversed},allow_nan=False))
raise SystemExit(0 if report['status']=='PASS' else 1)
