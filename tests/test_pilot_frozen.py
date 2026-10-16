import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from evr.pilot_frozen import verify_freeze,verify_video,prediction_rows
from evr.pilot_identity import digest_array
from evr.bootstapir_dev import sha256


def identity(example,index):
    return {'archive_index':index,'video_sha256':digest_array(example['video']),
      'prefix64_sha256':digest_array(example['video'][:64]),
      'arrays':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest_array(a)} for k,a in example.items()}}




def test_future_prediction_grid_and_zero_query_group_retained(tmp_path):
    a={'video':np.zeros((64,256,256,3),np.uint8),'points':np.zeros((1,64,2),np.float32),'occluded':np.zeros((1,64),bool)}
    b={k:v.copy() for k,v in a.items()};b['occluded'][:]=True
    manifest={'identities':[identity(a,0),identity(b,1)],'groups':[{'group_id':'a','members':[0],'partition':'evaluation'},{'group_id':'empty','members':[1],'partition':'evaluation'}]}
    for group,e,ids in [('a',a,[0]),('empty',b,[])]:
        folder=tmp_path/group;folder.mkdir();i=0 if group=='a' else 1;file=folder/f'video-{i:03}.csv'
        with file.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['frame','query','point','x','y','occlusion_logit','expected_dist_logit'])
            if ids:
                for t in range(64):w.writerow([t,0,0,0,0,-1,-1])
        (folder/'receipt.json').write_text(json.dumps({'status':'completed','freeze_input_commit':'fixed','group_id':group,'videos':{str(i):{'point_indices':ids,'frames':64,'raw_sha256':sha256(file)}}}))
    rows,frame=prediction_rows(tmp_path,manifest,'evaluation',[a,b],'fixed')
    assert len(rows)==64 and frame==['a','empty']
    (tmp_path/'a/video-000.csv').write_text('tampered')
    with pytest.raises(ValueError,match='prediction identity'):prediction_rows(tmp_path,manifest,'evaluation',[a,b],'fixed')


def test_array_or_prefix_identity_change_rejected():
    e={'video':np.zeros((64,256,256,3),np.uint8),'points':np.zeros((1,64,2),np.float32),'occluded':np.zeros((1,64),bool)}
    i=identity(e,0);e['video'][0,0,0,0]=1
    with pytest.raises(ValueError,match='video identity'):verify_video(e,i)


from evr.pilot_frozen import (CANONICAL_RECEIPT, CANONICAL_CONFIG,
    CANONICAL_MANIFEST, CANONICAL_PROTOCOL, MANDATORY_INPUTS,
    MANDATORY_DEPENDENCIES, PREREQUISITE_FILES)
from evr.bootstapir_dev import UPSTREAM, UPSTREAM_HASHES


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()


def commit_fixture(root, message):
    git(root, 'add', '.')
    return git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
               'commit', '-qm', message) or git(root, 'rev-parse', 'HEAD')


def publish_receipt(root, receipt):
    (root/CANONICAL_RECEIPT).write_text(json.dumps(receipt))
    commit_fixture(root, 'receipt fixture')
    git(root, 'update-ref', 'refs/remotes/origin/main', git(root, 'rev-parse', 'HEAD'))


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    for path in MANDATORY_INPUTS:
        p=tmp_path/path;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('fixture: '+path)
    model='87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f'
    archive='ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc'
    config={'version':2,'manifest_path':CANONICAL_MANIFEST,'protocol_path':CANONICAL_PROTOCOL,
            'freeze_receipt_path':CANONICAL_RECEIPT,'upstream_commit':UPSTREAM,
            'checkpoint_sha256':model,'archive_sha256':archive}
    (tmp_path/CANONICAL_CONFIG).write_text(json.dumps(config))
    (tmp_path/CANONICAL_MANIFEST).write_text(json.dumps({'archive':{'sha256':archive}}))
    dep='results/0002-preflight/dependencies.json'
    (tmp_path/dep).write_text(json.dumps({'packages':{k:{'version':v} for k,v in MANDATORY_DEPENDENCIES.items()}}))
    input_commit=commit_fixture(tmp_path, 'input fixture')
    receipt={'version':2,'status':'frozen_exploratory_v2','input_commit':input_commit,
      'manifest_path':CANONICAL_MANIFEST,'config_path':CANONICAL_CONFIG,
      'protocol_path':CANONICAL_PROTOCOL,'receipt_path':CANONICAL_RECEIPT,
      'dependency_identity_receipt':dep,'dependency_versions':dict(MANDATORY_DEPENDENCIES),
      'input_sha256':{p:sha256(tmp_path/p) for p in MANDATORY_INPUTS},
      'prerequisite_receipt_sha256':{p:sha256(tmp_path/p) for p in PREREQUISITE_FILES},
      'checkpoint_sha256':model,'archive_sha256':archive,'upstream_commit':UPSTREAM,
      'upstream_module_sha256':dict(UPSTREAM_HASHES)}
    publish_receipt(tmp_path,receipt)
    monkeypatch.setattr('evr.pilot_frozen.importlib.metadata.version',lambda p:MANDATORY_DEPENDENCIES[p])
    return tmp_path,receipt


def test_complete_committed_published_v2_passes(frozen_repo):
    root,r=frozen_repo
    assert verify_freeze(root,CANONICAL_RECEIPT)['input_commit']==r['input_commit']
    assert CANONICAL_RECEIPT not in MANDATORY_INPUTS
    assert not git(root,'ls-tree',r['input_commit'],'--',CANONICAL_RECEIPT)


@pytest.mark.parametrize('field',['input_sha256','dependency_versions','prerequisite_receipt_sha256'])
@pytest.mark.parametrize('change',['empty','missing','omit','extra'])
def test_incomplete_or_extra_maps_fail_closed(frozen_repo,field,change):
    root,r=frozen_repo
    if change=='empty':r[field]={}
    elif change=='missing':r[field].pop(next(iter(r[field])))
    elif change=='omit':r.pop(field)
    else:r[field]['unexpected']='bad'
    publish_receipt(root,r)
    with pytest.raises(ValueError,match='map required'):verify_freeze(root,CANONICAL_RECEIPT)


def test_old_commit_with_new_current_hash_cannot_pass(frozen_repo):
    root,r=frozen_repo;p='src/evr/pilot_metrics.py'
    (root/p).write_text('new code with newly claimed hash')
    r['input_sha256'][p]=sha256(root/p)
    publish_receipt(root,r)  # current/HEAD/receipt all match; OLD input blob does not
    with pytest.raises(ValueError,match='input commit/current'):verify_freeze(root,CANONICAL_RECEIPT)


@pytest.mark.parametrize('kind',['dirty','staged','untracked','unpublished'])
def test_receipt_must_be_tracked_unchanged_and_published(frozen_repo,kind):
    root,r=frozen_repo;p=root/CANONICAL_RECEIPT
    if kind=='untracked':git(root,'rm','--cached',CANONICAL_RECEIPT)
    else:
        r['extra']='changed';p.write_text(json.dumps(r))
        if kind=='staged':git(root,'add',CANONICAL_RECEIPT)
        if kind=='unpublished':commit_fixture(root,'unpublished receipt edit')
    with pytest.raises(ValueError):verify_freeze(root,CANONICAL_RECEIPT)


@pytest.mark.parametrize('field',['manifest_path','config_path','protocol_path','receipt_path','dependency_identity_receipt'])
def test_replaced_bound_paths_rejected(frozen_repo,field):
    root,r=frozen_repo;r[field]='replacement.json';publish_receipt(root,r)
    with pytest.raises(ValueError,match='path binding'):verify_freeze(root,CANONICAL_RECEIPT)


@pytest.mark.parametrize('path',['receipt.json','protocols/0002-freeze-v1.json','protocols/../protocols/0002-freeze-v2.json'])
def test_only_canonical_receipt_argument_accepted(frozen_repo,path):
    root,_=frozen_repo
    with pytest.raises(ValueError,match='canonical v2 receipt'):verify_freeze(root,path)


def test_dependency_tamper_and_actual_version_mismatch(frozen_repo,monkeypatch):
    root,r=frozen_repo
    monkeypatch.setattr('evr.pilot_frozen.importlib.metadata.version',lambda p:'wrong')
    with pytest.raises(ValueError,match='dependency version changed'):verify_freeze(root,CANONICAL_RECEIPT)
    r['dependency_versions']['torch']='2.5.1+cpu';publish_receipt(root,r)
    with pytest.raises(ValueError,match='dependency map required'):verify_freeze(root,CANONICAL_RECEIPT)


def test_current_input_modified_or_staged_fails(frozen_repo):
    root,r=frozen_repo;p='src/evr/pilot_metrics.py';(root/p).write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):verify_freeze(root,CANONICAL_RECEIPT)
    git(root,'add',p)
    with pytest.raises(ValueError,match='staged frozen'):verify_freeze(root,CANONICAL_RECEIPT)


def test_receipt_symlink_and_self_containing_commit_rejected(frozen_repo):
    root,r=frozen_repo;p=root/CANONICAL_RECEIPT;copy=root/'copy.json';copy.write_bytes(p.read_bytes())
    p.unlink();p.symlink_to('../copy.json')
    with pytest.raises(ValueError,match='canonical v2 receipt'):verify_freeze(root,CANONICAL_RECEIPT)
    p.unlink();p.write_bytes(copy.read_bytes());r['input_commit']=git(root,'rev-parse','HEAD')
    publish_receipt(root,r)
    with pytest.raises(ValueError,match='circular freeze'):verify_freeze(root,CANONICAL_RECEIPT)


def test_missing_published_receipt_rejects(frozen_repo):
    root,r=frozen_repo;git(root,'update-ref','refs/remotes/origin/main',r['input_commit'])
    with pytest.raises(ValueError):verify_freeze(root,CANONICAL_RECEIPT)


def revise_fixture_inputs(root,r,path,content):
    # Fresh descendant of the original input tree, before ANY v2 receipt exists.
    git(root,'checkout','--detach',r['input_commit'])
    (root/path).write_text(json.dumps(content))
    r['input_commit']=commit_fixture(root,'revised input fixture')
    r['input_sha256']={p:sha256(root/p) for p in MANDATORY_INPUTS}
    publish_receipt(root,r)


@pytest.mark.parametrize('field',['manifest_path','protocol_path','freeze_receipt_path'])
def test_committed_config_replacement_rejected_even_when_all_hashes_match(frozen_repo,field):
    root,r=frozen_repo;config=json.loads((root/CANONICAL_CONFIG).read_text())
    config[field]='replacement.json'
    revise_fixture_inputs(root,r,CANONICAL_CONFIG,config)
    with pytest.raises(ValueError,match='config path/version binding'):verify_freeze(root,CANONICAL_RECEIPT)


def test_committed_dependency_provenance_tamper_rejected(frozen_repo):
    root,r=frozen_repo;path=r['dependency_identity_receipt'];d=json.loads((root/path).read_text())
    d['packages']['torch']['version']='2.5.1+cpu'
    revise_fixture_inputs(root,r,path,d)
    with pytest.raises(ValueError,match='dependency provenance'):verify_freeze(root,CANONICAL_RECEIPT)


def test_duplicate_receipt_json_keys_rejected(frozen_repo):
    root,r=frozen_repo;p=root/CANONICAL_RECEIPT
    p.write_text(json.dumps(r)[:-1]+',"version":2}')
    commit_fixture(root,'duplicate receipt key fixture')
    git(root,'update-ref','refs/remotes/origin/main',git(root,'rev-parse','HEAD'))
    with pytest.raises(ValueError,match='duplicate JSON key'):verify_freeze(root,CANONICAL_RECEIPT)


@pytest.mark.parametrize('field',['input_sha256','dependency_versions'])
def test_maps_from_v1_cannot_substitute_for_v2(frozen_repo,field):
    root,r=frozen_repo
    if field=='input_sha256':
        r[field]['protocols/0002-exploratory-frozen-v1.md']=r[field].pop(CANONICAL_PROTOCOL)
    else:r[field]['torch']='2.5.1+cpu'
    publish_receipt(root,r)
    with pytest.raises(ValueError,match='map required'):verify_freeze(root,CANONICAL_RECEIPT)
