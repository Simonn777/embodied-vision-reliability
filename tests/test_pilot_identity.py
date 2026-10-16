from pathlib import Path
import runpy
import pytest
from evr.pilot_identity import allocate


def test_transitive_frame_and_prefix_groups_never_cross():
    entries=[{'video_sha256':str(i),'prefix64_sha256':str(i)} for i in range(35)]
    entries[2]['prefix64_sha256']=entries[1]['prefix64_sha256']
    hashes=[[str(i)] for i in range(35)];hashes[3]=['bridge'];hashes[2].append('bridge')
    result=allocate(entries,hashes)
    group=next(g for g in result['groups'] if 1 in g['members'])
    assert group['members']==[1,2,3]
    assert result['targets_met']
    assert next(g for g in result['groups'] if 0 in g['members'])['partition']=='development'
    assert result==allocate(entries,hashes)


def test_development_group_quarantines_duplicates_and_shortage():
    entries=[{'video_sha256':'same','prefix64_sha256':'same'} for i in range(3)]
    result=allocate(entries,[['x']]*3)
    assert result['counts']=={'development':3,'calibration':0,'evaluation':0,'reserve':0}
    assert not result['targets_met']
    assert [x['development_inference_allowed'] for x in result['allocation']]==[True,False,False]


def test_output_reuse_is_refused(tmp_path):
    reserve=runpy.run_path(str(Path('scripts/run_pilot_preflight.py')))['reserve_output']
    output=tmp_path/'run';reserve(output)
    with pytest.raises(FileExistsError):reserve(output)
