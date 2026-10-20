"""Owned source and immutable artifact contracts for the physical diagnostic."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT / 'protocols/0025b-rs-depth-semantics.md'
OWNED = ('acquire.py', 'common.py', 'geometry_core.py', 'predict.py',
         'evaluate.py', 'audit_saved.py', 'supervise.py')
ASSETS = {
    'living_room_traj2_frei_png.tar.gz': ('living_room', 429825445),
    'traj2_frei_png.tar.gz': ('office', 301828475),
}
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
ASSET_HEADERS = {
    'living_room_traj2_frei_png.tar.gz': ('https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz', '"199e9da5-5111591ffd040"'),
    'traj2_frei_png.tar.gz': ('https://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz', '"11fd897b-51115bf98c980"'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def relative_name(value):
    require(isinstance(value, str) and value and not value.startswith('/') and
            '\\' not in value and '\0' not in value and ':' not in value and
            '..' not in value.split('/'), 'unsafe relative member/source path')
    return str(PurePosixPath(value))


def owned_paths():
    return {(HERE / name).relative_to(ROOT).as_posix() for name in OWNED} | {PROTOCOL.relative_to(ROOT).as_posix()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def save_receipt(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def verify_sources(commit):
    supervisor = module('_evr0025b_owned_supervisor', HERE / 'supervise.py')
    return supervisor.verify_frozen(commit, [HERE / name for name in OWNED] + [PROTOCOL])


def verify_source_bytes(bindings, expected=None):
    require(isinstance(bindings, dict) and bindings, 'empty source bindings')
    if expected is not None:
        require(set(bindings) == set(expected), 'incomplete/unexpected source binding set')
    for path, digest in bindings.items():
        require(relative_name(path) == path and isinstance(digest, str) and
                re.fullmatch('[0-9a-f]{64}', digest) is not None, 'invalid source binding')
        current = ROOT / path
        require(current.resolve().is_relative_to(ROOT.resolve()) and not current.is_symlink()
                and sha(current) == digest, 'owned/frozen source changed')


def private_output(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), 'fresh output required')
    require(not path.resolve().is_relative_to(ROOT), 'scientific output must be outside public repository')
    path.mkdir(parents=True)
    return path


def command_flags(command):
    require(isinstance(command, list) and all(isinstance(v, str) and '\0' not in v for v in command)
            and len(command) >= 2 and len(command[2:]) % 2 == 0, 'outer command shape')
    pairs = list(zip(command[2::2], command[3::2]))
    require(all(k.startswith('--') for k, _ in pairs) and
            len({k for k, _ in pairs}) == len(pairs), 'outer command flags')
    return dict(pairs)


def expand(value):
    return Path(value.replace('$REPOSITORY', str(ROOT)).replace('$PRIVATE', str(ROOT.parent))).resolve()


def verify_outer(path, operation, commit, output_dir, output_flag):
    path = Path(path)
    outer = read_json(path)
    require(outer.get('schema') == 'evr0025b.supervision.v1' and
            outer.get('operation') == operation and outer.get('expected_commit') == commit,
            'outer operation/source identity')
    require(outer.get('status') == 'completed' and type(outer.get('exit_code')) is int and outer['exit_code'] == 0 and
            outer.get('termination') is None and outer.get('remaining_descendant_pids') == [] and
            outer.get('cleanup_errors') == [], 'outer supervision did not complete cleanly')
    require(finite_number(outer.get('hard_timeout_seconds')) and outer['hard_timeout_seconds'] == 1800 and
            integer(outer.get('group_rss_limit_kib')) and outer['group_rss_limit_kib'] == 8 * 1024 * 1024 and
            finite_number(outer.get('wall_seconds')) and 0 <= outer['wall_seconds'] < 1800 and
            integer(outer.get('peak_group_rss_kib')) and outer['peak_group_rss_kib'] <= 8 * 1024 * 1024,
            'outer resource contract')
    require(outer.get('thread_environment') == {name: '4' for name in THREAD_KEYS}, 'outer thread contract')
    for name in ('stdout', 'stderr'):
        require(sha(path.parent / (name + '.log')) == outer[name + '_sha256'], 'outer log binding')
    flags = command_flags(outer['command'])
    expected_script = {'acquire': 'acquire.py', 'predict': 'predict.py', 'evaluate': 'evaluate.py',
                       'audit': 'audit_saved.py'}[operation]
    require(outer['command'][:2] == ['$PYTHON', '$REPOSITORY/experiments/0025b-physical-depth/' + expected_script],
            'outer executable identity')
    require(flags.get('--expected-commit') == commit and output_flag in flags and
            expand(flags[output_flag]) == Path(output_dir).resolve(), 'outer child source/output binding')
    if operation == 'acquire':
        require('--repo-root' in flags and expand(flags['--repo-root']) == ROOT.resolve() and
                flags.get('--source-relative', 'experiments/0025b-physical-depth/acquire.py') ==
                (HERE / 'acquire.py').relative_to(ROOT).as_posix() and
                flags.get('--protocol-relative') == PROTOCOL.relative_to(ROOT).as_posix(),
                'acquisition source/protocol location')
    verify_source_bytes(outer['source_sha256'], owned_paths())
    return outer


def verify_acquisition(directory, outer_path, commit):
    directory = Path(directory)
    outer = verify_outer(outer_path, 'acquire', commit, directory, '--destination')
    receipt = read_json(directory / 'receipt.json')
    total = sum(value[1] for value in ASSETS.values())
    require(receipt.get('schema') == 'evr0025b.acquisition.v1' and
            receipt.get('status') == 'completed' and receipt.get('expected_commit') == commit and
            integer(receipt.get('body_bytes')) and receipt['body_bytes'] == total,
            'completed acquisition required')
    require(receipt.get('thread_environment') == {name: '4' for name in THREAD_KEYS} and
            finite_number(receipt.get('wall_seconds')) and 0 <= receipt['wall_seconds'] < 1800,
            'acquisition thread/wall contract')
    expected_limits = dict(wall_seconds=1800, body_bytes=total, request_timeout_seconds=30,
                           outer_descendants_rss_bytes=8*1024**3, maximum_payload_GETs=2,
                           automatic_retries=0)
    require(receipt.get('limits') == expected_limits and
            all(type(v) is int for v in receipt['limits'].values()), 'acquisition limits')
    require([a['name'] for a in receipt['assets']] == list(ASSETS), 'exact two-archive population')
    expected_source = {(HERE / 'acquire.py').relative_to(ROOT).as_posix(), PROTOCOL.relative_to(ROOT).as_posix()}
    verify_source_bytes(receipt['source_sha256'], expected_source)
    require(all(outer['source_sha256'][k] == v for k, v in receipt['source_sha256'].items()),
            'inner/outer source mismatch')
    indices = {}
    for row in receipt['assets']:
        name = row['name']
        path = directory / name
        size = ASSETS[name][1]
        url, etag = ASSET_HEADERS[name]
        require(row.get('url') == url and row.get('etag') == etag and
                type(row.get('bytes')) is int and row['bytes'] == size,
                'acquisition asset URL/ETag/size identity')
        response, observed = row['response'], row['observed_response']
        require(type(observed.get('status')) is int and observed['status'] == 200 and
                observed.get('final_url') == url and isinstance(observed.get('headers'), list),
                'observed HTTP identity')
        headers = {}
        for pair in observed['headers']:
            require(isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, str) for v in pair),
                    'malformed observed header')
            headers.setdefault(pair[0].lower(), []).append(pair[1].strip())
        require(headers.get('content-length') == [str(size)] and headers.get('etag') == [etag] and
                'transfer-encoding' not in headers and
                headers.get('content-encoding', [None]) in ([None], ['identity'], ['x-gzip'], ['gzip']),
                'missing/duplicate/changed HTTP header')
        encoding = headers.get('content-encoding', [None])[0]
        require(response == dict(content_length=size, etag=etag, content_encoding=encoding,
                                 status=200, final_url=url) and
                type(response['status']) is int and type(response['content_length']) is int,
                'normalized/observed HTTP disagreement')
        require(row['status'] == 'completed' and integer(row['body_bytes']) and row['body_bytes'] == size
                and path.is_file() and not path.is_symlink() and path.stat().st_size == size and
                sha(path) == row['sha256'] and row.get('sha256_kind') == 'trust-on-first-use; not a publisher digest',
                'archive bytes changed or incomplete')
        index = row['index']
        index_path = directory / (name + '.index.json')
        require(index['filename'] == index_path.name and integer(index['bytes'], 1) and
                not index_path.is_symlink() and index_path.stat().st_size == index['bytes']
                and sha(index_path) == index['sha256'], 'index bytes changed')
        values = read_json(index_path)
        require(values.get('schema') == 'evr0025b.archive-index.v1' and values.get('archive') == name and
                index['archive_sha256_tofu'] == values['archive_sha256_tofu'] == row['sha256'] and
                values['member_values_decoded'] is False, 'index/archive binding')
        validate_index(values)
        require(index['member_count'] == values['member_count'] and
                index['expanded_bytes'] == values['expanded_bytes'] and
                integer(index['member_count']) and integer(index['expanded_bytes']), 'index count binding')
        indices[name] = values
    return indices, {'receipt_sha256': sha(directory / 'receipt.json'),
                     'supervision_sha256': sha(outer_path),
                     'archive_sha256': {row['name']: row['sha256'] for row in receipt['assets']}}


def validate_index(values):
    members = values.get('members')
    require(isinstance(members, list) and 0 < len(members) <= 100000, 'structural member population')
    names, content, blocks = set(), 0, 1024
    for member in members:
        name = relative_name(member['name'])
        require(name == member['name'] and name not in names and member['type'] in ('file', 'directory') and
                integer(member['size']) and member['size'] <= 512*1024**2, 'invalid indexed member')
        require(member['type'] != 'directory' or member['size'] == 0, 'nonempty directory')
        names.add(name)
        content += member['size']
        blocks += 512 + ((member['size'] + 511)//512)*512
    require(integer(values.get('member_count')) and values['member_count'] == len(members) and
            integer(values.get('member_content_bytes')) and values['member_content_bytes'] == content and
            integer(values.get('expanded_bytes')) and blocks <= values['expanded_bytes'] <= 16*1024**3,
            'invalid index totals')


def select_frames(indices):
    """Only structural paired numeric filenames determine the 16-frame population."""
    require(set(indices) == set(ASSETS), 'exact archive inventory required')
    selected, populations = [], []
    for archive, (scene, _) in ASSETS.items():
        validate_index(indices[archive])
        archive_sha = indices[archive]['archive_sha256_tofu']
        require(isinstance(archive_sha, str) and re.fullmatch('[0-9a-f]{64}', archive_sha) is not None,
                'missing archive identity')
        categories = {'rgb': {}, 'depth': {}}
        roots = {'rgb': set(), 'depth': set()}
        for member in indices[archive]['members']:
            if member['type'] != 'file':
                continue
            path = PurePosixPath(member['name'])
            if path.suffix.lower() != '.png' or path.parent.name not in categories:
                continue
            require(re.fullmatch(r'[0-9]+', path.stem) is not None, 'non-numeric image stem')
            index = int(path.stem)
            target = categories[path.parent.name]
            require(index not in target, 'duplicate numeric image identity')
            require(0 < member['size'] <= 4 * 1024 * 1024, 'selected image member size bound')
            target[index] = member
            roots[path.parent.name].add(str(path.parent.parent))
        require(len(roots['rgb']) == len(roots['depth']) == 1 and roots['rgb'] == roots['depth'],
                'ambiguous RGB/depth directory association')
        stems = sorted(set(categories['rgb']) & set(categories['depth']))
        require(len(stems) >= 8, 'fewer than eight paired filenames')
        ranks = [(2 * i * (len(stems)-1) + 7) // 14 for i in range(8)]
        require(len(set(ranks)) == 8, 'non-distinct fixed ranks')
        populations.append({'archive': archive, 'scene': scene, 'paired_count': len(stems),
                            'rgb_count': len(categories['rgb']), 'depth_count': len(categories['depth']),
                            'ranks': ranks, 'paired_stems': stems, 'archive_sha256': archive_sha,
                            'rgb_only_stems': sorted(set(categories['rgb']) - set(stems)),
                            'depth_only_stems': sorted(set(categories['depth']) - set(stems)),
                            'paired_inventory': [dict(numeric_stem=stem, rgb_member=categories['rgb'][stem],
                                                      depth_member=categories['depth'][stem]) for stem in stems]})
        for rank in ranks:
            stem = stems[rank]
            selected.append({'index': len(selected), 'scene': scene, 'archive': archive,
                             'archive_sha256': archive_sha,
                             'rank': rank, 'numeric_stem': stem,
                             'rgb_member': categories['rgb'][stem],
                             'depth_member': categories['depth'][stem]})
    return selected, populations


def extract_selected(directory, selected, kind, destination):
    """Extract only preselected RGB OR depth PNG bytes; never trajectories."""
    require(kind in ('rgb', 'depth'), 'unsupported payload category')
    directory = Path(directory)
    # Reconstruct the deterministic population from the hash-bound structural
    # indices before opening any compressed member stream. The caller separately
    # verifies the outer acquisition chain (and, for depth, the prediction seal).
    receipt = read_json(directory / 'receipt.json')
    require(receipt.get('status') == 'completed' and
            [r['name'] for r in receipt['assets']] == list(ASSETS), 'completed extraction provenance')
    indices = {}
    for row in receipt['assets']:
        index_path = directory / (row['name'] + '.index.json')
        require(row['status'] == 'completed' and not index_path.is_symlink() and
                index_path.stat().st_size == row['index']['bytes'] and sha(index_path) == row['index']['sha256'],
                'extraction index changed')
        indices[row['name']] = read_json(index_path)
        require(indices[row['name']]['archive_sha256_tofu'] == row['sha256'], 'extraction archive binding')
    expected, _ = select_frames(indices)
    require(selected == expected, 'selected population/member identity changed')
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    extracted = {}
    # Use the bounded header reader instead of TarFile extension handling. No
    # image/pose decoder or filesystem extraction of arbitrary paths is used.
    acquisition = module('_evr0025b_extraction_reader', HERE / 'acquire.py')
    for archive in ASSETS:
        wanted = {row[kind + '_member']['name']: row for row in selected if row['archive'] == archive}
        path = directory / archive
        bound = indices[archive]['archive_sha256_tofu']
        require(not path.is_symlink() and sha(path) == bound, 'archive changed before extraction')
        seen, actual_members = set(), []
        with path.open('rb') as compressed:
            reader = acquisition.MeteredGzipReader(compressed, acquisition.time.monotonic())
            while True:
                header = reader.read(512)
                require(len(header) == 512, 'truncated extraction header')
                if header == bytes(512):
                    require(reader.read(512) == bytes(512), 'missing extraction end blocks')
                    break
                member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
                canonical = relative_name(member.name)
                require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE) and
                        not member.linkname and 0 <= member.size <= 512*1024**2,
                        'unsupported extraction archive entry')
                actual_members.append(dict(name=canonical, size=member.size,
                                           type='file' if member.isfile() else 'directory'))
                require(len(actual_members) <= len(indices[archive]['members']) and
                        actual_members[-1] == indices[archive]['members'][len(actual_members)-1],
                        'archive header/index mismatch')
                remaining = ((member.size + 511)//512)*512
                data = bytearray() if canonical in wanted else None
                while remaining:
                    block = reader.read(min(1024*1024, remaining))
                    require(bool(block), 'truncated member content')
                    if data is not None:
                        data.extend(block[:max(0, member.size-len(data))])
                    remaining -= len(block)
                if data is not None:
                    row = wanted[canonical]
                    require(canonical not in seen and member.isfile() and len(data) == member.size and
                            PurePosixPath(canonical).parent.name == kind, 'selected category/length/duplicate')
                    output_path = destination / ('frame-%02d.png' % row['index'])
                    with output_path.open('xb') as output:
                        output.write(data)
                    extracted[row['index']] = {'path': str(output_path), 'sha256': hashlib.sha256(data).hexdigest(),
                                               'bytes': len(data), 'member': canonical}
                    seen.add(canonical)
            while True:
                tail = reader.read(1024*1024)
                if not tail:
                    break
                require(not any(tail), 'nonzero content after archive end')
            reader.stream.close()
        require(actual_members == indices[archive]['members'] and sha(path) == bound,
                'archive changed during extraction')
        require(seen == set(wanted), 'selected member missing')
    require(len(extracted) == 16, 'exact selected payload count')
    return extracted
