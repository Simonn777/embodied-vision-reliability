"""Freeze compatible wheel metadata without importing numerical runtimes."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT.parent / '0022-frontend-preparation/dependency-metadata'
PINS = {
    'scipy': '1.13.1', 'opencv-python': '4.11.0.86', 'torchvision': '0.23.0',
    'kornia': '0.8.1', 'kornia-rs': '0.1.9', 'huggingface-hub': '0.34.4',
    'hf-xet': '1.1.9', 'PyYAML': '6.0.2', 'requests': '2.32.5',
    'urllib3': '2.5.0', 'certifi': '2025.10.5', 'charset-normalizer': '3.4.4',
    'idna': '3.11', 'tqdm': '4.67.1',
}


def main():
    env = default_environment()
    env['extra'] = ''
    assert env['python_version'] == '3.9'
    assert env['platform_system'] == 'Darwin' and env['platform_machine'] == 'arm64'
    supported = list(sys_tags())
    rank = {tag: index for index, tag in enumerate(supported)}
    wheels = []
    packages = {}
    for name, version in PINS.items():
        path = PRIVATE / (name + '-' + version + '.json')
        body = path.read_bytes()
        obj = json.loads(body)
        info = obj['info']
        assert platform.python_version() in SpecifierSet(info['requires_python'])
        candidates = []
        for entry in obj['urls']:
            if entry['packagetype'] != 'bdist_wheel' or entry['yanked']:
                continue
            _, _, _, tags = parse_wheel_filename(entry['filename'])
            compatible = tags.intersection(rank)
            if compatible:
                candidates.append((min(rank[t] for t in compatible), entry))
        assert candidates, name
        # A pure-Python charset wheel avoids an unnecessary binary extension.
        if name == 'charset-normalizer':
            candidates = [c for c in candidates if c[1]['filename'].endswith('py3-none-any.whl')]
        chosen = min(candidates, key=lambda item: item[0])[1]
        requires = info['requires_dist'] or []
        active = [s for s in requires if Requirement(s).marker is None or Requirement(s).marker.evaluate(env)]
        record = {
            'name': canonicalize_name(name), 'version': version,
            'filename': chosen['filename'], 'url': chosen['url'],
            'size': chosen['size'], 'sha256': chosen['digests']['sha256'],
            'requires_python': info['requires_python'], 'requires_dist': requires,
            'active_requires_dist': active, 'yanked': False,
            'metadata_url': 'https://pypi.org/pypi/' + name + '/' + version + '/json',
            'metadata_body_bytes': len(body),
            'metadata_sha256': hashlib.sha256(body).hexdigest(),
        }
        wheels.append(record)
        packages[canonicalize_name(name)] = record
    pending = ['torch', 'numpy', 'pillow'] + [Requirement(s).name for p in wheels for s in p['active_requires_dist']]
    existing = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in packages or name in existing:
            continue
        dist = metadata.distribution(name)
        requires = dist.requires or []
        active = [s for s in requires if Requirement(s).marker is None or Requirement(s).marker.evaluate(env)]
        assert platform.python_version() in SpecifierSet(dist.metadata.get('Requires-Python', ''))
        record = {
            'name': name, 'version': dist.version,
            'requires_python': dist.metadata.get('Requires-Python'),
            'active_requires_dist': active,
            'installed_metadata_text_sha256': hashlib.sha256(dist.read_text('METADATA').encode()).hexdigest(),
        }
        existing[name] = record
        pending.extend(Requirement(s).name for s in active)
    versions = {name: p['version'] for name, p in {**packages, **existing}.items()}
    for record in list(packages.values()) + list(existing.values()):
        for text in record['active_requires_dist']:
            requirement = Requirement(text)
            assert versions[canonicalize_name(requirement.name)] in requirement.specifier, (record['name'], text)
    ledger = [json.loads(s) for s in (PRIVATE / 'http.jsonl').read_text().splitlines()]
    result = {
        'schema': 'evr0022.dependency-wheel-metadata.v1',
        'scope': 'Metadata compatibility only; no wheels downloaded, installed, or imported',
        'python': platform.python_version(), 'platform': env,
        'wheels': wheels, 'wheel_total_bytes': sum(p['size'] for p in wheels),
        'reuse_installed': sorted(existing.values(), key=lambda p: p['name']),
        'active_requires_dist_satisfied_by_manifest': True,
        'metadata_http_body_bytes': sum(row['bytes'] for row in ledger),
        'metadata_http_ledger_sha256': hashlib.sha256((PRIVATE / 'http.jsonl').read_bytes()).hexdigest(),
        'source_import_closure_is_separate': True,
        'no_extras': True,
    }
    output = ROOT / 'results/0022-preparation/dependency-wheels.json'
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'wheels': len(wheels), 'wheel_bytes': result['wheel_total_bytes'],
                      'metadata_bytes': result['metadata_http_body_bytes'],
                      'existing_packages': len(existing)}))


if __name__ == '__main__':
    main()
