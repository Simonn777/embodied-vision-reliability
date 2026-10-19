"""Summarize saved public file metadata only; never open scientific payloads."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/0021-metadata-screen'


def main():
    inventory_path = RESULTS / 'eth3d-file-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    ledger_path = RESULTS / 'fresh-source-http.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    scenes = ['pipes', 'meadow', 'office']
    rows = []
    for scene in scenes:
        row = inventory['scenes'][scene]
        rows.append({
            'scene': scene,
            'images': row['images']['count'],
            'image_bytes': row['images']['bytes'],
            'intrinsic_bytes': row['intrins']['bytes'],
            'pose_bytes': row['poses']['bytes'],
            'total_bytes': sum(row[k]['bytes'] for k in ['images', 'intrins', 'poses']),
        })
    model_bytes = 140550416
    matcher_bytes = 47500279 + 5206086
    total = sum(row['total_bytes'] for row in rows)
    result = {
        'schema': 'evr0021.fresh-source-metadata.v1',
        'scope': 'Metadata/source only; no acquisition or scientific allocation',
        'eth3d': {
            'repository': 'shngjz/e1a6043209bf56124cfa8fbe263cbcad',
            'revision': inventory['revision'],
            'public_api_access': True,
            'private': False,
            'gated': False,
            'payload_access_tested': False,
            'files': inventory['file_count'],
            'total_file_bytes': inventory['total_file_bytes'],
            'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            'predicted_depth_or_match_files_in_complete_tree': False,
            'license': 'CC BY-NC-SA 4.0, official ETH3D homepage; HF card absent',
        },
        'imc2021': {
            'repository': 'shngjz/975921f0c7d37a4a5287de3c1e54fb09',
            'revision': '4c1486d2c4905fda3ca33bd6b341f2c8e4acb12f',
            'public_api_access': True,
            'private': False,
            'gated': False,
            'full_inventory_complete': False,
            'full_size_bytes': None,
            'example_bag': 'phototourism_british_museum/5bag_000',
            'example_bag_files': 15,
            'example_bag_bytes': 2155395,
            'independent_capture_domain_claim': False,
            'license_verified_for_this_mirror': False,
        },
        'resource_example_not_allocation': {
            'scenes': rows,
            'image_count': sum(row['images'] for row in rows),
            'image_intrinsic_pose_bytes': total,
            'moge2_s_weight_bytes': model_bytes,
            'superpoint_and_lightglue_weight_bytes': matcher_bytes,
            'data_plus_weights_bytes': total + model_bytes + matcher_bytes,
            'includes_dependency_installation': False,
            'runtime_or_memory_qualified': False,
        },
        'transfers': {
            'direct_http_requests_logged': len(ledger),
            'direct_http_response_body_bytes': sum(r['body_bytes'] for r in ledger),
            'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            'scientific_or_model_payload_bytes': 0,
            'web_tool_search_calls': 1,
            'web_tool_open_calls': 1,
            'web_tool_network_body_bytes': None,
            'headers_tls_and_wire_bytes': None,
            'root_reused_tree_commit_body_bytes_not_double_counted': 37515,
        },
    }
    assert total == 681610362
    assert total + model_bytes + matcher_bytes == 874867143
    with (RESULTS / 'fresh-source-receipt.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'metadata_bytes': result['transfers']['direct_http_response_body_bytes'],
                      'example_data_plus_weights_bytes': result['resource_example_not_allocation']['data_plus_weights_bytes']}))


if __name__ == '__main__':
    main()
