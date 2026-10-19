"""Read only public repository metadata/source; reject scientific payload paths."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / 'results/0021-metadata-screen'
PRIVATE = Path(__file__).resolve().parents[2].parent / '0021-fresh-source-screen/private-snapshots'


def fetch(url, name):
    parsed = urllib.parse.urlsplit(url)
    permitted = (
        parsed.hostname == 'api.github.com'
        or (parsed.hostname == 'huggingface.co' and parsed.path.startswith('/api/'))
        or (parsed.hostname == 'raw.githubusercontent.com' and parsed.path.endswith(('.py', '.md', '/LICENSE', '/LICENSE.txt')))
        or (parsed.hostname in ('www.eth3d.net', 'eth3d.net', 'eth3d.ethz.ch') and parsed.path in ('/', '/datasets', '/download_datasets', '/documentation', '/overview'))
        or (parsed.hostname == 'huggingface.co' and parsed.path.endswith('/README.md'))
    )
    if parsed.scheme != 'https' or not permitted:
        raise ValueError('Only whitelisted public metadata/source URLs are allowed')
    if Path(name).name != name:
        raise ValueError('Output must be a basename')
    out = PRIVATE / name
    if out.exists():
        raise FileExistsError(out)
    request = urllib.request.Request(url, headers={'User-Agent': 'public-metadata-review', 'Accept-Encoding': 'identity'})
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as error:
        receipt = {'url': url, 'status': None, 'body_bytes': 0,
                   'kind': 'public_metadata_or_source_only', 'authorization_header_used': False,
                   'error': str(error), 'body_bytes_scope': 'No HTTP response body received'}
        with (ROOT / 'fresh-source-http.jsonl').open('a') as handle:
            handle.write(json.dumps(receipt, sort_keys=True) + '\n')
        return receipt, b''
    with response:
        body = response.read()
        receipt = {
            'url': url, 'final_url': response.geturl(), 'status': response.status,
            'body_bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
            'kind': 'public_metadata_or_source_only', 'authorization_header_used': False,
            'saved_private': str(out.relative_to(PRIVATE.parent.parent)), 'link_header': response.headers.get('Link'),
            'content_type': response.headers.get('Content-Type'),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(body)
    with (ROOT / 'fresh-source-http.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + '\n')
    return receipt, body


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('name')
    args = parser.parse_args()
    receipt, _ = fetch(args.url, args.name)
    print(json.dumps(receipt, indent=2))
