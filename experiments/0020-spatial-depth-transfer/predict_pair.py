"""One fixed-bank spatial depth scoring call; no estimator or label access."""
import argparse
import json
from pathlib import Path
import sys
import time

import common as c

SCHEMA = 'evr0020.prediction.v1'


def identity(row, manifest_hash, sources, library, binding):
    i = row['pair_index']
    return {'schema': SCHEMA, **{k: row[k] for k in ('pair_index','scene','key','filename')},
            'manifest_sha256': manifest_hash, 'source_sha256': sources, 'library': library,
            'dataset_sha256_declared': row['dataset_sha256'], 'dataset_sha256_verified': row['dataset_sha256'],
            'dataset_bytes': row['dataset_bytes'], 'baseline_seal_sha256': binding['seal_sha256'],
            'candidate_prediction_sha256': {arm: binding['predictions'][f'{i}:{arm}'] for arm in c.ARMS},
            'candidate_order': list(c.ARMS), 'max_fits': c.MAX_FITS_PER_PAIR}


def invoke(pair, candidates, key):
    import score
    return score.score_pair(key, pair['x1'], pair['x2'], pair['depth1'], pair['depth2'],
                            pair['calibration'], candidates)


def run(data, manifest_path, baseline_seal, output, index):
    total_start = time.perf_counter()
    with Path(output).open('x+') as stream:
        record = {'schema': SCHEMA, 'status': 'started', 'phase': 'binding', 'pair_index': index,
                  'timings': {k: None for k in ('binding_seconds','input_seconds','score_seconds',
                                               'output_seconds','worker_total_seconds')}}
        save = lambda: c.write_record(stream, record)
        save(); failure_kind = 'binding'
        try:
            if type(index) is not int or not 0 <= index < 80: raise ValueError('Invalid pair index')
            manifest, bank, binding = c.baseline_bundle(manifest_path, baseline_seal)
            row, candidates = manifest['selected_pairs'][index], bank[index]
            sources, mh = c.source_hashes(), c.sha256(manifest_path)
            library = c.library_config()['current']
            if Path(data).name != row['filename'] or Path(data).stat().st_size != row['dataset_bytes']:
                raise ValueError('Dataset path/size differs')
            if c.sha256(data) != row['dataset_sha256']: raise ValueError('Dataset hash differs')
            record.update(identity(row, mh, sources, library, binding))
            record['timing_semantics'] = {
                'score_seconds': 'One core score_pair call for every candidate and all five selectors',
                'output_seconds': 'Score validation and finite JSON conversion/encoding; file writes are also charged by whole worker/process wall',
                'candidate_generation_seconds': 'Historical sum of all four original candidate process wall times; frontend cost unavailable'}
            record['candidate_generation_seconds'] = sum(r['timings']['charged_total_seconds'] for r in candidates)
            record['timings']['binding_seconds'] = time.perf_counter()-total_start
            failure_kind='input'; record['phase']='inputs'; save(); started=time.perf_counter()
            import h5py
            with h5py.File(data,'r') as handle: pair=c.read_pair(handle,row['key'])
            record.update(input_arrays=c.input_hashes(pair), calibration=c.json_safe(pair['calibration']),
                          input_diagnostics=c.json_safe(pair['input_diagnostics']))
            c.check_inputs(record,candidates)
            record['timings']['input_seconds']=time.perf_counter()-started
            failure_kind='scoring'; record['phase']='scoring'; save(); started=time.perf_counter()
            try: result=invoke(pair,candidates,row['key'])
            finally: record['timings']['score_seconds']=time.perf_counter()-started
            failure_kind='output'; record['phase']='output'; save(); started=time.perf_counter()
            result=c.json_safe(result)
            c.validate_scores(result,candidates,row['key'])
            json.dumps(result,sort_keys=True,allow_nan=False)
            record['scores']=result
            record['timings']['output_seconds']=time.perf_counter()-started
            failure_kind='binding'; record['phase']='postcheck'; save()
            if c.source_hashes()!=sources or c.sha256(manifest_path)!=mh:
                raise ValueError('Source or manifest changed during scoring')
            if c.sha256(baseline_seal)!=binding['seal_sha256'] or c.library_config()['current']!=library:
                raise ValueError('Baseline seal or source-library declaration changed')
            record.update(status='completed',phase='complete')
            record['timings']['worker_total_seconds']=time.perf_counter()-total_start
            save(); return 0
        except BaseException as exc:
            message=type(exc).__name__+': '+str(exc)
            for private in (str(Path(data).resolve()),str(Path(output).resolve()),str(c.ROOT.resolve())):
                message=message.replace(private,'<local-path>')
            record.update(status='error',failed_phase=record['phase'],phase='error',error=message,
                          failure_kind=failure_kind if isinstance(exc,Exception) else 'interruption')
            record['timings']['worker_total_seconds']=time.perf_counter()-total_start
            save(); return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','manifest','baseline-seal','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pair-index',type=int,required=True)
    a=parser.parse_args(); return run(a.data,a.manifest,a.baseline_seal,a.output,a.pair_index)

if __name__=='__main__': sys.exit(main())
