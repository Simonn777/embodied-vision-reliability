"""Bounded cached-source inventory; no source execution, archive or model access."""
import hashlib,json,re
from pathlib import PurePosixPath

REVISION='a053342a72fed32af4134b4ca0c44a78a3ac0fdb'
SOURCES={
 'official-tracking.body':('utils/gen_tracking_indoor.py','413f680d2dd8b7bf65179bdf305baf94207441dd186c4fbba0d7d9bf66787f2a'),
 'official-reprojection.body':('utils/reprojection.py','c226ff24be5037fbce6ed747bf31feb1fea1de17b7a8260317fd84737b5f1c00'),
 'official-export.body':('export_annotation.py','1464c84e8c802bed555aa097900b4fe410aed704e5f2eeca43136b60d79175cc'),
 'official-scene_export.body':('utils/export_scene.py','1dd0b26f1f95661cb2c97a054d4fae28bbac0b44a8db693f585dbad61b7511e3'),
 'official-readme.body':('README.md','f8bb2457a46f5121efa880437c0654ea85d8e6f89c03a550db38a95db096cbfd')}
TREE_SHA256='56df73777b676dfcdf875a6b8ecb347478d705ebbae2d37628a7f9eb6bc0281b'
TERMS={'dtype':r'\bdtype\b','astype':r'\bastype\b','scale':r'\bscal(?:e|es|ing)\b','units':r'\bunits?\b','translation':r'\btranslat(?:ion|e)\b','export':r'\bexport[A-Za-z_]*\b','annotations_file':r'(?<![A-Za-z_])annotations\.npz',
 'annot_file':r'(?<![A-Za-z_])annot\.npz','anno_file':r'(?<![A-Za-z_])anno\.npz',
 'float16':r'\bfloat16\b','float32':r'\bfloat32\b','loadtxt':r'\bloadtxt\b',
 'savez':r'\bsavez(?:_compressed)?\b','intrinsics':r'\bintrinsics\b','extrinsics':r'\bextrinsics\b',
 'visibilities':r'\bvisibilities\b','visibs':r'\bvisibs\b','valids':r'\bvalids\b',
 'rename':r'\brename\b','replace':r'\breplace\b','version_1_0':r'[vV]1\.0',
 'version_1_1':r'[vV]1\.1','version_1_2':r'[vV]1\.2'}


def read_bounded(path,cap=65536):
 if path.is_symlink()or not path.is_file():raise ValueError('regular source file required')
 with path.open('rb')as f:raw=f.read(cap+1)
 if len(raw)>cap:raise ValueError('source byte cap')
 raw.decode('utf8');return raw


def source_record(raw,path,expected_sha,entry):
 if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('pinned source SHA256')
 blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 if entry['type']!='blob'or entry['size']!=len(raw)or entry['sha']!=blob:raise ValueError('cached tree/blob binding')
 lines=raw.decode('utf8').splitlines()
 return {'path':path,'revision':REVISION,'bytes':len(raw),'sha256':expected_sha,'Git_blob_sha':blob,
  'read_line_ranges':[[1,len(lines)]],'source_url':f'https://github.com/y-zheng18/point_odyssey/blob/{REVISION}/{path}',
  'lexical_occurrence_lines':{term:[i for i,line in enumerate(lines,1)if re.search(pattern,line)]for term,pattern in TERMS.items()},
  'scope':'complete cached body statically read; lexical hits are not proof of execution or conversion history'}


def inspect_sources(cache):
 raw=read_bounded(cache/'official-tree.body')
 if hashlib.sha256(raw).hexdigest()!=TREE_SHA256:raise ValueError('cached tree digest')
 tree=json.loads(raw)
 if tree['sha']!=REVISION or tree['truncated']is not False or not 0<len(tree['tree'])<=1000:raise ValueError('tree identity/completeness')
 bypath={}
 for e in tree['tree']:
  name=e['path'];p=PurePosixPath(name)
  if p.is_absolute()or '\0'in name or any(x in ('','..','.')for x in name.split('/'))or name in bypath:raise ValueError('safe unique source-tree path')
  bypath[name]=e
 records={};total=len(raw)
 for name,(path,want)in SOURCES.items():
  body=read_bounded(cache/name);total+=len(body)
  if total>262144:raise ValueError('aggregate source byte cap')
  records[name]=source_record(body,path,want,bypath[path])
 blobs=[e for e in tree['tree']if e['type']=='blob']
 covered={v['path']for v in records.values()}
 return {'revision':REVISION,'cached_tree_sha256':TREE_SHA256,'cached_tree_bytes':len(raw),'cached_tree_entries':len(tree['tree']),
  'tree_blob_count':len(blobs),'cached_body_count':len(records),'source_body_bytes':total-len(raw),'total_source_and_tree_bytes_read':total,
  'sources':records,'uncached_tree_blob_paths':[e['path']for e in blobs if e['path']not in covered],
  'coverage_limit':'five bodies at one revision; complete cached tree path listing does not supply missing bodies or commit history'},tree
