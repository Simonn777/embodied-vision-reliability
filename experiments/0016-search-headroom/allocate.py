"""Allocate only HDF5 root names, before any fresh scientific arrays are read."""
import argparse,hashlib,json
from pathlib import Path
import common

def select(keys,excluded):
    dev=[k for k in keys if k.startswith('0015-')]
    if len(dev)!=600 or len(set(dev))!=600: raise ValueError('Unexpected scene inventory')
    candidates=set(dev)-set(excluded)
    if len(candidates)!=584: raise ValueError('Prior16 missing or duplicated')
    return sorted(candidates,key=lambda k:hashlib.sha256((common.SALT+k).encode()).hexdigest())[:64]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    old=json.loads((common.ROOT/'configs/0012-development-manifest.json').read_text())
    if a.data.stat().st_size!=old['dataset_bytes'] or common.sha256(a.data)!=old['dataset_sha256']: raise ValueError('Dataset differs')
    import h5py
    with h5py.File(a.data,'r') as h: keys=list(h.keys())
    chosen=select(keys,old['selected_pair_keys'])
    common.write_new(a.output,{'schema':'evr0016.development.v1','dataset_bytes':old['dataset_bytes'],
        'dataset_sha256':old['dataset_sha256'],'development_scene':'0015','development_scene_pairs':600,
        'selected_pair_keys':chosen,'selection_salt':common.SALT,'selection':'64 lowest salted SHA256 from584 previously unread pair keys',
        'excluded_manifest_sha256':common.sha256(common.ROOT/'configs/0012-development-manifest.json'),
        'root_name_inventory_sha256':hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest(),
        'held_out_scenes':['0022'],'coordinate_convention':'raw_pixels','access':'Only root names plus opaque whole-file integrity; no scientific array values'})
    print(json.dumps({'selected':len(chosen),'sha256':common.sha256(a.output),'numeric_arrays_read':False}))

if __name__=='__main__': main()
