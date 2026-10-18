# 0010 capability integration proposal

**Software only; not frozen or admitted for execution.** No download, installation,
image decoding, tokenizer, checkpoint, model or quality run was performed to make
this directory. The parent owns the canonical protocol, published freeze and
`results/0010-admission/` evidence. No StatA solver is integrated here.

## Proposed repository layout

- `experiments/0010/acquire.py`: exact reviewed stdlib downloader, unchanged.
- `run_capability.py`: reviewed probe, plus separately recorded module and
  distribution versions. `PIL` maps to distribution `Pillow`; config pins the
  distribution version, including regex's release version. Both observed values
  and import paths are retained before rejecting a version mismatch.
- `cli.py`: small fixed-stage caller; explicitly checks helper/manifest hashes,
  then uses the existing `scripts/supervise_0009.py` unchanged. Public stages are
  `wheels`, `assets`, `probe`; underscored forms are supervised child entrypoints.
- `bundle/`: five exact MIT CLIP source/license files and the three already
  acquired spatial filename lists. `selection.json` is their fixed eight-name
  selection, unchanged. `bundle-manifest.json` gives hashes and provenance URLs.
- `dependencies.json`: only the four qualified wheels, no extras.
- `downloads.json`: BPE, EuroSAT ZIP, ViT-B/32 checkpoint, and the small
  exact StatA EuroSAT literal source; no StatA source is redistributed here.
- `layout.json`: eight bundled files plus four external assets map to the fixed
  capability layout. The downloader uses a dedicated flat cache; after successful
  acquisition all bytes are rechecked and hard-linked into a fresh asset root.
  Hard links require one filesystem; an EXDEV error stops, with no copy fallback.

Fresh layout directory and exclusive reservation marker are created **before**
asset acquisition. Existing directories, partials, operation markers, symlinks,
changed source/bytes, cross-stage overlap and publication races stop rather than
resume/overwrite. A failed stage leaves evidence and does not authorize retry.

## Runtime and remaining admission work

The parent reports a real Linux import preflight: Python 3.12, Torch 2.10.0+cpu,
NumPy 2.0.2, torchvision 0.25.0+cpu, Pillow 13.0.0 and packaging 26.3, affinity
0–3 and 8 GiB AS/cgroup limits. Those are the known config pins; this local
integration did not independently reproduce that cloud environment.

PyPI metadata qualified ftfy 6.3.1, regex 2026.9.29 (CPython 3.12 x86-64 manylinux),
tqdm 4.70.1 and wcwidth 0.9.2. Wheel bytes and successful offline installation/
imports remain **UNKNOWN/unexecuted**. No replacement torch, NumPy, torchvision
or CUDA wheel is permitted. wcwidth stays in the dependency receipt; it is not
added to the model/data asset schema.

`config.json` contains prospective exact distribution versions, not proof
that missing packages are installed. Source/config/protocol may be frozen after
parent review of the fixed dependencies, budgets and source; installation is not
a prerequisite for that source freeze. Actual probing additionally requires the
separately captured successful offline installation and all import checks under
those already fixed versions.

One actual no-follow HEAD of the pinned HF ZIP returned HTTPS 302 to
`us.aws.cdn.hf.co`, with matching repository commit, linked size and linked SHA256;
zero response-body bytes were read. Therefore the ZIP allowlist is exactly
`huggingface.co` and `us.aws.cdn.hf.co`. No ZIP payload or redirected GET was tested.
Any other redirect host still fails closed; no post-failure broadening is allowed.

The public CLI requires the expected config SHA-256, a fresh external-supervision
output, and an existing Python path. It invokes the unchanged supervisor with
900 seconds for each preparation operation and 300 seconds for the single probe,
4 CPU slots and 8 GiB address space. The child checks cgroup memory.max before ML
imports. The canonical published-freeze check is a parent admission prerequisite,
not a claim that this unfrozen CLI independently establishes publication.

## Separate response and scientific-asset budgets

The new DIRECT application-read response-body ceiling is 536,870,912 bytes.
Conservative known charges are 192,952 source/scouting bytes, 523,363 spatial-list
and redirect bytes, and 4,519,443 PyPI metadata bytes including the preserved
failed all-history regex request: total **5,235,758**, remaining **531,635,154**.
The wheel manifest reserves **4,194,304** and the asset manifest **527,440,850**;
these sum exactly to that remainder. Unused wheel allowance is not reassigned.
Expected wheel payload is 1,234,470 bytes; the four external asset payloads total
449,615,692. Redirect/error/overflow-read bytes count within each subcap. Any
further relevant metadata response must be charged before final freeze.

This is not a bound on network-wire or all browser traffic. Historical all-HTTP
traffic remains UNKNOWN. The separate original scientific data/weight ceiling is
10 GiB, with the supplied conservative historical asset bound 7,532,690,072 bytes
and known directory increment 163,286 kept distinct. No historical failure is
reclassified as a pass. See `budget.json`; central receipts belong to the
parent's admission evidence, not duplicated here.

## Future commands, not executed

After published admission, use `python experiments/0010/cli.py wheels` or `assets`
with `--config`, `--config-sha256`, `--cache`, `--supervision`, `--python`;
`assets` also needs `--assetdir`. Cache roots and supervision outputs are distinct.
The `probe` stage needs the config flags plus `--assetdir`, `--output`,
`--supervision`, `--python`. The selection is always the pinned local file.

Before acquisition, inspect only the four exact pinned wheel filenames in the
known `bootstapir-cache/wheels`, `tapnextpp-cache/wheels`,
`tapnextpp-recovery/wheels` and proposed 0010 cache. Missing imports do not prove
wheel files absent: the earlier bounded inventory pruned wheel directories.
Verify size/hash only for exact matches and, if reusable, pre-seed the dedicated
flat cache with a no-overwrite copy/link before calling the downloader. Existing
operation markers or partial/mismatched files remain untouched. Apply the same
named-path check to the exact checkpoint/ZIP paths; a bounded earlier inventory
is not an assertion that every workspace/cache lacks them. No recursive sweep or
rehash of unrelated files is part of this recipe. The downloader rechecks any
pre-seeded file and charges zero HTTP bytes for verified reuse.

There is deliberately **no automatic installer**. After verified wheel acquisition,
reserve an entirely fresh 0010 overlay, then the offline recipe is:

```sh
python -m pip install --no-deps --no-index --disable-pip-version-check \
  --target NEW_EMPTY_0010_OVERLAY \
  VERIFIED_WHEEL_CACHE/ftfy-6.3.1-py3-none-any.whl \
  VERIFIED_WHEEL_CACHE/regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
  VERIFIED_WHEEL_CACHE/tqdm-4.70.1-py3-none-any.whl \
  VERIFIED_WHEEL_CACHE/wcwidth-0.9.2-py3-none-any.whl
```

The parent must enforce freshness, capture real command/exit and import origins,
and prepend only that verified overlay to the prior verified runtime paths.
No prior overlay is altered. This illustrative recipe is not an execution receipt.

## Verification and rights

`PYTHONDONTWRITEBYTECODE=1 python3 -W error -m unittest discover -s experiments/0010 -p 'test_*.py' -v`
ran **54 artificial tests**, 0.084 s unittest time, exit 0: original 34 downloader,
10 capability, two distribution-version checks, and eight portable-layout/caller
checks. Fake requests and fake supervision only; no actual ML import or network.
Raw logs and the integration receipt remain outside the proposed repo tree for
parent review. Earlier prototype receipts remain unchanged in their original
locations and a separate local historical copy; no old success is relabeled.

Bundled OpenAI CLIP code remains MIT with its exact LICENSE, pinned commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. Spatial-list metadata is pinned to
TorchGeo EuroSAT `f9fa29ffb3980d9959c3d177dd2bc7c71ffeddcc`, whose provider card
states MIT. The original StatA literal source, if later acquired, is AGPL-3.0 and
is parsed as AST data only; this does not relicense it or bundle a StatA solver.
Filename selection exposes class words in filenames; no claim of label-blind
metadata access or accuracy evaluation is made. Only the eight fixed train
members may later be decoded; class tokens do not change the selection rule.
