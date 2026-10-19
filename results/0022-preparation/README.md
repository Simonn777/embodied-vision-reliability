# 0022 frontend preparation: asset-only admission

This record freezes acquisition of a traceable per-image MoGe2-S and
SuperPoint/LightGlue frontend. It is not a pose experiment or method result.
No benchmark inputs were downloaded or allocated at this freeze. The later
installation and manufactured-inference implementations require separate review
and source freezes before execution.

The exact manifests list 105 source/model files totaling **194,142,010 bytes**
and 14 dependency wheels totaling **76,809,693 bytes**. Source Git blob IDs,
MoGe publisher LFS SHA256, and PyPI wheel SHA256 are known before acquisition.
The two official LightGlue-release assets have no publisher digest; their
first-download hashes will be explicitly labeled and bound for later use.
The combined response-body ceiling is 300,000,000 bytes. The protocol additionally
reserves 20,000,000 bytes for recent/follow-up metadata, including unmetered web
reads. No model/source assets will be redistributed in the research repository.

The adapted stdlib acquisition helper passes **38 artificial tests**. The initial
run and saved verification both returned 0; the saved log and actual subprocess
receipt are `acquisition-tests.log` and `acquisition-tests.json`. Tests use only
invented byte strings, fake HTTP responses and temporary files. They cover byte
and digest failure retention, redirect hosts/TLS, deadline/error accounting,
atomic no-overwrite publication and explicit first-download provenance. These
are correctness checks, not model or speed results.

The [source import contract](../../docs/literature/0022-runtime-source-contract.md)
and [checkpoint contract](checkpoint-and-runtime-contract.md) identify the full
minimal dependency closure and hidden download/optional-install paths. The
14-wheel metadata manifest verifies packaging constraints against the existing
Python/Torch/NumPy versions; native runtime compatibility is not yet measured.
No existing shared environment is modified.

Before any real experiment, complete the [protocol](../../protocols/0022-traceable-frontend-assets.md),
including local weight checks, blocked implicit network, one depth realization
per artificial image and measured full-process resources. Then separately freeze
scientific scenes, observations, methods and same-information controls. The
original efficacy/novelty gate remains unmet; no manuscript.
