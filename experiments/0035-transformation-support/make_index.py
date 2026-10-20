"""Build prediction metadata from complete initial/RGB acquisition receipts."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def run(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    loader = importlib.util.spec_from_file_location("index_admission", spec["admit_path"])
    admit = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = admit
    loader.loader.exec_module(admit)
    out = Path(spec["output"])
    admit.require(not out.exists(), "fresh index output required")
    fingerprints = {}
    admit.require(spec["manifest"]["sha256"] == admit.MANIFEST_SHA256, "fixed manifest")
    manifest = admit.bound_json(spec["manifest"], fingerprints)
    _, initial = admit.acquisition(spec["initial_acquisition"], spec["initial_outer"],
                                   "initial", manifest, fingerprints, spec["repo"])
    rgb_receipt, rgb = admit.acquisition(spec["rgb_acquisition"], spec["rgb_outer"],
                                        "rgb", manifest, fingerprints, spec["repo"])
    inspection = admit.bound_json(spec["initial_inspection"], fingerprints)
    admit.clean_outer(spec["initial_inspection_outer"], spec["initial_inspection"]["sha256"], fingerprints)
    admit.require(inspection["status"] == "completed" and inspection["eligible"] is True
                  and inspection["source_sha256"] == spec["source_sha256"]
                  and rgb_receipt["initial_admission"]["inspection_sha256"] == spec["initial_inspection"]["sha256"],
                  "initial resource admission before RGB")
    sequences = []
    for name, count in admit.COHORT:
        sequences.append(dict(sequence=name, role="TRAIN", initial=initial[(name, 0)],
                              rgb=[rgb[(name, frame)] for frame in range(count)]))
    value = dict(schema="evr0035.predictive-input-index.v1", sequences=sequences,
                 noninitial_labels_included=False, spec_sha256=admit.sha(spec_path),
                 source_sha256=spec["source_sha256"], fingerprints=fingerprints,
                 prediction_output_cap_bytes=inspection["prediction_output_cap_bytes"])
    admit.require(all(admit.sha(p) == h for p, h in spec["source_sha256"].items()), "index source changed")
    admit.postcheck(value)
    with out.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(path=str(out), bytes=out.stat().st_size, sha256=admit.sha(out), frames=446)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    print(json.dumps(run(parser.parse_args().spec)))
