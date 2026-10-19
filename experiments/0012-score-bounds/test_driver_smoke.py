"""Synthetic-only smoke test: 16 groups, forbidden labels and held-out values."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import run_development as driver


def main():
    rng = driver.np.random.default_rng(32026)
    keys = ["synthetic_dev/pair_%02d" % i for i in range(16)]
    reads = []
    original_getitem = driver.h5py.Dataset.__getitem__

    def guarded_read(dataset, *args, **kwargs):
        reads.append(dataset.name)
        assert dataset.name in {"/" + k + "/" + f for k in keys for f in ("x1", "x2")}
        return original_getitem(dataset, *args, **kwargs)

    with tempfile.TemporaryDirectory(prefix="0012-synthetic-", dir=Path(__file__).resolve().parents[3]) as directory:
        root = Path(directory)
        data, manifest, output = (root / n for n in ("fake.h5", "manifest.json", "out.jsonl"))
        with driver.h5py.File(data, "w") as handle:
            for key in keys + ["synthetic_heldout/pair_00"]:
                group = handle.create_group(key)
                for field in ("x1", "x2"):
                    group.create_dataset(field, data=rng.uniform(0, 80, size=(40, 2)))
                for field in ("R", "t", "K1", "K2", "confidence"):
                    group.create_dataset(field, data=[999.0])
        manifest.write_text(json.dumps({"schema": "evr0012.development.v1",
            "selected_pair_keys": keys, "coordinate_convention": "raw_pixels",
            "dataset_bytes": data.stat().st_size, "synthetic_only": True}))
        argv = ["run_development.py", "--data", str(data), "--manifest", str(manifest), "--output", str(output)]
        with patch.object(sys, "argv", argv), patch.object(driver.h5py.Dataset, "__getitem__", guarded_read), redirect_stdout(io.StringIO()):
            assert driver.main() == 0
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(reads) == 32 and len(set(reads)) == 32
        assert len(records) == 18 and records[0]["type"] == "start"
        assert records[0]["max_refinements"] == 128
        assert all(int(value) <= 4 for value in records[0]["thread_limits"].values())
        for record in records[1:-1]:
            assert record["status"] == "completed" and not record["bound_violations"]
            assert len(record["proposals"]) == 64 and len(record["exact_scores"]) == 64
            assert all(record["rankings"][m]["agrees_with_exact_winner"] for m in ("moments", "culling"))
            assert record["containment_cell_checks_per_mode"] == record["initial_cells"] * 64
        assert records[-1] == {"type": "complete", "completed_pairs": 16, "pose_accuracy_evaluated": False}
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1", "Run smoke test with caller thread limit 1"
        print(json.dumps({"synthetic_pairs": 16, "coordinate_dataset_reads": len(reads),
            "forbidden_reads": 0, "schema_and_winners": "passed", "caller_thread_limit": 1}))


if __name__ == "__main__":
    main()
