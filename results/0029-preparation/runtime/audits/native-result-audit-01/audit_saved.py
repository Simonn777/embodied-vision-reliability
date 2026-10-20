"""Saved manufactured qualification audit; stdlib and NumPy, no model imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

P = Path(__file__).resolve().parents[3]
R = P / "work/embodied-vision-reliability"
BASE = P / "work/0029-runtime-02"
OUT = Path(__file__).resolve().parent
COMMIT = "37b54982bcc3b22b56a5817b6881f22489eba8d5"
SUP = "experiments/0027-affine-tangent/supervise.py"
MON = "experiments/0025c-rs-noise/supervise.py"
THREADS = {k: "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
report = {"schema": "evr0029.native-saved-audit.v1", "passed": False, "checks": 0,
          "failures": [], "source_commit": COMMIT,
          "scope": "saved manufactured outputs; no native/model execution",
          "limitations": ["Original in-memory state and attention tensors were not exported; "
                         "their hash/unchanged claims are source-bound execution evidence. "
                         "This audit independently checks saved logits and trace consistency.",
                         "JPEG bytes are rehashed, not decoded or regenerated.",
                         "CI is checked against the preserved CI response, not a new network query."]}
originals = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, label):
    report["checks"] += 1
    if not ok:
        report["failures"].append(label)
    return ok


def load(path):
    originals[str(path.relative_to(P))] = digest(path)
    return json.loads(path.read_text())


def safe(folder, name):
    path = folder / name
    if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
        raise ValueError("unsafe saved path")
    return path


def sources(mapping, commit):
    for rel, expected in mapping.items():
        raw = subprocess.run(["git", "-C", str(R), "show", commit + ":" + rel],
                             check=True, capture_output=True).stdout
        check(hashlib.sha256(raw).hexdigest() == expected, "frozen source " + rel)
        check(digest(safe(R, rel)) == expected, "current source " + rel)
        originals[str((R / rel).relative_to(P))] = expected


def outer(base, name, result, commit):
    folder = base / (name + "-supervision")
    rec = load(folder / "receipt.json")
    spec_path = base / (name + "-spec.json")
    spec = load(spec_path)
    check(rec["status"] == "completed" and rec["exit_code"] == 0
          and rec["termination"] is None and rec["remaining_descendant_pids"] == []
          and rec["cleanup_errors"] == [], name + " clean outer")
    check(rec["expected_commit"] == spec["commit"] == commit
          and rec["output_sha256"] == digest(result)
          and rec["spec_sha256"] == digest(spec_path), name + " hashes/commit")
    check(spec["repo"] == str(R) and spec["result_file"] == str(result)
          and spec["operation_output"] == str(folder) and spec["timeout"] == 1800,
          name + " spec paths/limit")
    command = [x.replace(str(R), "$REPOSITORY").replace(str(R.parent), "$PRIVATE")
               for x in spec["command"]]
    check(rec["command"] == command, name + " exact command")
    check(rec["hard_timeout_seconds"] == 1800 and rec["group_rss_limit_kib"] == 8388608
          and all(type(rec[k]) in (int, float) and math.isfinite(rec[k]) and 0 <= rec[k] <= limit
                  for k, limit in (("wall_seconds", 1800), ("peak_group_rss_kib", 8388608)))
          and rec["thread_environment"] == THREADS
          and all(spec["environment"][k] == v for k, v in THREADS.items()), name + " resources")
    check(set(rec["log_sha256"]) == {"stdout.log", "stderr.log"}, name + " log set")
    for filename, value in rec["log_sha256"].items():
        check(digest(folder / filename) == value, name + " log " + filename)
        originals[str((folder / filename).relative_to(P))] = value
    for key, rel in (("supervisor_sha256", SUP), ("monitor_sha256", MON)):
        sources({rel: rec[key]}, commit)
    return rec


def main():
    qualification = BASE / "qualification"
    fixture = qualification / "native-fixture"
    installed = BASE / "installation"
    q = load(qualification / "receipt.json")
    f = load(fixture / "fixture-record.json")
    inst = load(installed / "install-receipt.json")
    qo = outer(BASE, "qualification", qualification / "receipt.json", COMMIT)
    io = outer(BASE, "installation", installed / "install-receipt.json", COMMIT)
    check(q["status"] == "completed" and q["eligible"] is True
          and not q["offline_attempts"] and q["scientific_input_access"] is False,
          "qualification admitted without forbidden attempt")
    check(inst["status"] == "completed" and inst["model_invocations"] == 0
          and inst["scientific_input_access"] is False, "installation scope")
    check(q["native_fixture"] == f and f["status"] == "completed"
          and f["manufactured_only"] is True, "identical completed fixture copies")
    binding = q["installation_bindings"]
    check(binding == {"receipt_sha256": digest(installed / "install-receipt.json"),
          "supervision_sha256": digest(BASE / "installation-supervision/receipt.json"),
          "spec_sha256": digest(BASE / "installation-spec.json"),
          "inventory_entries": len(inst["installed_inventory"])}, "installation linkage")
    sources(q["source_sha256"], COMMIT)
    sources(inst["source_sha256"], COMMIT)
    ci = load(BASE / "source-ci.json")
    check(ci["headSha"] == COMMIT and ci["status"] == "completed"
          and ci["conclusion"] == "success", "preserved exact CI response")
    acq = inst["acquisition_bindings"]
    acqbase = P / "work/0029-acquisition-01"
    for key, rel in (("operation_sha256", "assets/operation.json"),
                     ("outer_sha256", "supervision/receipt.json"), ("spec_sha256", "spec.json")):
        check(digest(acqbase / rel) == acq[key], "acquisition linkage " + key)
    for name, value in acq["batch_ledger_sha256"].items():
        check(digest(acqbase / "assets" / name / "ledger.json") == value, "acquisition ledger " + name)
    for name, value in acq["log_sha256"].items():
        check(digest(acqbase / "supervision" / name) == value, "acquisition log " + name)
    sources(acq["source_sha256"], acq["commit"])
    inv = inst["installed_inventory"]
    observed = [installed / "offline_guard.py"]
    for dirname in ("sources", "weights", "venv", "built-wheels"):
        observed.extend(x for x in (installed / dirname).rglob("*") if x.is_file() or x.is_symlink())
    check({str(x.relative_to(installed)) for x in observed} == set(inv), "complete installed inventory")
    for rel, item in inv.items():
        path = installed / rel
        if item["type"] == "symlink":
            check(path.is_symlink() and os.readlink(path) == item["target"]
                  and digest(path.resolve()) == item["resolved_sha256"], "installed link " + rel)
        else:
            check(not path.is_symlink() and path.stat().st_size == item["bytes"]
                  and digest(path) == item["sha256"], "installed file " + rel)
    normalize = lambda x: re.sub(r"[-_.]+", "-", x).lower()
    manifest = load(R / "configs/0029-runtime-assets.json")
    expected = {(normalize(x["name"]), x["version"]) for x in
                manifest["assets"] + manifest["non_downloaded_inputs"]["wheels"]}
    check(len(inst["installed_distributions"]) == len(expected) == 26
          and {(normalize(x["name"]), x["version"]) for x in inst["installed_distributions"]} == expected,
          "exact installed distribution closure")
    if report["failures"]:
        return
    # Installed NumPy only, without processing site/.pth or importing Torch.
    sys.path.insert(0, str(installed / "venv/lib/python3.12/site-packages"))
    import numpy as np
    report["numpy_version"] = np.__version__
    check(Path(np.__file__).resolve().is_relative_to(installed.resolve()), "NumPy origin")
    arrays = {}
    check(len(f["inputs"]) == 9 and len(f["outputs"]) == 24, "fixture file counts")
    for item in f["inputs"] + f["outputs"]:
        path = safe(fixture, item["filename"])
        check(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
              "fixture bytes " + item["filename"])
        originals[str(path.relative_to(P))] = item["sha256"]
        if "arrays" not in item:
            continue
        with np.load(path, allow_pickle=False) as data:
            check(set(data.files) == set(item["arrays"]), "NPZ fields " + item["filename"])
            values = {key: data[key] for key in data.files}
        for key, value in values.items():
            meta = item["arrays"][key]
            check(list(value.shape) == meta["shape"] and str(value.dtype) == meta["dtype"]
                  and hashlib.sha256(value.tobytes(order="C")).hexdigest() == meta["raw_c_order_sha256"],
                  "raw array " + item["filename"] + ":" + key)
            check(np.isfinite(value).all(), "finite array " + item["filename"] + ":" + key)
        arrays[item["filename"]] = values
    expected_files = {x["filename"] for x in f["inputs"] + f["outputs"]} | {"fixture-record.json"}
    check({str(x.relative_to(fixture)) for x in fixture.rglob("*") if x.is_file()} == expected_files,
          "complete fixture file set")
    def equal(left, right, key="logits"):
        a, b = arrays[left][key], arrays[right][key]
        check(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
              "bitwise parity " + left + " == " + right)
    for name in ("probe-0-full.npz", "probe-5-full.npz"):
        equal(name, "baseline-07.npz")
    equal("probe-1-remove_6.npz", "probe-4-remove_6.npz")
    equal("probe-2-remove_4.npz", "probe-3-remove_4.npz")
    equal("future-after-probes-08.npz", "baseline-08.npz")
    check([x["frame"] for x in f["baseline"]] == list(range(9)), "all nine baseline frames")
    check([x["removed_frame"] for x in f["probes"]] == [None, 6, 4, 4, 6, None], "fixed probe order")
    for row in f["probes"]:
        trace, drop = row["trace"], row["removed_frame"]
        spatial = [x for x in range(7) if x != drop]
        pointers = [0] + [x for x in range(6, 0, -1) if x != drop]
        expected_blocks = [("spatial", x, 4096) for x in spatial] + [("pointer", x, 4) for x in pointers]
        check(trace["spatial_frames"] == spatial and trace["pointer_frames"] == pointers
              and trace["num_obj_ptr_tokens"] == 4 * len(pointers), "carrier identities " + str(row["order"]))
        check(len(trace["blocks"]) == len(expected_blocks), "carrier count " + str(row["order"]))
        cursor = 0
        for block, (kind, frame, size) in zip(trace["blocks"], expected_blocks):
            check((block["kind"], block["frame"], block["start"], block["end"]) ==
                  (kind, frame, cursor, cursor + size), "carrier range " + str(row["order"]) + ":" + str(cursor))
            check(all(re.fullmatch(r"[0-9a-f]{64}", block[k]) for k in ("values_sha256", "position_sha256")),
                  "carrier digest syntax")
            cursor += size
        check(trace["total_tokens"] == cursor and row["status"] == "completed"
              and row["exact_parity"] and row["state_unchanged"]
              and row["maximum_absolute_parity_error"] == 0, "probe execution proof")
    full = {(b["kind"], b["frame"]): b for b in f["probes"][0]["trace"]["blocks"]}
    for row in f["probes"][1:]:
        for block in row["trace"]["blocks"]:
            ref = full[(block["kind"], block["frame"])]
            check(all(block[k] == ref[k] for k in ("values_sha256", "position_sha256")),
                  "retained carrier hashes")
    for a, b in ((0, 5), (1, 4), (2, 3)):
        check(f["probes"][a]["trace"] == f["probes"][b]["trace"], "reversed carrier parity")
    reset = f["initial_memory_prompt_trace"]
    check(reset["spatial_frames"] == reset["pointer_frames"] == [0]
          and reset["num_obj_ptr_tokens"] == 4 and reset["total_tokens"] == 4100,
          "initial-memory-only carriers")
    for b in reset["blocks"]:
        check(all(b[k] == full[(b["kind"], 0)][k] for k in ("values_sha256", "position_sha256")),
              "initial-memory retained hashes")
    mask = arrays["initial-mask.npz"]["mask"]
    want = np.zeros((384, 512), bool); want[118:182, 207:271] = True
    check(np.array_equal(mask, want), "exact manufactured initial mask")
    bank = arrays["native-video-decoder-bank.npz"]
    quality = bank["predicted_quality"]; selected = int(np.argmax(quality[0]))
    check(quality.ndim == 2 and quality.shape[0] == 1 and quality.shape[1] >= 1
          and f["native_video_decoder_bank"]["count"] == quality.shape[1]
          and selected == f["native_video_decoder_bank"]["selected_index"], "actual decoder count/quality selection")
    for resolution in ("low", "high", "original"):
        all_masks = bank[resolution + "_resolution_logits"]
        chosen = bank["selected_" + resolution + "_resolution_logits"]
        check(np.array_equal(chosen, all_masks[:, selected:selected + 1]), "decoder chosen " + resolution)
    check(np.array_equal(bank["selected_original_resolution_logits"], arrays["baseline-07.npz"]["logits"]),
          "native decoder baseline parity")
    feature = [arrays["features-%d.npz" % frame] for frame in (0, 8)]
    matches = arrays["first-last-matches.npz"]
    pair = matches["matches"][0] if matches["matches"].ndim == 3 else matches["matches"]
    check(pair.ndim == 2 and pair.shape[1] == 2 and len(pair) == f["match_count"] >= 1
          and pair.dtype.kind in "iu", "compact native matches")
    for side in (0, 1):
        n = feature[side]["keypoints"].shape[1]
        check(0 < n <= 2048 and np.array_equal(feature[side]["image_size"], [[512, 384]]),
              "feature original coordinates/dimensions")
        check(np.all((pair[:, side] >= 0) & (pair[:, side] < n))
              and len(np.unique(pair[:, side])) == len(pair), "match bounds/uniqueness")
        dense = matches["matches%d" % side][0]
        check(dense.shape == (n,) and np.array_equal(dense[pair[:, side]], pair[:, 1-side])
              and np.array_equal(np.flatnonzero(dense >= 0), np.sort(pair[:, side])),
              "complete dense/compact mutuality")
    check(all(t["status"] == "completed" and math.isfinite(t["seconds"]) and t["seconds"] >= 0
              for t in f["timings"]), "all stage timings")
    check(f["execution"]["torch_intraop"] == 4 and f["execution"]["torch_interop"] == 1
          and f["execution"]["opencv_threads"] == 1
          and f["native_model"]["observed_compact_memory_dtypes"] == ["torch.bfloat16"],
          "declared observed execution settings")
    check(f["future_after_discarded_probes"] == {
        "frame": 8, "exact_logit_parity": True, "maximum_absolute_parity_error": 0.0,
        "exact_complete_state_parity": True, "query_checkpoint_unchanged": True,
        "original_baseline_unchanged": True}, "source-bound future state isolation claims")
    check(all(re.fullmatch(r"[0-9a-f]{64}", f[key]) for key in
              ("state_after_baseline_sha256", "state_after_frame_7_sha256")),
          "source-bound state digest syntax")
    report["summary"] = {"installed_inventory_entries": len(inv), "distribution_count": 26,
        "fixture_inputs": 9, "fixture_outputs": 24, "baseline_frames": 9, "probes": 6,
        "native_bank_count": quality.shape[1], "native_predicted_ious": quality.tolist(),
        "native_selected_index": selected, "mutual_lightglue_matches": len(pair),
        "qualification_wall_seconds": qo["wall_seconds"], "qualification_peak_rss_kib": qo["peak_group_rss_kib"],
        "installation_wall_seconds": io["wall_seconds"], "fixture_warnings": f["warnings"]}
    for rel, expected in originals.items():
        check(digest(P / rel) == expected, "original unchanged " + rel)
    report["original_bindings"] = originals


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        report["failures"].append(type(error).__name__ + ": " + str(error))
    report["source_sha256"] = digest(Path(__file__))
    report["passed"] = not report["failures"]
    with (OUT / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "failures")}))
    raise SystemExit(0 if report["passed"] else 1)
