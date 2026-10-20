"""Independent saved-only first-stage audit. Never imports a model or selector.

Prepared before outcomes. Run only after the parent admits completion. The three
specifications determine the allowed acquisition/evidence roots; no query-GT
argument exists. Only six first masks are decoded. RGB bytes are hashed and
headers inspected, without pixel decoding. NumPy/Pillow use the qualified runtime.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

COHORT = (("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
          ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86))
OWN = "experiments/0029-memory-information/"
MANIFEST = "configs/0029-davis-development.json"
PROTOCOL = "protocols/0029-signed-memory-diagnostic.md"
RANGES = "experiments/0023-mapfree/range_inputs.py"
SUP = "experiments/0027-affine-tangent/supervise.py"
MON = "experiments/0025c-rs-noise/supervise.py"
NATIVE_COMMIT = "37b54982bcc3b22b56a5817b6881f22489eba8d5"
THREADS = {k: "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
ACQ_SOURCES = {OWN + "acquire_davis.py", RANGES, MANIFEST, PROTOCOL}
EVIDENCE_SOURCES = {OWN + n for n in ("run_evidence.py", "frontend_evidence.py",
                                    "diagnostic_core.py", "acquire_davis.py")}
EVIDENCE_SOURCES |= {MANIFEST, PROTOCOL, SUP, MON}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def schedule(n):
    return [4 + (2*j*(n-5)+7)//14 for j in range(8)]


def independent_witnesses(labels, first, current, ids, scores, shape, sequence, frame):
    """Scalar nine-pixel labels; group collisions; sort (within-cell rank, cell).

    No production morphology/selector is called. This is an independent
    implementation of the frozen rule, not independent-team replication.
    """
    ih, iw = labels.shape
    h, w = shape
    target = min(int(v) for v in set(labels.flat) if v > 0 and v != 255)
    counts = dict(matches=len(ids), invalid_coordinates=0, boundary_or_void=0, labeled=0,
                  opposite_collision_points=0, same_label_discarded=0, after_collisions=0)
    groups = defaultdict(list)
    for index, pair in enumerate(ids):
        source = [float(v) for v in first[int(pair[0])]]
        dest = [float(v) for v in current[int(pair[1])]]
        if not (all(math.isfinite(v) for v in source + dest)
                and 0 <= source[0] < iw and 0 <= source[1] < ih
                and 0 <= dest[0] < w and 0 <= dest[1] < h):
            counts["invalid_coordinates"] += 1
            continue
        sx, sy, tx, ty = [math.floor(v + .5) for v in source + dest]
        if not (0 <= sx < iw and 0 <= sy < ih and 0 <= tx < w and 0 <= ty < h):
            counts["invalid_coordinates"] += 1
            continue
        neighborhood = [0 <= yy < ih and 0 <= xx < iw and int(labels[yy, xx]) == target
                        for yy in range(sy-1, sy+2) for xx in range(sx-1, sx+2)]
        label = 1 if all(neighborhood) else (
            0 if not any(neighborhood) and int(labels[sy, sx]) != 255 else None)
        if label is None:
            counts["boundary_or_void"] += 1
            continue
        cy, cx = math.floor(8*dest[1]/h), math.floor(8*dest[0]/w)
        row = dict(match_index=index, initial_keypoint_index=int(pair[0]),
                   current_keypoint_index=int(pair[1]), label=label, source_xy=source,
                   target_xy=dest, rounded_current_xy=[tx, ty], cell=[cy, cx],
                   confidence=float(scores[index]))
        groups[tx, ty].append(row)
        counts["labeled"] += 1
    survivors = []
    for rows in groups.values():
        if len({r["label"] for r in rows}) == 2:
            counts["opposite_collision_points"] += len(rows)
        else:
            survivors.append(sorted(rows, key=lambda r: (-r["confidence"], r["match_index"]))[0])
            counts["same_label_discarded"] += len(rows)-1
    counts["after_collisions"] = len(survivors)
    selected = {}
    for label, name in ((1, "positive"), (0, "negative")):
        buckets = defaultdict(list)
        for row in survivors:
            if row["label"] == label:
                buckets[8*row["cell"][0]+row["cell"][1]].append(row)
        ranked = []
        for cell, rows in buckets.items():
            for rank, row in enumerate(sorted(rows, key=lambda r: (-r["confidence"], r["match_index"]))):
                ranked.append((rank, cell, row))
        selected[name] = [v[2] for v in sorted(ranked, key=lambda v: v[:2])[:32]]
        counts[name + "_before_cap"] = len(ranked)
        counts[name + "_retained"] = len(selected[name])
    return dict(target_id=target, initial_shape=[ih, iw], current_shape=list(shape), counts=counts,
                **selected, actionable=len(selected["positive"]) >= 4 and len(selected["negative"]) >= 4,
                sequence=sequence, frame=frame)


class Audit:
    def __init__(self, args):
        self.args, self.root = args, Path(args.repo_root).resolve()
        self.bindings = {}
        self.report = dict(schema="evr0029.frontend-saved-audit.v1", passed=False, checks=0,
            failures=[], rows=[], source_commit=args.expected_commit,
            scope="saved first-stage artifacts, independent witness arithmetic, six initial masks only",
            limitations=["Does not re-extract features, rematch, run a model or assess match correctness.",
                         "HTTP identity/compressed-body digests are retained receipt evidence; response bodies were not separately saved.",
                         "RGB encoded hashes and header dimensions are checked without pixel decoding; only initial PNG IDs are decoded.",
                         "Coverage is a necessary witness-availability screen, not segmentation gain or independent validation."])

    def check(self, value, name, fatal=True):
        self.report["checks"] += 1
        if not bool(value):
            self.report["failures"].append(name)
            if fatal:
                raise ValueError(name)

    def file(self, path, digest=None, size=None):
        path = Path(path).absolute()
        self.check(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular file: " + str(path))
        observed = sha(path)
        self.check((digest is None or observed == digest) and (size is None or path.stat().st_size == size),
                   "file binding: " + str(path))
        self.bindings[str(path)] = observed
        return path

    def load(self, path):
        return json.loads(self.file(path).read_text(),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

    def child(self, root, name):
        rel = Path(name)
        self.check(isinstance(name, str) and not rel.is_absolute() and ".." not in rel.parts, "safe relative path")
        return Path(root) / rel

    def sources(self, mapping, commit, expected=None):
        self.check(re.fullmatch("[0-9a-f]{40}", commit), "full commit")
        if expected is not None:
            self.check(set(mapping) == set(expected), "exact source set")
        for rel, digest in mapping.items():
            path = self.child(self.root, rel)
            frozen = subprocess.run(["git", "-C", str(self.root), "show", commit + ":" + rel],
                                    check=True, capture_output=True, timeout=15).stdout
            self.check(hashlib.sha256(frozen).hexdigest() == digest, "frozen source: " + rel)
            self.file(path, digest)

    def outer(self, spec_path, commit, script, required_flags):
        spec = self.load(spec_path)
        command = spec["command"]
        self.check(isinstance(command, list) and len(command) >= 3
                   and command[1] in ("-B", "-S") and command[2] == str(self.root / script), "operation entrypoint")
        tokens = command[3:]
        self.check(len(tokens) % 2 == 0 and len(set(tokens[::2])) == len(tokens[::2]), "unique flags")
        flags = dict(zip(tokens[::2], tokens[1::2]))
        self.check(set(flags) == set(required_flags) and flags["--repo-root"] == str(self.root)
                   and flags["--expected-commit"] == commit, "operation flags")
        result = Path(flags["--output-dir"]).resolve()
        self.check(spec["repo"] == str(self.root) and spec["commit"] == commit
                   and spec["timeout"] == 1800 and Path(spec["result_file"]) == result / "receipt.json",
                   "spec identity/result")
        folder = Path(spec["operation_output"])
        outer = self.load(folder / "receipt.json")
        self.check(outer["schema"] == "evr0026.operation.v1" and outer["status"] == "completed"
                   and outer["exit_code"] == 0 and outer["termination"] is None
                   and outer["remaining_descendant_pids"] == outer["cleanup_errors"] == [], "clean outer")
        normalized = [x.replace(str(self.root), "$REPOSITORY").replace(str(self.root.parent), "$PRIVATE") for x in command]
        self.check(outer["command"][1:] == normalized[1:] and outer["command"][0] in (normalized[0], "$PYTHON"),
                   "outer exact command")
        self.check(outer["expected_commit"] == commit and outer["spec_sha256"] == sha(spec_path)
                   and outer["output_sha256"] == sha(result / "receipt.json"), "outer hashes")
        self.check(outer["hard_timeout_seconds"] == 1800 and outer["group_rss_limit_kib"] == 8388608
                   and outer["thread_environment"] == THREADS
                   and all(spec["environment"].get(k) == v for k, v in THREADS.items())
                   and all(type(outer[k]) in (int, float) and math.isfinite(outer[k]) and 0 <= outer[k] <= lim
                           for k, lim in (("wall_seconds", 1800), ("peak_group_rss_kib", 8388608))), "resource bounds")
        self.check(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "outer logs")
        for name, digest in outer["log_sha256"].items():
            self.file(folder / name, digest)
        self.sources({SUP: outer["supervisor_sha256"], MON: outer["monitor_sha256"]}, commit)
        record = self.load(result / "receipt.json")
        self.check(record["status"] == "completed" and record["expected_commit"] == commit, "completed receipt")
        return record, flags, result, outer

    def acquisition(self, spec_path, phase, plan):
        flags = {"--repo-root", "--expected-commit", "--metadata-root", "--phase", "--output-dir"}
        rec, argv, folder, outer = self.outer(spec_path, self.args.expected_commit, OWN+"acquire_davis.py", flags)
        cap, count = (57000000, 386) if phase == "predictive_rgb" else (100000, 6)
        self.check(argv["--phase"] == rec["phase"] == phase
                   and rec["schema"] == "evr0029.davis-acquisition.v1"
                   and rec["archive"] == plan["archive"] and rec["manifest_sha256"] == sha(self.root / MANIFEST)
                   and rec["media_decoded"] is False and rec["gt_admission"] is None
                   and rec["hidden_retries"] is False and rec["full_zip_fallback"] is False, "acquisition scope")
        self.sources(rec["source_sha256"], self.args.expected_commit, ACQ_SOURCES)
        for name, info in plan["source_bindings"].items():
            self.file(self.child(argv["--metadata-root"], name), info["sha256"], info["bytes"])
        expected = [x for x in plan["members"] if x["stage"] == phase]
        self.check(len(expected) == count == len(rec["members"])
                   and rec["selected_names"] == [x["filename"] for x in expected]
                   and len(rec["requests"]) == 3*count, "complete fixed acquisition")
        verified, files, body_total = {}, set(), 0
        for i, (actual, wanted) in enumerate(zip(rec["members"], expected)):
            self.check(all(actual.get(k) == v for k, v in wanted.items()), "member metadata")
            rel = "members/" + wanted["output_relative"]
            self.check(actual["saved_relative"] == rel and actual["bytes"] == wanted["uncompressed_bytes"], "member path/size")
            path = self.file(self.child(folder, rel), actual["sha256"], actual["bytes"])
            self.check("%08x" % (zlib.crc32(path.read_bytes()) & 0xffffffff) == wanted["crc32_hex"], "member CRC")
            verified[wanted["sequence"], wanted["frame_index"]] = (path, actual)
            files.add(str(path.relative_to(folder)))
            start, data, packed = wanted["local_header_offset"], actual["data_offset"], wanted["compressed_bytes"]
            self.check(data == start+30+wanted["filename_bytes"]+actual["local_extra_bytes"]
                       and 0 <= actual["local_extra_bytes"] <= 4096, "member data location")
            for j, (lo, hi) in enumerate(((start, start+29), (start+30, data-1), (data, data+packed-1))):
                req = rec["requests"][3*i+j]
                self.check(req["start"] == lo and req["end"] == hi
                           and req["expected_bytes"] == req["body_bytes"] == hi-lo+1
                           and req["status"] == 206 and req["etag"] == plan["archive"]["etag"]
                           and req["content_length"] == str(hi-lo+1)
                           and req["content_range"] == "bytes %d-%d/%d" % (lo, hi, plan["archive"]["archive_bytes"])
                           and re.fullmatch("[0-9a-f]{64}", req["body_sha256"])
                           and 0 <= req["elapsed_seconds"] <= 30, "exact range receipt")
                body_total += req["body_bytes"]
            self.check(rec["requests"][3*i+2]["body_sha256"] == actual["compressed_sha256"], "compressed body binding")
        self.check({str(p.relative_to(folder)) for p in (folder/"members").rglob("*") if p.is_file()} == files,
                   "no extra acquisition members")
        self.check(rec["response_body_bytes_exact"] is True and rec["response_body_bytes"] == body_total <= cap
                   and rec["requested_body_upper_bound"] == body_total+3*count
                   and rec["output_bytes"] == sum(x["uncompressed_bytes"] for x in expected)
                   and rec["phase_body_cap_bytes"] == cap and rec["current_member"] is None, "phase accounting")
        return verified, folder, rec, argv

    def arrays(self, folder, item, np):
        path = self.file(self.child(folder, item["filename"]), item["sha256"], item["bytes"])
        with np.load(path, allow_pickle=False) as pack:
            self.check(set(pack.files) == set(item["arrays"]), "NPZ field inventory")
            values = {key: pack[key] for key in pack.files}
        for name, value in values.items():
            meta = item["arrays"][name]
            self.check(list(value.shape) == meta["shape"] and str(value.dtype) == meta["dtype"]
                       and hashlib.sha256(value.tobytes(order="C")).hexdigest() == meta["raw_c_order_sha256"]
                       and np.isfinite(value).all(), "NPZ array hash/shape/finite: " + name)
        return values

    def run(self):
        self.check(sys.flags.no_site == 1 and all(os.environ.get(k) == v for k,v in THREADS.items()),
                   "isolated audit launch and thread bounds")
        plan = self.load(self.root / MANIFEST)
        self.check([(x["sequence"], x["frame_count"]) for x in plan["cohorts"]] == list(COHORT), "fixed cohort")
        rgbs, rgbroot, _, rgbargv = self.acquisition(self.args.rgb_spec, "predictive_rgb", plan)
        initials, initialroot, _, initialargv = self.acquisition(self.args.initial_spec, "initial_supervision", plan)
        self.check(rgbroot != initialroot and rgbroot not in initialroot.parents and initialroot not in rgbroot.parents,
                   "disjoint media roots")
        flags = {"--repo-root", "--expected-commit", "--native-root", "--native-commit", "--metadata-root",
                 "--rgb-root", "--rgb-spec", "--initial-root", "--initial-spec", "--output-dir"}
        rec, argv, out, _ = self.outer(self.args.evidence_spec, self.args.expected_commit, OWN+"run_evidence.py", flags)
        self.check(argv["--native-commit"] == NATIVE_COMMIT
                   and Path(argv["--rgb-root"]).resolve() == rgbroot
                   and Path(argv["--initial-root"]).resolve() == initialroot
                   and Path(argv["--rgb-spec"]).resolve() == Path(self.args.rgb_spec).resolve()
                   and Path(argv["--initial-spec"]).resolve() == Path(self.args.initial_spec).resolve()
                   and argv["--metadata-root"] == rgbargv["--metadata-root"] == initialargv["--metadata-root"], "evidence input identity")
        self.check(rec["schema"] == "evr0029.evidence-operation.v1" and rec["eligible"] is True
                   and rec["noninitial_annotation_access"] is False and rec["sam_sequence_invocations"] == 0
                   and rec["offline_attempts"] == [], "evidence scope")
        self.sources(rec["source_sha256"], self.args.expected_commit, EVIDENCE_SOURCES)
        self.check(set(rec["input_bindings"]) == {"predictive_rgb", "initial_supervision"}, "input phase set")
        for phase, specpath, folder in (("predictive_rgb", self.args.rgb_spec, rgbroot),
                                        ("initial_supervision", self.args.initial_spec, initialroot)):
            binding = rec["input_bindings"][phase]
            spec = self.load(specpath)
            self.check(binding["receipt_sha256"] == sha(folder/"receipt.json")
                       and binding["spec_sha256"] == sha(specpath)
                       and binding["outer_sha256"] == sha(Path(spec["operation_output"])/"receipt.json"), "input linkage")
            logs = Path(spec["operation_output"])
            allowed = {str(folder/"receipt.json"), str(Path(specpath).resolve()),
                       *[str(logs/n) for n in ("receipt.json","stdout.log","stderr.log")],
                       *[str(self.root/n) for n in ACQ_SOURCES]}
            self.check(set(binding["file_sha256"]) == allowed, "exact acquisition snapshot paths")
            for path, digest in binding["file_sha256"].items():
                self.file(path, digest)
        native = Path(argv["--native-root"])
        self.check(native.resolve() == self.root.parent/"0029-runtime-02", "fixed qualified runtime location")
        inst = self.load(native/"installation/install-receipt.json")
        qflags = {"--repo-root","--expected-commit","--installation","--installation-supervision",
                  "--installation-spec","--output-dir"}
        qual, qargv, qout, _ = self.outer(native/"qualification-spec.json", NATIVE_COMMIT, OWN+"qualify.py", qflags)
        self.check(qout == native/"qualification" and qargv["--installation"] == str(native/"installation")
                   and qargv["--installation-supervision"] == str(native/"installation-supervision/receipt.json")
                   and qargv["--installation-spec"] == str(native/"installation-spec.json"), "native operation input identity")
        self.check(inst["status"] == qual["status"] == "completed" and qual["eligible"] is True
                   and inst["expected_commit"] == qual["expected_commit"] == NATIVE_COMMIT
                   and qual["offline_attempts"] == [] and qual["scientific_input_access"] is False,
                   "qualified runtime receipts")
        self.sources(inst["source_sha256"], NATIVE_COMMIT)
        self.sources(qual["source_sha256"], NATIVE_COMMIT)
        self.check(rec["native_binding"]["receipt_sha256"] == sha(native/"qualification/receipt.json")
                   and rec["native_binding"]["outer_sha256"] == sha(native/"qualification-supervision/receipt.json")
                   and rec["native_binding"]["spec_sha256"] == sha(native/"qualification-spec.json"), "native linkage")
        allowed = {str(self.root/n) for n in qual["source_sha256"]}
        allowed |= {str(native/n) for n in ("qualification/receipt.json", "qualification-spec.json",
            "qualification-supervision/receipt.json", "qualification-supervision/stdout.log",
            "qualification-supervision/stderr.log", "installation/install-receipt.json",
            "installation-spec.json", "installation-supervision/receipt.json",
            "installation-supervision/stdout.log", "installation-supervision/stderr.log")}
        self.check(set(rec["native_binding"]["file_sha256"]) == allowed, "exact native snapshot paths")
        for path, digest in rec["native_binding"]["file_sha256"].items():
            self.file(path, digest)
        for rel, item in inst["installed_inventory"].items():
            path = self.child(native/"installation", rel)
            if item["type"] == "file":
                self.file(path, item["sha256"], item["bytes"])
            else:
                self.check(path.is_symlink() and os.readlink(path) == item["target"]
                           and sha(path.resolve()) == item["resolved_sha256"], "installed interpreter link")
        folder = out/"evidence"
        saved = self.load(folder/"evidence-record.json")
        self.check(saved == rec["evidence"] == rec["frontend_evidence"]
                   and saved["status"] == "completed" and saved["complete"] is True
                   and saved["source_sha256"] == rec["source_sha256"]
                   and saved["sam_calls"] == 0 and saved["noninitial_annotation_access"] is False
                   and saved["sam_sequence_admitted"] is False, "identical completed evidence records")
        self.check(saved["calls"] == dict(superpoint_intent=54, superpoint_completed=54,
                                          lightglue_intent=48, lightglue_completed=48), "fixed call counts")
        self.check(saved["execution"] == dict(torch_intraop=4, torch_interop=1, opencv_threads=1, autocast=False)
                   and saved["settings"] == dict(device="cpu", dtype="float32", resize=512,
                       resize_side="long", maximum_keypoints=2048,
                       coordinate_convention="original-image zero-based pixel centers",
                       superpoint={"max_num_keypoints":2048},
                       lightglue=dict(features="superpoint", flash=False, mp=False,
                                      depth_confidence=-1, width_confidence=-1)), "fixed frontend settings")
        expected_weights = {
            "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth":
                sha(native/"installation/weights/superpoint_v1.pth"),
            "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth":
                sha(native/"installation/weights/superpoint_lightglue.pth")}
        self.check(saved["checkpoint_sha256"] == expected_weights, "qualified frontend weights")
        # All acquisition, source, outer and file barriers precede initial-mask/array decoding.
        runtime = native/"installation/venv/lib/python3.12/site-packages"
        sys.path.insert(0, str(runtime))
        import numpy as np
        import PIL
        from PIL import Image
        self.check(all(Path(m.__file__).resolve().is_relative_to(runtime.resolve()) for m in (np, PIL)),
                   "audit numerical/image-library origins")
        self.report["libraries"] = dict(numpy=np.__version__, pillow=PIL.__version__)
        images = {(r["sequence"], r["frame"]): r for r in saved["images"]}
        queries = {(r["sequence"], r["frame"]): r for r in saved["queries"]}
        expected_images = [(s, t) for s, n in COHORT for t in [0, *schedule(n)]]
        expected_queries = [(s, t) for s, n in COHORT for t in schedule(n)]
        self.check(len(saved["images"]) == len(images) == 54 and set(images) == set(expected_images)
                   and len(saved["queries"]) == len(queries) == 48 and set(queries) == set(expected_queries)
                   and [r["sequence"] for r in saved["initial_masks"]] == [s for s, _ in COHORT], "all54/48/6 identities")
        features, files = {}, {"evidence-record.json"}
        for identity in expected_images:
            row = images[identity]
            path, media = rgbs[identity]
            self.check(row["status"] == "completed" and row["superpoint_call_intent"] is True
                       and Path(row["input"]["path"]) == path and row["input"]["sha256"] == media["sha256"]
                       and row["input"]["bytes"] == media["bytes"], "feature input")
            a = self.arrays(folder, row["output"], np); features[identity] = a
            files.add(row["output"]["filename"])
            n, shape = row["keypoint_count"], row["shape"]
            with Image.open(path) as header:
                self.check(header.format == "JPEG" and list(header.size) == [shape[1],shape[0]],
                           "RGB header dimensions; no pixel decode")
            self.check(set(a) == {"keypoints", "keypoint_scores", "descriptors", "image_size"}
                       and 0 <= n <= 2048 and a["keypoints"].shape == (1,n,2)
                       and a["keypoint_scores"].shape == (1,n) and a["descriptors"].shape == (1,n,256)
                       and len(shape) == 3 and shape[2] == 3
                       and np.array_equal(a["image_size"], [[shape[1], shape[0]]]), "feature dimensions")
        counts = {}
        for (sequence, n), mask_row in zip(COHORT, saved["initial_masks"]):
            path, media = initials[sequence, 0]
            self.check(mask_row["status"] == "completed" and Path(mask_row["input"]["path"]) == path
                       and mask_row["input"]["sha256"] == media["sha256"]
                       and mask_row["input"]["bytes"] == media["bytes"], "initial input identity")
            with Image.open(path) as image:
                self.check(image.format == "PNG" and image.mode in ("P","L","I","I;16"), "initial mask format")
                labels = np.asarray(image).copy()
            self.check(labels.ndim == 2 and labels.dtype.kind in "biu" and np.all(labels >= 0)
                       and list(labels.shape) == mask_row["shape"] == images[sequence,0]["shape"][:2]
                       and hashlib.sha256(labels.tobytes()).hexdigest() == mask_row["decoded_ids_sha256"], "decoded initial IDs")
            targets = {int(v) for v in np.unique(labels) if v > 0 and v != 255}
            self.check(bool(targets) and min(targets) == mask_row["target_id"], "smallest initial target")
            counts[sequence] = 0
            for frame in schedule(n):
                row = queries[sequence,frame]
                a = self.arrays(folder, row["matches"], np)
                files.add(row["matches"]["filename"])
                ids, scores = a["match_ids"], a["match_scores"]
                i0, it = features[sequence,0], features[sequence,frame]
                sizes = [i0["keypoints"].shape[1], it["keypoints"].shape[1]]
                compact = a["matches"][0] if a["matches"].ndim == 3 else a["matches"]
                conf = a["scores"][0] if a["scores"].ndim == 2 else a["scores"]
                self.check(row["status"] == "completed" and row["lightglue_call_intent"] is True
                           and ids.dtype.kind in "iu" and ids.ndim == 2 and ids.shape[1] == 2
                           and scores.shape == (len(ids),) and row["match_count"] == len(ids)
                           and np.array_equal(ids,compact) and np.array_equal(scores,conf), "canonical matches")
                for side in (0,1):
                    dense = a["matches%d" % side]
                    self.check(dense.shape == (1,sizes[side]) and np.all(ids[:,side] >= 0)
                               and np.all(ids[:,side] < sizes[side]) and len(np.unique(ids[:,side])) == len(ids)
                               and np.array_equal(dense[0,ids[:,side]], ids[:,1-side])
                               and np.array_equal(np.flatnonzero(dense[0] >= 0), np.sort(ids[:,side])),
                               "complete native match mutuality")
                item = row["witnesses"]
                witness = json.loads(self.file(self.child(folder,item["filename"]),item["sha256"],item["bytes"]).read_text())
                files.add(item["filename"])
                expected = independent_witnesses(labels, i0["keypoints"][0], it["keypoints"][0],
                    ids, scores, images[sequence,frame]["shape"][:2], sequence, frame)
                same = witness == expected
                self.check(same, "independent witnesses: %s/%d" % (sequence,frame), fatal=False)
                self.check(row["target_id"] == expected["target_id"] and row["actionable"] is expected["actionable"]
                           and row["witness_counts"] == expected["counts"], "witness row accounting", fatal=False)
                counts[sequence] += expected["actionable"]
                self.report["rows"].append(dict(sequence=sequence, frame=frame, exact_witness_match=same,
                    match_count=len(ids), positive=len(expected["positive"]), negative=len(expected["negative"]),
                    actionable=expected["actionable"], counts=expected["counts"]))
        videos = sum(v > 0 for v in counts.values())
        coverage = dict(queries=48, videos=6, actionable_queries=sum(counts.values()),
                        actionable_queries_by_video=counts, videos_with_evidence=videos, **{"pass": videos >= 4})
        self.check(saved["coverage"] == coverage, "independent full48 coverage", fatal=False)
        self.check({str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == files, "complete evidence files")
        self.check(all(x["status"] == "completed" and math.isfinite(x["seconds"]) and x["seconds"] >= 0
                       for x in saved["timings"]), "all stage timings")
        self.report["summary"] = dict(input_files=392, feature_files=54, match_files=48, witness_files=48,
                                     initial_masks_decoded=6, query_masks_decoded=0, coverage=coverage)
        for path, digest in self.bindings.items():
            self.check(sha(path) == digest, "input unchanged: " + path)
        self.report["input_sha256"] = self.bindings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "rgb-spec", "initial-spec", "evidence-spec", "output-dir"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh audit output required")
    output.mkdir(parents=True)
    began = time.perf_counter()
    audit = Audit(args)
    try:
        audit.run()
    except BaseException as error:
        audit.report["failures"].append(type(error).__name__ + ": " + str(error))
    audit.report.update(passed=not audit.report["failures"], wall_seconds=time.perf_counter()-began,
                        source_sha256=sha(__file__))
    with (output/"audit.json").open("x") as stream:
        json.dump(audit.report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k:audit.report[k] for k in ("passed","checks","failures")}), flush=True)
    raise SystemExit(0 if audit.report["passed"] else 1)
