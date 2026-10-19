"""Read-only fixed input adapter for the complete, admitted 0023 frontend.

No image decoding, model/native imports, pose files, labels or inference.
Caller first admits outer frontend supervision and supplies its sealed receipt
and provenance bindings. Every51+120 output/hash is checked before np.load.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("s00513", "s00475", "s00478")
IMAGE_READS = {"xy", "K_original", "K_resized", "image_size_original", "image_size_resized",
               "depth_sample_raw", "depth_sample_normalized", "depth_sample_valid", "depth_positive_valid_median"}
IMAGE_ALLOWED = IMAGE_READS | {"depth", "mask", "points", "intrinsics", "normal", "keypoints",
                                "descriptors", "keypoint_scores", "image_size"}
EDGE_READS = {"match_ids", "match_scores", "xy0", "xy1", "depth0_raw", "depth1_raw",
              "depth0_normalized", "depth1_normalized", "depth_valid0", "depth_valid1"}
EDGE_ALLOWED = EDGE_READS | {"matches", "scores", "matches0", "matches1", "matching_scores0",
                            "matching_scores1", "prune0", "prune1", "stop"}
BINDING_FIELDS = ("plan_sha256", "expected_commit", "frontend_freeze", "input_receipt_sha256",
                  "source_sha256", "frontend_bindings", "input_acquisition_bindings", "input_sha256")
SETTINGS = dict(moge=dict(num_tokens=1200, use_fp16=False, apply_mask=False, force_projection=True),
                superpoint=dict(max_num_keypoints=2048),
                lightglue=dict(features="superpoint", flash=False, mp=False,
                               depth_confidence=-1, width_confidence=-1),
                max_edge=512, extract_resize=None, device="cpu", dtype="float32", threads=4)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16*1024**2,
            "bounded regular JSON required")
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def safe_child(root, name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name
            and not PurePosixPath(name).is_absolute() and all(x not in ("", ".", "..") for x in name.split("/")),
            "unsafe input path")
    root = Path(root)
    require(not root.is_symlink(), "symlink root")
    out = root/name
    p = out
    while p != root.parent:
        require(not p.is_symlink(), "symlink input path")
        p = p.parent
    require(out.resolve().is_relative_to(root.resolve()) and out.is_file(), "missing input or path escape")
    return out


def plan_layout(plan):
    """Rebuild the declared chronological roles and exact32+8 graph."""
    require(plan.get("schema") == "evr0023.proposed-plan.v1", "plan schema")
    require(isinstance(plan.get("scenes"), list) and tuple(s["scene"] for s in plan["scenes"]) == SCENES,
            "fixed three-place order")
    layouts = []; all_images = []; all_edges = []; all_members = []
    for scene in plan["scenes"]:
        prefix = "val/"+scene["scene"]+"/"
        require(scene["anchor"] == prefix+"seq0/frame_00000.jpg", "anchor identity")
        rows = scene["selected_seq1_chronological"]
        require(len(rows) == 16, "all sixteen chronological images required")
        mapping = {scene["anchor"]: "A0"}; names = [scene["anchor"]]; numbers = []
        for index, row in enumerate(rows, 1):
            name = row["filename"]
            match = re.fullmatch(re.escape(prefix)+r"seq1/frame_([0-9]{5})\.jpg", name)
            require(match is not None and row["selected_index_one_based"] == index
                    and row["frame_number"] == int(match[1])
                    and row["role"] == ("B_query" if index % 2 else "C_reference"), "chronological role identity")
            require(type(row["chronological_rank_zero_based"]) is int and row["chronological_rank_zero_based"] >= 0,
                    "chronological rank")
            numbers.append(row["frame_number"]); names.append(name)
            mapping[name] = ("B" if index % 2 else "C")+str((index+1)//2)
        require(all(a < b for a, b in zip(numbers, numbers[1:])) and len(mapping) == 17,
                "strict chronological unique image order")
        require(all(a["chronological_rank_zero_based"] < b["chronological_rank_zero_based"]
                    for a, b in zip(rows, rows[1:])), "strict source member ranks")
        b, c = rows[::2], rows[1::2]
        refs = [dict(source=scene["anchor"], target=r["filename"], role="A_to_C") for r in c]
        for query in b:
            near = sorted(c, key=lambda r:(abs(r["frame_number"]-query["frame_number"]), r["frame_number"], r["filename"]))[:3]
            refs += [dict(source=query["filename"], target=r["filename"], role="B_to_C",
                          frame_number_distance=abs(r["frame_number"]-query["frame_number"])) for r in near]
        queries = [dict(source=scene["anchor"], target=r["filename"], role="A_to_B") for r in b]
        require(scene["reference_edges"] == refs and scene["query_edges"] == queries, "fixed reference/query graph changed")
        require(scene["query_images"] == [r["filename"] for r in b]
                and scene["reference_images"] == [r["filename"] for r in c], "image role lists")
        require([r["filename"] for r in scene["image_members"]] == names
                and scene["intrinsics_member"]["filename"] == prefix+"intrinsics.txt", "frontend members")
        edges = [dict(e, scene=scene["scene"]) for e in refs+queries]
        layouts.append(dict(scene=scene["scene"], mapping=mapping, image_paths=names,
                            reference_edges=edges[:32], query_edges=edges[32:]))
        all_images += names; all_edges += edges; all_members += names+[prefix+"intrinsics.txt"]
    require(plan["frontend_acquisition"]["image_names"] == all_images
            and plan["frontend_acquisition"]["member_names"] == all_members
            and plan["frontend_acquisition"]["intrinsics_names"] == ["val/"+s+"/intrinsics.txt" for s in SCENES],
            "complete frontend member list")
    return layouts, all_images, all_edges, all_members


def verify_cache(plan_path, cache_dir, bindings):
    """Whole-cohort byte/metadata barrier; this function never opens an NPZ."""
    cache_dir = Path(cache_dir)
    require(set(BINDING_FIELDS) | {"frontend_receipt_sha256"} <= set(bindings), "admitted bindings missing")
    require(sha(plan_path) == bindings["plan_sha256"], "plan binding")
    plan = read_json(plan_path); layouts, names, edges, members = plan_layout(plan)
    receipt_path = safe_child(cache_dir, "receipt.json")
    require(sha(receipt_path) == bindings["frontend_receipt_sha256"], "frontend receipt binding")
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == "evr0023.frontend.v1" and receipt.get("status") == "completed"
            and receipt.get("eligible") is True and receipt.get("stage") == "completed"
            and receipt.get("offline_attempts") == [], "frontend outcome not admitted")
    require(all(receipt.get(k) == bindings[k] for k in BINDING_FIELDS), "frontend provenance changed")
    require(receipt.get("settings") == SETTINGS, "frontend settings changed")
    require(all(re.fullmatch(r"[0-9a-f]{40}", receipt[k]) for k in ("expected_commit", "frontend_freeze")), "commit format")
    require(set(receipt["input_sha256"]) == set(members)
            and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in receipt["input_sha256"].values()), "all54 input bindings")
    require(isinstance(receipt["frontend_bindings"], dict) and receipt["frontend_bindings"], "model/installation bindings")
    acquisition = receipt["input_acquisition_bindings"]
    require(isinstance(acquisition, dict) and set(acquisition) == {
        "input_operation_sha256", "input_supervision_sha256", "input_supervision_log_sha256"},
        "input acquisition binding schema")
    logs = acquisition["input_supervision_log_sha256"]
    require(isinstance(logs, dict) and set(logs) == {"stdout", "stderr"}
            and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)
                    for h in [acquisition["input_operation_sha256"], acquisition["input_supervision_sha256"], *logs.values()]),
            "input acquisition binding hashes")
    source = receipt["source_sha256"]
    required = {"experiments/0023-mapfree/"+f for f in (
        "frontend.py", "range_inputs.py", "propose_plan.py", "acquire_operation.py", "supervise.py")}
    require(isinstance(source, dict) and required <= set(source) and len(source) == 6
            and all(k in required or (k.startswith("protocols/") and k.endswith(".md")) for k in source), "frozen frontend source set")
    for name, digest in source.items():
        require(sha(safe_child(ROOT, name)) == digest, "frontend source bytes changed")
    require(len(receipt.get("images", [])) == 51 and len(receipt.get("edges", [])) == 120, "complete51/120 output population")
    inventory = {"receipt.json"}; file_hashes = {"receipt.json": sha(receipt_path)}; total = 0
    for kind, rows, expected in (("image", receipt["images"], names), ("edge", receipt["edges"], edges)):
        for index, (row, identity) in enumerate(zip(rows, expected)):
            require(row.get(kind+"_index") == index and row.get("status") == "completed"
                    and row.get("stage") == "completed", "incomplete or reordered frontend outcome")
            if kind == "image":
                require(row.get("filename") == identity, "image outcome identity")
                require(row["input"]["original_encoded_sha256"] == receipt["input_sha256"][identity], "encoded image input binding")
            else:
                require(all(row.get(k) == v for k, v in identity.items()), "edge outcome identity")
                require(row.get("geometry_observations_removed_for_depth") == 0, "frontend depth filtering forbidden")
            stem = "%s-%03d" % (kind, index); json_name = stem+".json"; npz_name = stem+".npz"
            row_path = safe_child(cache_dir, json_name)
            require(read_json(row_path) == row, "per-output receipt differs from cohort receipt")
            meta = row["cache"]
            require(meta["filename"] == npz_name and type(meta["bytes"]) is int and 0 < meta["bytes"] <= 64*1024**2,
                    "cache name/size contract")
            path = safe_child(cache_dir, npz_name)
            require(path.stat().st_size == meta["bytes"] and sha(path) == meta["sha256"], "cache bytes changed")
            needed, allowed = (IMAGE_READS, IMAGE_ALLOWED) if kind == "image" else (EDGE_READS, EDGE_ALLOWED)
            require(needed <= set(meta["arrays"]) <= allowed, "unexpected/missing cache array; GT prohibited")
            for entry in meta["arrays"].values():
                require(isinstance(entry["shape"], list) and all(type(n) is int and n >= 0 for n in entry["shape"])
                        and re.fullmatch(r"[0-9a-f]{64}", entry["raw_c_order_sha256"]), "array metadata")
            inventory.update((json_name, npz_name)); file_hashes[json_name] = sha(row_path); file_hashes[npz_name] = meta["sha256"]
            total += meta["bytes"]
    require(total <= 2*1024**3, "cache cohort byte cap")
    require({p.name for p in cache_dir.iterdir()} == inventory, "unlisted cache artifacts")
    return dict(plan=plan, layouts=layouts, receipt=receipt, file_sha256=file_hashes, cache_bytes=total)


def read_arrays(root, row, fields):
    meta = row["cache"]; path = safe_child(root, meta["filename"])
    require(sha(path) == meta["sha256"], "cache changed after whole-cohort barrier")
    result = {}
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)) and set(saved.files) == set(meta["arrays"]), "NPZ directory/receipt mismatch")
        for name in sorted(fields):
            a = saved[name]; info = meta["arrays"][name]
            require(a.dtype.kind != "O" and list(a.shape) == info["shape"] and str(a.dtype) == info["dtype"]
                    and hashlib.sha256(a.tobytes(order="C")).hexdigest() == info["raw_c_order_sha256"], "canonical array changed")
            result[name] = a.copy()
    require(sha(path) == meta["sha256"], "cache changed during load")
    return result


def image_input(row, arrays):
    xy = arrays["xy"]; size = arrays["image_size_resized"]; original_size = arrays["image_size_original"]
    require(size.shape == original_size.shape == (2,) and size.dtype.kind in "iu" and original_size.dtype.kind in "iu"
            and (size >= 1).all() and (size <= 512).all() and (original_size >= 1).all(), "image dimensions")
    w, h = map(int, size); ow, oh = map(int, original_size)
    require(xy.dtype == np.float32 and xy.ndim == 2 and xy.shape[1] == 2 and len(xy) <= 2048
            and np.isfinite(xy).all() and (xy >= 0).all() and (xy[:, 0] < w).all() and (xy[:, 1] < h).all(), "cached pixels")
    k, old = arrays["K_resized"], arrays["K_original"]
    require(k.shape == old.shape == (3, 3) and k.dtype == old.dtype == np.float64
            and np.isfinite(k).all() and np.isfinite(old).all()
            and min(old[0, 0], old[1, 1]) > 0 and old[0, 1] == old[1, 0] == 0
            and np.array_equal(old[2], [0., 0., 1.]), "cached authored calibration")
    scale = min(1., 512/max(ow, oh))
    require((w, h) == (max(1, math.floor(ow*scale+.5)), max(1, math.floor(oh*scale+.5))), "fixed resize dimensions")
    # Match the frozen producer's floating operation order exactly. Multiplying
    # by an integer dimension and then dividing can differ by one ULP.
    sx, sy = w/ow, h/oh
    expected = old.copy(); expected[0, 0] *= sx; expected[1, 1] *= sy
    expected[0, 2] = (old[0, 2]+.5)*sx-.5; expected[1, 2] = (old[1, 2]+.5)*sy-.5
    require(np.array_equal(k, expected) and row["input"]["authored_K_original"] == old.tolist()
            and row["input"]["authored_K_resized"] == k.tolist()
            and row["input"]["resized_shape_hwc"] == [h, w, 3]
            and row["input"]["original_shape_hwc"] == [oh, ow, 3], "calibration/coordinate metadata mismatch")
    d, raw, valid = arrays["depth_sample_normalized"], arrays["depth_sample_raw"], arrays["depth_sample_valid"]
    median = arrays["depth_positive_valid_median"]
    require(d.shape == raw.shape == valid.shape == (len(xy),) and d.dtype == raw.dtype == np.float64
            and valid.dtype == bool and median.shape == () and median.dtype == np.float64, "depth shape/type")
    require(np.isfinite(d[valid]).all() and np.isfinite(raw[valid]).all() and (d[valid] > 0).all()
            and (raw[valid] > 0).all(), "valid sampled depths")
    value = float(median)
    if valid.any():
        require(math.isfinite(value) and value > 0 and np.array_equal(d[valid], raw[valid]/value), "whole-image depth normalization")
    require(row["keypoint_count"] == len(xy) and row["sampled_depth_valid_count"] == int(valid.sum())
            and row["positive_valid_depth_median"] == (value if math.isfinite(value) else None), "image summary disagrees")
    return dict(xy=xy, K=k, width=w, height=h, depth=d, depth_raw=raw, depth_valid=valid,
                depth_positive_valid_median=value, original_path=row["filename"],
                image_index=row["image_index"], cache_sha256=row["cache"]["sha256"])


def select_matches(match_ids, scores, source_xy, width, height, cap):
    """8x8 source cells; score/index within cells; row-major round robin."""
    pairs = np.asarray(match_ids); scores = np.asarray(scores); xy = np.asarray(source_xy)
    require(cap in (128, 256) and pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu"
            and scores.shape == (len(pairs),) and np.isfinite(scores).all(), "selection input contract")
    require(type(width) is int and type(height) is int and min(width, height) > 0
            and xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all()
            and (pairs >= 0).all() and (pairs[:, 0] < len(xy)).all(), "source selection coordinates")
    positions = xy[pairs[:, 0]]
    require((positions >= 0).all() and (positions[:, 0] < width).all() and (positions[:, 1] < height).all(), "source point outside image")
    cells = np.floor(positions/np.array([width, height])*8).astype(np.int64)
    number = cells[:, 1]*8+cells[:, 0]
    buckets = [[] for _ in range(64)]
    for index, cell in enumerate(number):
        buckets[int(cell)].append(index)
    for bucket in buckets:
        bucket.sort(key=lambda i:(-float(scores[i]), i))
    selected = []; level = 0
    while len(selected) < min(cap, len(pairs)):
        for bucket in buckets:
            if level < len(bucket):
                selected.append(bucket[level])
                if len(selected) == min(cap, len(pairs)):
                    break
        level += 1
    return np.asarray(selected, dtype=np.int64)


def load_inputs(plan_path, cache_dir, bindings):
    started = time.perf_counter(); verified = verify_cache(plan_path, cache_dir, bindings)
    receipt = verified["receipt"]; images = {}
    for row in receipt["images"]:
        images[row["filename"]] = image_input(row, read_arrays(cache_dir, row, IMAGE_READS))
    scene_results = []; selections = []; edge_index = 0
    for layout in verified["layouts"]:
        refs = []; queries = []; mapping = layout["mapping"]
        for expected in layout["reference_edges"]+layout["query_edges"]:
            row = receipt["edges"][edge_index]; array = read_arrays(cache_dir, row, EDGE_READS)
            a, b = images[row["source"]], images[row["target"]]
            pairs, scores = array["match_ids"], array["match_scores"]
            require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in "iu"
                    and scores.shape == (len(pairs),) and np.isfinite(scores).all()
                    and (pairs >= 0).all() and (pairs[:, 0] < len(a["xy"])).all() and (pairs[:, 1] < len(b["xy"])).all(), "canonical matching indices")
            require(len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1])), "nonmutual canonical matches")
            for end, image in enumerate((a, b)):
                ids = pairs[:, end]
                for name, key in (("xy%d"%end, "xy"), ("depth%d_raw"%end, "depth_raw"),
                                  ("depth%d_normalized"%end, "depth"), ("depth_valid%d"%end, "depth_valid")):
                    require(np.array_equal(array[name], image[key][ids], equal_nan=True), "edge/image canonical copy mismatch")
            require(row["match_count"] == len(pairs)
                    and row["both_depth_valid_count"] == int((array["depth_valid0"] & array["depth_valid1"]).sum()), "edge summary disagreement")
            cap = 256 if row["role"] == "A_to_B" else 128
            selected = select_matches(pairs, scores, a["xy"], a["width"], a["height"], cap)
            edge = dict(i=mapping[row["source"]], j=mapping[row["target"]],
                        ids_i=pairs[selected, 0].copy(), ids_j=pairs[selected, 1].copy(),
                        scores=scores[selected].copy(), original_match_indices=selected.copy(),
                        source_path=row["source"], target_path=row["target"], edge_index=edge_index,
                        role=row["role"], input_count=len(pairs), cap=cap, status="completed",
                        cache_sha256=row["cache"]["sha256"])
            (queries if cap == 256 else refs).append(edge)
            selections.append(dict(scene=layout["scene"], edge_index=edge_index, i=edge["i"], j=edge["j"],
                                   role=edge["role"], input_count=len(pairs), selected_count=len(selected), cap=cap,
                                   original_match_indices=selected.tolist(), status="completed"))
            edge_index += 1
        scene_results.append(dict(scene=layout["scene"], images={mapping[p]:images[p] for p in layout["image_paths"]},
                                   reference_edges=refs, query_edges=queries))
    # Recheck every metadata/cache file and source after canonical input assembly.
    post = verify_cache(plan_path, cache_dir, bindings)
    require(post["file_sha256"] == verified["file_sha256"], "frontend changed during adapter read")
    return dict(scenes=scene_results, receipt=receipt,
                selection_receipt=dict(schema="evr0023.fixed-selection.v1", status="completed",
                    source_sha256=sha(__file__), plan_sha256=bindings["plan_sha256"],
                    frontend_receipt_sha256=bindings["frontend_receipt_sha256"], file_sha256=verified["file_sha256"],
                    image_count=51, edge_count=120, rules=dict(grid=[8, 8], cell="floor(8*x/width),floor(8*y/height)",
                    within_cell="descending match score, then original compact-match index", across_cells="row-major round robin",
                    reference_cap=128, query_cap=256, depth_filtering=False), edges=selections,
                    seconds=time.perf_counter()-started))
