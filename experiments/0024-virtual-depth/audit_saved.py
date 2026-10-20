"""Independent saved-output audit, prepared before reading the 0024 probe.

Only stdlib and NumPy. No production imports, image decoding, Torch, inference,
pose labels, optimizer rerun, or parameter selection. Execute only after the
parent explicitly admits the completed probe and outer-supervision outputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np


INPUT_COMMIT = "acf502b831ad9e3485ab56549217c8a603d56bf7"
MODEL_COMMIT = "fa454585e67a8f0395ad5f67a03d5a28cc33a66a"
PLAN_SHA = "cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5"
ORIGINAL_OUTER_SHA = "4439b3d24550e425d976a18f114aeb5734c526e3ec08bba73dfb4e206c3bba5f"
REVIEWED = {
    "experiments/0024-virtual-depth/run.py": "56db02536178762d540bef75feceef2eae8ea6a5fabac1f8d3bde2c1bbe53f1f",
    "experiments/0024-virtual-depth/math_core.py": "9e6f18fc1781c39e087ee8f40edfac79963fac5068ff4f99ec9b96b0897ae737",
    "experiments/0024-virtual-depth/supervise.py": "3e7124bfccf8642c40d212b62c1bb638fdd094e060e291fa0a8813f2f97f3bcd",
    "protocols/0024-virtual-rotation-falsifier.md": "4ae8f943112afa705a072e194e94a424252c282dd6beff54606584310007da3b",
}
DIRECTIONS = ("yaw_to_pitch", "pitch_to_yaw")
VIEWS = ("identity", "yaw_positive", "yaw_negative", "pitch_positive", "pitch_negative")
# Fixed BEFORE outputs: alternative projection/interpolation order and QR/FWL
# offset elimination differ from the production SVD reconstruction. Allowed
# offset designs have condition <=1e5. This conservative margin exceeds typical
# float64 forward error; every discrete decision and coverage count is exact.
COEFF_ATOL, COEFF_RTOL = 2e-8, 2e-7
ERROR_ATOL, ERROR_RTOL = 1e-9, 1e-7


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def atomic_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    pending.replace(path)


def git_bytes(repo, commit, relative):
    require(re.fullmatch("[0-9a-f]{40}", commit) is not None, "invalid commit")
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe source path")
    return subprocess.check_output(["git", "show", commit+":"+relative], cwd=repo)


def leaf(directory, name):
    require(Path(name).name == name and name not in ("", ".", ".."), "unsafe output filename")
    path = directory / name
    require(path.is_file() and not path.is_symlink(), "missing or symlink output: "+name)
    return path


def public_path(value, repo):
    value = str(value).replace("$REPOSITORY", str(repo)).replace("$PRIVATE", str(repo.parent))
    path = Path(value)
    return (path if path.is_absolute() else repo/path).resolve()


def command_flags(command, script):
    require(command[:2] == ["$PYTHON", "$REPOSITORY/"+script], "command executable/script")
    require(len(command) % 2 == 0, "command pairs")
    flags = {}
    for key, value in zip(command[2::2], command[3::2]):
        require(isinstance(key, str) and key.startswith("--") and key not in flags, "command duplicate/invalid flag")
        require(isinstance(value, str), "command value")
        flags[key] = value
    return flags


def bilinear(depth, mask, x, y):
    """Independent separable bilinear interpolation with exact support accounting."""
    height, width = depth.shape
    good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x < width-1) & (y < height-1)
    result = np.zeros(x.shape, dtype=np.float64)
    for k in np.flatnonzero(good):
        ix, iy = math.floor(float(x[k])), math.floor(float(y[k]))
        block = depth[iy:iy+2, ix:ix+2]
        if not (mask[iy:iy+2, ix:ix+2].all() and np.isfinite(block).all() and (block > 0).all()):
            good[k] = False
            continue
        dx, dy = float(x[k]-ix), float(y[k]-iy)
        top = (1-dx)*float(block[0, 0])+dx*float(block[0, 1])
        bottom = (1-dx)*float(block[1, 0])+dx*float(block[1, 1])
        result[k] = (1-dy)*top+dy*bottom
    require(np.isfinite(result).all(), "independent interpolation arithmetic")
    return result, good


def spectrum(design):
    norms = np.sqrt(np.sum(design*design, axis=0))
    if (norms == 0).any():
        return norms, np.array([])
    return norms, np.linalg.svd(design/norms, compute_uv=False)


def shared_offset(base, depth, factor):
    n = len(base)
    blocks = [np.array([depth[j]/factor[j], 1/factor[j]]).T for j in range(2)]
    design = np.zeros((2*n, 5)); design[:, 0] = -1
    design[:n, 1:3], design[n:, 3:5] = blocks
    _, singular = spectrum(design)
    ratio = float(singular[-1]/singular[0]) if len(singular) else 0.
    result = dict(usable=False, reason="rank_deficient", coefficients=None,
                  singular_values=singular.tolist(), singular_ratio=ratio, training_rows=2*n)
    if len(singular) != 5 or singular[-1] <= 0 or ratio < 1e-5:
        return result
    # Frisch-Waugh-Lovell: eliminate both per-view two-column nuisance blocks
    # with QR, then solve the ONE shared coefficient. No production 5D solver.
    numerator, denominator = 0., 0.
    for block in blocks:
        Q, _ = np.linalg.qr(block, mode="reduced")
        projection_y = np.sum(Q*base[:, None], axis=0)
        projection_one = np.sum(Q, axis=0)
        residual_y = base-np.sum(Q*projection_y, axis=1)
        residual_one = 1-np.sum(Q*projection_one, axis=1)
        numerator += float(np.sum(residual_one*residual_y))
        denominator += float(np.sum(residual_one*residual_one))
    require(denominator > 0, "independent offset excitation vanished")
    offset = -numerator/denominator
    coefficients = [np.linalg.lstsq(block, base+offset, rcond=None)[0] for block in blocks]
    scales = [float(c[0]) for c in coefficients]
    shifts = [float(c[1]) for c in coefficients]
    require(all(math.isfinite(v) for v in [offset, *scales, *shifts]), "independent nonfinite offset")
    result["coefficients"] = dict(t0=offset, scales=scales, shifts=shifts)
    result.update(usable=min(scales) > 0, reason=None if min(scales) > 0 else "nonpositive_scale")
    return result


def nuisance(base, depth, factor, offset):
    design = np.array([depth/factor, 1/factor]).T
    norms, singular = spectrum(design)
    full = len(singular) == 2 and singular[-1] > max(design.shape)*np.finfo(float).eps*singular[0]
    boundary = None if full else "rank_deficient"
    coefficients = None
    if full:
        coefficients = np.linalg.lstsq(design/norms, base+offset, rcond=0)[0]/norms
        if coefficients[0] < 0:
            boundary = "negative_scale"
    if boundary is not None:
        inverse_h = 1/factor
        coefficients = [0., float(np.sum(inverse_h*(base+offset))/np.sum(inverse_h*inverse_h))]
    return dict(scale=float(coefficients[0]), shift=float(coefficients[1]), boundary=boundary,
                singular_values=singular.tolist(), training_rows=len(base))


def unavailable(result, reason):
    result.update(usable_offset=False, unavailable_reason=reason,
        directions={name: dict(transferred_offset=None, candidate_error=1., zero_offset_error=1.,
                                fallback_to_baseline=True, views=[]) for name in DIRECTIONS})
    return result


def independent_analysis(depths, masks, camera, rotations):
    height, width = depths.shape[1:]
    valid_pixels = masks & np.isfinite(depths) & (depths > 0)
    counts = [int(x.sum()) for x in valid_pixels]
    medians = [float(np.median(d[v])) if v.any() else None for d, v in zip(depths, valid_pixels)]
    result = dict(medians=medians, positive_valid_pixel_counts=counts, size=[width, height],
                  support=dict(total=4096, common=0, train=0, evaluation=0), offsets={},
                  offset_disagreement=None, directions={}, usable_offset=False, unavailable_reason=None)
    if None in medians:
        return unavailable(result, "positive_valid_median_unavailable")
    ij = [(i, j) for j in range(64) for i in range(64)]
    x = np.array([(i+.5)*width/64-.5 for i, _ in ij])
    y = np.array([(j+.5)*height/64-.5 for _, j in ij])
    training = np.array([(i+j) % 2 == 0 for i, j in ij])
    rx, ry = (x-camera[0, 2])/camera[0, 0], (y-camera[1, 2])/camera[1, 1]
    observations, validity, hs = [], [], []
    for j in range(5):
        if j == 0:
            px, py, geometric = x, y, np.ones(4096, bool)
        else:
            R = rotations[j]
            xx = R[0, 0]*rx+R[0, 1]*ry+R[0, 2]
            yy = R[1, 0]*rx+R[1, 1]*ry+R[1, 2]
            zz = R[2, 0]*rx+R[2, 1]*ry+R[2, 2]
            geometric = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(zz) & (zz > 0)
            px, py = np.full(4096, np.nan), np.full(4096, np.nan)
            px[geometric] = camera[0, 0]*(xx[geometric]/zz[geometric])+camera[0, 2]
            py[geometric] = camera[1, 1]*(yy[geometric]/zz[geometric])+camera[1, 2]
            hs.append(zz)
        sampled, supported = bilinear(depths[j], masks[j], px, py)
        observations.append(sampled/medians[j]); validity.append(supported & geometric)
    common = np.all(np.array(validity), axis=0)
    fit, test = common & training, common & ~training
    result["support"].update(common=int(common.sum()), train=int(fit.sum()), evaluation=int(test.sum()))
    if min(fit.sum(), test.sum()) < 256:
        return unavailable(result, "insufficient_common_support")
    values, h = np.array(observations), np.array(hs)
    all_base_positive = depths[0][valid_pixels[0]]/medians[0]
    for name, indices in (("yaw", [0, 1]), ("pitch", [2, 3])):
        fitted = shared_offset(values[0, fit], values[np.array(indices)+1][:, fit], h[indices][:, fit])
        fitted["positive_fraction"] = None
        if fitted["coefficients"] is not None:
            positive = float(np.count_nonzero(all_base_positive+fitted["coefficients"]["t0"] > 0)/len(all_base_positive))
            fitted["positive_fraction"] = positive
            if fitted["usable"] and positive < .99:
                fitted.update(usable=False, reason="full_image_positivity")
        result["offsets"][name] = fitted
    result["usable_offset"] = all(v["usable"] for v in result["offsets"].values())
    if all(v["coefficients"] is not None for v in result["offsets"].values()):
        result["offset_disagreement"] = abs(result["offsets"]["yaw"]["coefficients"]["t0"]-
                                               result["offsets"]["pitch"]["coefficients"]["t0"])
    if not result["usable_offset"]:
        result["unavailable_reason"] = "offset_fit_unavailable"
    for name, source, view_indices in ((DIRECTIONS[0], "yaw", [3, 4]), (DIRECTIONS[1], "pitch", [1, 2])):
        offset = result["offsets"][source]["coefficients"]["t0"] if result["usable_offset"] else None
        views = []
        for j in view_indices:
            zero = nuisance(values[0, fit], values[j, fit], h[j-1, fit], 0.)
            candidate = nuisance(values[0, fit], values[j, fit], h[j-1, fit], offset) if offset is not None else None
            def error(fit_result, value):
                prediction = fit_result["scale"]*(values[j, test]/h[j-1, test])+fit_result["shift"]/h[j-1, test]-value
                return float(np.sum(np.abs(prediction-values[0, test]))/int(test.sum()))
            baseline_error = error(zero, 0.)
            views.append(dict(view_index=j, zero_offset_fit=zero, candidate_fit=candidate,
                              zero_offset_error=baseline_error,
                              candidate_error=error(candidate, offset) if candidate is not None else baseline_error))
        result["directions"][name] = dict(transferred_offset=offset, fallback_to_baseline=offset is None,
            views=views, candidate_error=math.fsum(v["candidate_error"] for v in views)/2,
            zero_offset_error=math.fsum(v["zero_offset_error"] for v in views)/2)
    return result


def aggregate(rows):
    require(len(rows) == 27 and len({r["image_id"] for r in rows}) == 27, "independent aggregate identities")
    groups = {}
    for row in rows:
        a = row["analysis"]
        require(type(a["usable_offset"]) is bool, "coverage flag type")
        for direction in DIRECTIONS:
            v = a["directions"][direction]
            require(all(math.isfinite(v[k]) and v[k] >= 0 for k in ("candidate_error", "zero_offset_error")), "invalid directional loss")
            if not a["usable_offset"]:
                require(v["candidate_error"] == v["zero_offset_error"], "unavailable penalty differs")
        groups.setdefault(row["scene"], []).append(a)
    require(len(groups) == 3 and all(len(v) == 9 for v in groups.values()), "three fixed places")
    def mean(items, key):
        return math.fsum(a["directions"][d][key] for a in items for d in DIRECTIONS)/(2*len(items))
    def gain(c, b):
        return (0. if c == 0 else None) if b == 0 else 1-c/b
    scenes = {}
    for scene, items in groups.items():
        c, b = mean(items, "candidate_error"), mean(items, "zero_offset_error")
        g = gain(c, b)
        scenes[scene] = dict(image_count=9, direction_count=18, usable_offset_count=sum(a["usable_offset"] for a in items),
            candidate_mean=c, zero_offset_mean=b, relative_gain=g, strictly_improved=c < b,
            floor_pass=g is not None and g >= -.02)
    items = [r["analysis"] for r in rows]
    c, b, usable = mean(items, "candidate_error"), mean(items, "zero_offset_error"), sum(a["usable_offset"] for a in items)
    g = gain(c, b)
    gates = dict(coverage=usable >= 19 and all(s["usable_offset_count"] >= 5 for s in scenes.values()),
                 pooled_gain=g is not None and g >= .10,
                 scene_improvement=sum(s["strictly_improved"] for s in scenes.values()) >= 2,
                 scene_floor=all(s["floor_pass"] for s in scenes.values()))
    return dict(image_count=27, direction_count=54, usable_offset_count=usable, candidate_mean=c,
        zero_offset_mean=b, relative_gain=g, scenes=scenes, gates=gates, **{"pass": all(gates.values())},
        interpretation="necessary_predictor_model_transfer_test_only")


def compare(saved, independent, stats, path="analysis"):
    if isinstance(saved, dict):
        require(isinstance(independent, dict) and saved.keys() == independent.keys(), "field mismatch: "+path)
        for key in saved:
            compare(saved[key], independent[key], stats, path+"."+key)
    elif isinstance(saved, list):
        require(isinstance(independent, list) and len(saved) == len(independent), "list mismatch: "+path)
        for i, (x, y) in enumerate(zip(saved, independent)):
            compare(x, y, stats, path+"[%d]" % i)
    elif isinstance(saved, bool) or saved is None or isinstance(saved, (str, int)):
        require(saved == independent and (not isinstance(saved, bool) or type(independent) is bool), "discrete mismatch: "+path)
        stats["discrete_comparisons"] += 1
    else:
        require(isinstance(saved, float) and math.isfinite(saved) and math.isfinite(independent), "nonfinite numeric value: "+path)
        error_metric = any(k in path for k in ("_error", "_mean", "relative_gain"))
        atol, rtol = (ERROR_ATOL, ERROR_RTOL) if error_metric else (COEFF_ATOL, COEFF_RTOL)
        difference = abs(saved-independent)
        allowed = atol+rtol*max(abs(saved), abs(independent))
        require(difference <= allowed, "numerical mismatch: "+path+" difference="+str(difference)+" allowed="+str(allowed))
        stats["float_comparisons"] += 1
        stats["maximum_fraction_of_tolerance"] = max(stats["maximum_fraction_of_tolerance"], difference/allowed)


def audit(args):
    repo, output, supervision = args.repository.resolve(), args.probe_dir.resolve(), args.supervision_dir.resolve()
    snapshots = {}
    def bound(path, expected=None):
        require(path.is_file() and not path.is_symlink(), "missing/symlink evidence")
        value = digest(path)
        require(expected is None or value == expected, "hash mismatch: "+path.name)
        snapshots[path] = value
        return value
    outer_path, receipt_path = leaf(supervision, "receipt.json"), leaf(output, "receipt.json")
    bound(outer_path); bound(receipt_path)
    outer, receipt = read_json(outer_path), read_json(receipt_path)
    require(outer["schema"] == "evr0024.supervision.v1" and outer["operation"] == "probe", "outer operation")
    require(outer["status"] == "completed" and outer["exit_code"] == 0 and outer["termination"] is None
            and not outer["remaining_descendant_pids"] and not outer["cleanup_errors"], "outer not clean")
    require(outer["expected_commit"] == receipt["expected_commit"] == args.expected_commit, "source commit identity")
    require(outer["hard_timeout_seconds"] == 1800 and outer["group_rss_limit_kib"] == 8*1024**2
            and 0 <= outer["wall_seconds"] < 1800 and 0 <= outer["peak_group_rss_kib"] <= 8*1024**2,
            "outer resource limits")
    require(all(v == "4" for v in outer["thread_environment"].values()) and len(outer["thread_environment"]) == 5,
            "thread environment")
    for stream in ("stdout", "stderr"):
        bound(leaf(supervision, stream+".log"), outer[stream+"_sha256"])
    flags = command_flags(outer["command"], "experiments/0024-virtual-depth/run.py")
    require(set(flags) == {"--expected-commit", "--output-dir", "--original-frontend-supervision"}, "probe flags")
    require(flags["--expected-commit"] == args.expected_commit and public_path(flags["--output-dir"], repo) == output,
            "probe command output binding")
    original_outer = public_path(flags["--original-frontend-supervision"], repo)
    bound(original_outer, ORIGINAL_OUTER_SHA)
    original = read_json(original_outer)
    require(original["status"] == "completed" and original["exit_code"] == 0 and original["expected_commit"] == INPUT_COMMIT,
            "original frontend identity")
    for stream in ("stdout", "stderr"):
        bound(leaf(original_outer.parent, stream+".log"), original[stream+"_sha256"])
    require(receipt["schema"] == "evr0024.probe.v1" and receipt["status"] == "completed" and receipt["eligible"] is True
            and receipt["stage"] == "complete" and receipt["eligibility_requires_clean_outer_supervision"] is True,
            "probe not completed")
    require(receipt["input_freeze"] == INPUT_COMMIT and receipt["model_freeze"] == MODEL_COMMIT
            and receipt["original_frontend_supervision_sha256"] == ORIGINAL_OUTER_SHA, "frozen predecessor identities")
    require(receipt["depth_forwards_attempted"] == receipt["depth_forwards_completed"] == 135
            and receipt["feature_forwards"] == receipt["matcher_forwards"] == receipt["native_calls"] == 0
            and receipt["gt_read"] is False and receipt["offline_attempts"] == [], "forbidden or incomplete calls")
    require(0 <= receipt["wall_seconds"] <= outer["wall_seconds"], "nested timing")
    require(outer["source_sha256"] == receipt["source_sha256"] == REVIEWED, "reviewed current source set")
    seal_path, diagnostic_path = leaf(output, "seal.json"), leaf(output, "diagnostic.json")
    bound(seal_path, receipt["seal_sha256"]); bound(diagnostic_path, receipt["diagnostic_sha256"])
    seal, diagnostic = read_json(seal_path), read_json(diagnostic_path)
    require(seal["schema"] == "evr0024.probe-seal.v1" and seal["image_count"] == 27
            and seal["depth_forward_count"] == 135, "seal header")
    require(diagnostic["schema"] == "evr0024.cross-axis-diagnostic.v1"
            and diagnostic["prediction_seal_sha256"] == receipt["seal_sha256"], "diagnostic binding")
    for relative, value in seal["source_sha256"].items():
        frozen = args.expected_commit if relative in REVIEWED else INPUT_COMMIT if relative in receipt["input_source_sha256"] else MODEL_COMMIT
        require(hashlib.sha256(git_bytes(repo, frozen, relative)).hexdigest() == value, "frozen Git source mismatch: "+relative)
        bound(repo/relative, value)
    require(all(seal["source_sha256"].get(k) == v for k, v in {**REVIEWED, **receipt["input_source_sha256"]}.items()),
            "seal omits declared source")
    require(receipt["settings"]["protocol_sha256"] == REVIEWED["protocols/0024-virtual-rotation-falsifier.md"]
            and receipt["settings"]["views"] == list(VIEWS) and receipt["settings"]["tokens"] == 1200
            and receipt["settings"]["cpu_threads"] == 4, "settings")
    for key, expected in dict(num_tokens=1200, use_fp16=False, apply_mask=False, force_projection=True).items():
        require(receipt["settings"]["moge"][key] == expected, "frozen model option: "+key)
    plan_bytes = git_bytes(repo, INPUT_COMMIT, "configs/0023-prospective-plan.json")
    require(hashlib.sha256(plan_bytes).hexdigest() == PLAN_SHA, "plan identity")
    plan = json.loads(plan_bytes)
    selected = [(s["scene"], image, s["intrinsics_member"]["filename"])
                for s in plan["scenes"] for image in [s["anchor"], *s["query_images"]]]
    require(len(selected) == 27 and len(set(x[1] for x in selected)) == 27, "population")
    expected_files = ["image-%03d%s" % (i, suffix) for i in range(27) for suffix in (".json", ".npz")]
    require([v["filename"] for v in seal["outputs"]] == expected_files, "complete ordered 54-file seal")
    for item in seal["outputs"]:
        path = leaf(output, item["filename"])
        require(path.stat().st_size == item["bytes"], "sealed file size")
        bound(path, item["sha256"])
    require(len(receipt["images"]) == 27, "receipt image coverage")
    # Every sealed file and provenance guard above passes BEFORE any NPZ numeric read.
    compared = dict(discrete_comparisons=0, float_comparisons=0, maximum_fraction_of_tolerance=0.)
    recomputed_rows, saved_rows = [], []
    for i, (scene, image, intrinsics_member) in enumerate(selected):
        row = read_json(output/("image-%03d.json" % i)); small = receipt["images"][i]
        for value in (row, small):
            require((value["index"], value["scene"], value["image_id"], value["intrinsics_member"], value["status"]) ==
                    (i, scene, image, intrinsics_member, "completed"), "row identity/status")
        require(small["json_sha256"] == snapshots[output/("image-%03d.json" % i)]
                and small["usable_offset"] == row["analysis"]["usable_offset"], "small row binding")
        require(row["original_encoded_sha256"] == receipt["input_sha256"][image], "encoded input hash linkage")
        require(len(row["views"]) == 5 and [v["name"] for v in row["views"]] == list(VIEWS)
                and [v["index"] for v in row["views"]] == list(range(5)), "five forward records")
        require(all(math.isfinite(v["inference_seconds"]) and v["inference_seconds"] >= 0 for v in row["views"]), "inference timing")
        metadata = row["arrays"]; npz_path = output/("image-%03d.npz" % i)
        require(metadata["filename"] == npz_path.name and metadata["sha256"] == snapshots[npz_path]
                and metadata["bytes"] == npz_path.stat().st_size, "array metadata file binding")
        names = {"view_%d_%s" % (j, k) for j in range(5) for k in ("depth", "mask", "intrinsics")}
        with np.load(npz_path, allow_pickle=False) as data:
            require(set(data.files) == names == set(metadata["arrays"]), "exact raw array keys")
            arrays = {}
            for name in sorted(names):
                a = data[name]
                descriptor = metadata["arrays"][name]
                require(list(a.shape) == descriptor["shape"] and str(a.dtype) == descriptor["dtype"]
                        and hashlib.sha256(a.tobytes(order="C")).hexdigest() == descriptor["raw_c_order_sha256"], "raw array hash/type/shape")
                arrays[name] = a
        depths = np.array([arrays["view_%d_depth" % j] for j in range(5)], dtype=np.float64)
        masks = np.array([arrays["view_%d_mask" % j] for j in range(5)])
        require(depths.ndim == 3 and masks.shape == depths.shape and masks.dtype == bool
                and min(depths.shape[1:]) >= 2 and max(depths.shape[1:]) <= 512, "raw depth dimensions")
        camera, rotations = np.array(row["geometry"]["Kc"]), np.array(row["geometry"]["rotations"])
        height, width = depths.shape[1:]
        require(camera.shape == (3, 3) and np.isfinite(camera).all() and camera[0, 0] == camera[1, 1]
                and camera[0, 0] > 0 and camera[0, 2] == (width-1)/2 and camera[1, 2] == (height-1)/2
                and np.array_equal(camera[2], [0, 0, 1]) and rotations.shape == (5, 3, 3), "canonical geometry")
        require(np.array_equal(rotations[0], np.eye(3)) and np.isfinite(rotations).all(), "rotation identity/finite")
        for R in rotations:
            require(np.max(np.abs(np.sum(R[:, :, None]*R[:, None, :], axis=0)-np.eye(3))) <= 1e-10
                    and abs(np.linalg.det(R)-1) <= 1e-10, "proper rotation")
        normalized = np.array([[camera[0, 0]/width, 0., .5], [0., camera[1, 1]/height, .5], [0., 0., 1.]])
        for j in range(5):
            actual = arrays["view_%d_intrinsics" % j]
            require(actual.shape == (3, 3) and np.isfinite(actual).all()
                    and np.all(np.abs(actual-normalized) <= 1e-5*np.abs(normalized)), "raw known-FOV intrinsics")
        independent = independent_analysis(depths, masks, camera, rotations)
        compare(row["analysis"], independent, compared, "image[%d].analysis" % i)
        recomputed_rows.append(dict(scene=scene, image_id=image, analysis=independent)); saved_rows.append(row)
    summary = aggregate(recomputed_rows)
    compare(diagnostic["summary"], aggregate(saved_rows), compared, "saved_aggregate")
    compare(diagnostic["summary"], summary, compared, "recomputed_aggregate")
    for path, before in snapshots.items():
        require(digest(path) == before, "evidence changed during audit: "+path.name)
    return dict(status="passed", independent_summary=summary, comparisons=compared,
        image_count=27, direction_count=54, forward_count=135, raw_arrays_checked=405,
        expected_commit=args.expected_commit, original_decision=diagnostic["summary"]["pass"],
        receipt_sha256=snapshots[receipt_path], seal_sha256=snapshots[seal_path],
        diagnostic_sha256=snapshots[diagnostic_path], supervision_sha256=snapshots[outer_path],
        supervision_log_sha256={k: outer[k+"_sha256"] for k in ("stdout", "stderr")},
        source_sha256=seal["source_sha256"], recomputed_rows=recomputed_rows,
        scope="Full saved-depth/mask numerical reconstruction, exact identities/coverage/discrete gates, source/seal/log hashes and outer resource limits.",
        limitations=["No image decoding, learned inference or model accuracy verification.",
            "Uses sealed Kc/rotations; does not recompute RGB remapping or canonical focal search from original images.",
            "Checks frozen predecessor/source/hash links, not a fresh traversal of all installer/model/acquisition payloads.",
            "Call counts are verified recorded counters and five-view records, not independent hardware instrumentation.",
            "A passing diagnostic is not evidence of improved physical depth or pose accuracy."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "probe-dir", "supervision-dir", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "fresh audit output directory required")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    receipt = dict(schema="evr0024.independent-audit.v1", status="started", script_sha256=digest(__file__),
        tolerances=dict(coefficient_atol=COEFF_ATOL, coefficient_rtol=COEFF_RTOL,
                        error_atol=ERROR_ATOL, error_rtol=ERROR_RTOL, discrete="exact"))
    path = args.output_dir/"audit.json"
    atomic_json(path, receipt)
    try:
        receipt.update(audit(args))
    except BaseException as error:
        receipt.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter()-started
        atomic_json(path, receipt)
    print(json.dumps(dict(status=receipt["status"], decision=receipt["original_decision"],
                          wall_seconds=receipt["wall_seconds"], comparisons=receipt["comparisons"])))


if __name__ == "__main__":
    main()
