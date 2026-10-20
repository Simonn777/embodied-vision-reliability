"""Raw MoGe2 output adapters; no model loading or inference at module import.

A/B reproduce the pinned author's *depth/normal/mask/intrinsics* postprocessing
with apply_mask=False. C is explicitly a general-camera shift adaptation.
Point reprojection is not returned: it cannot affect these four outputs.
Upstream algorithms remain in their installed, separately licensed source tree.
"""
import hashlib
import importlib
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import warnings

MOGE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
SOURCE_HASHES = {
    "moge.utils.geometry_torch": "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05",
    "moge.utils.geometry_numpy": "f540c5b8ef55bfffcc32b1a62b38be605683e29fbf399ca86dfcbfc458ce4c10",
    "utils3d.torch.utils": "414059eec3a446ae25433ec1fea8d58d5b49a79ecfa4bc4fd317c98a4002f9a3",
}
REFINE_OPTIONS = dict(iterations=10, damp=1e-3, eps=1e-12, kernel_size=5)
MODES = ("A", "B", "C")


class ContractError(ValueError):
    """Malformed adapter input or an incompatible/missing upstream provider."""


def load_provider():
    """Use the caller's admitted local installation; never change sys.path."""
    modules = {}
    for name, expected in SOURCE_HASHES.items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise ContractError("upstream_source_hash_mismatch:" + name)
        modules[name] = module
    tg, ng = modules["moge.utils.geometry_torch"], modules["moge.utils.geometry_numpy"]
    torch = importlib.import_module("torch")
    utils3d = importlib.import_module("utils3d")
    return SimpleNamespace(
        torch=torch,
        recover_focal_shift=tg.recover_focal_shift,
        solve_optimal_shift=ng.solve_optimal_shift,
        refine_depth_with_normal=tg.refine_depth_with_normal,
        intrinsics_from_focal_center=utils3d.pt.intrinsics_from_focal_center,
        window_namespace=utils3d.pt, window_compatibility=False,
        provenance={"kind": "pinned_author_functions", "moge_revision": MOGE_REVISION,
                    "source_sha256": dict(SOURCE_HASHES)})


def _provider(provider):
    provider = load_provider() if provider is None else provider
    required = ("torch", "recover_focal_shift", "solve_optimal_shift",
                "refine_depth_with_normal", "intrinsics_from_focal_center", "provenance")
    if any(not hasattr(provider, key) for key in required):
        raise ContractError("incomplete_provider")
    return provider


def with_window_compatibility(provider):
    """Opt in to a scoped alias; do not edit an installed module on disk."""
    p = _provider(provider)
    namespace = getattr(p, "window_namespace", None)
    if namespace is None or not callable(getattr(namespace, "sliding_window", None)):
        raise ContractError("missing_upstream_sliding_window_provider")
    copied = SimpleNamespace(**vars(p))
    copied.window_compatibility = True
    return copied


def last_two_axis_window(sliding_window, tensor, window_size, stride=1, dim=(-2, -1)):
    """Exact view alias for the single missing helper API, with no padding."""
    if tuple(dim) != (-2, -1):
        raise ContractError("compatibility_alias_only_supports_last_two_axes")
    return sliding_window(tensor, window_size=window_size, stride=stride, dim=(-2, -1))


@contextmanager
def _window_scope(provider, metadata):
    metadata.update(requested=bool(getattr(provider, "window_compatibility", False)),
                    applied=False, restored=False)
    if not metadata["requested"]:
        yield
        return
    namespace = provider.window_namespace
    sentinel = object()
    previous = getattr(namespace, "sliding_window_2d", sentinel)
    if previous is not sentinel:
        raise ContractError("unexpected_existing_sliding_window_2d_no_override")
    sliding = namespace.sliding_window
    def alias(tensor, window_size, stride=1, dim=(-2, -1)):
        return last_two_axis_window(sliding, tensor, window_size, stride, dim)
    setattr(namespace, "sliding_window_2d", alias)
    metadata.update(applied=True, alias="last_two_axis_window",
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        yield
    finally:
        delattr(namespace, "sliding_window_2d")
        metadata["restored"] = True


def _camera(K_pixels, height, width, torch):
    K = torch.as_tensor(K_pixels, dtype=torch.float32, device="cpu").clone()
    if tuple(K.shape) != (3, 3) or not bool(torch.isfinite(K).all()):
        raise ContractError("camera_shape_or_finite")
    if (float(K[0, 1]) != 0 or float(K[1, 0]) != 0 or
            not bool(torch.equal(K[2], torch.tensor([0., 0., 1.]))) or
            float(K[0, 0]) <= 0 or float(K[1, 1]) <= 0 or height < 1 or width < 1):
        raise ContractError("camera_must_be_positive_focal_zero_skew_PINHOLE")
    return K


def normalized_intrinsics(K_pixels, height, width, *, torch_module=None):
    """Integer pixel-center camera -> author's [0,1] cell-center UV camera."""
    torch = importlib.import_module("torch") if torch_module is None else torch_module
    K = _camera(K_pixels, height, width, torch)
    K[0, 0] /= width
    K[1, 1] /= height
    K[0, 2] = (K[0, 2] + 0.5) / width
    K[1, 2] = (K[1, 2] + 0.5) / height
    return K


def _pixel_intrinsics(K, height, width):
    result = K.clone()
    result[0, 0] *= width
    result[1, 1] *= height
    result[0, 2] = result[0, 2] * width - 0.5
    result[1, 2] = result[1, 2] * height - 0.5
    return result


def nearest_indices(length, *, torch_module=None):
    """Actual PyTorch nearest sampling indices, including repeated upsample rows."""
    torch = importlib.import_module("torch") if torch_module is None else torch_module
    if not isinstance(length, int) or length < 1:
        raise ContractError("invalid_sample_length")
    x = torch.arange(length, dtype=torch.float64).reshape(1, 1, length, 1)
    return torch.nn.functional.interpolate(x, (64, 1), mode="nearest")[0, 0, :, 0].long()


def _raw(raw, torch):
    if not isinstance(raw, dict) or not {"points", "normal", "mask", "metric_scale"} <= set(raw):
        raise ContractError("raw_forward_outputs_missing")
    values = {}
    for name in ("points", "normal", "mask", "metric_scale"):
        value = raw[name]
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ContractError("raw_outputs_must_be_CPU:" + name)
        try:
            values[name] = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ContractError("raw_output_not_numeric_array:" + name) from exc
    points, normal, prob, scale = (values[k] for k in ("points", "normal", "mask", "metric_scale"))
    if points.ndim == 3:
        points, normal, prob = points.unsqueeze(0), normal.unsqueeze(0), prob.unsqueeze(0)
    if (points.ndim != 4 or points.shape[0] != 1 or points.shape[-1] != 3 or
            tuple(normal.shape) != tuple(points.shape) or
            tuple(prob.shape) != tuple(points.shape[:-1]) or scale.numel() != 1):
        raise ContractError("single_image_forward_shape")
    if not bool(torch.isfinite(prob).all()) or bool(((prob < 0) | (prob > 1)).any()):
        raise ContractError("mask_must_be_finite_probability")
    scale = scale.reshape(1)
    if not bool(torch.isfinite(scale).all()) or float(scale[0]) <= 0:
        raise ContractError("metric_scale_must_be_positive_finite")
    return points, normal, prob > 0.5, scale


def _warning_records(caught):
    return [{"category": item.category.__name__, "message": str(item.message)}
            for item in caught]


def _finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _validity(depth, normal, mask, K, torch):
    normal_finite = torch.isfinite(normal).all(dim=-1)
    return {
        "depth_all_finite": bool(torch.isfinite(depth).all()),
        "normal_all_finite": bool(normal_finite.all()),
        "intrinsics_finite": bool(torch.isfinite(K).all()),
        "depth_nonfinite_count": int((~torch.isfinite(depth)).sum()),
        "depth_nonpositive_count": int((depth <= 0).sum()),
        "normal_nonfinite_count": int((~normal_finite).sum()),
        "normal_zero_count": int((normal.square().sum(dim=-1) == 0).sum()),
        "mask_count": int(mask.sum()),
        "masked_depth_finite_positive": bool((torch.isfinite(depth[mask]) & (depth[mask] > 0)).all()),
        "masked_normal_finite": bool(normal_finite[mask].all()),
    }


def postprocess(raw, K_pixels, mode, *, provider=None):
    """Accept CPU Torch/NumPy arrays; return unbatched CPU float32/bool tensors.

    Input contract errors raise. Author helper exceptions are returned as
    helper_error with warnings retained; numerical outputs are never repaired.
    Except named tensor fields, metadata is JSON-safe. Status is an arithmetic
    report, not scientific eligibility. Convert tensor fields with .numpy()
    for serialization; no raw model input is mutated.
    """
    if mode not in MODES:
        raise ContractError("unknown_mode")
    p = _provider(provider)
    torch = p.torch
    points, normal, original_mask, scale = _raw(raw, torch)
    height, width = points.shape[1:3]
    K = _camera(K_pixels, height, width, torch)
    rows, cols = nearest_indices(height, torch_module=torch), nearest_indices(width, torch_module=torch)
    sampled_mask = original_mask[0][rows[:, None], cols[None, :]]
    sample_count = int(sampled_mask.sum())
    result = {
        "schema": "evr0028.point-postprocess.v1", "mode": mode, "status": "started",
        "provenance": dict(p.provenance), "warnings": [],
        "original_mask": original_mask[0].clone(),
        "recovery": {"metric_scale": float(scale[0]), "sampling": "torch-nearest-64x64",
                     "apply_mask": False, "normal_unchanged": True,
                     "original_mask_count": int(original_mask.sum()),
                     "sampled_mask_count": sample_count, "sample_rows": rows.tolist(),
                     "sample_columns": cols.tolist(), "insufficient_support_fallback": sample_count < 2,
                     "camera_model": "authored_zero_skew" if mode == "C" else "official_centered_square_pixels"},
    }
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            if mode in ("A", "B"):
                if mode == "A":
                    focal, shift = p.recover_focal_shift(points.clone(), original_mask.clone())
                else:
                    fov = math.degrees(2 * math.atan(width / (2 * float(K[0, 0]))))
                    aspect = width / height
                    focal = aspect / math.sqrt(1 + aspect ** 2) / torch.tan(
                        torch.deg2rad(torch.as_tensor(fov, dtype=points.dtype) / 2))
                    focal = focal[None].expand(1)
                    _, shift = p.recover_focal_shift(points.clone(), original_mask.clone(), focal=focal.clone())
                    result["recovery"]["supplied_horizontal_fov_degrees"] = fov
                focal, shift = focal.reshape(1), shift.reshape(1)
                aspect = width / height
                fx = focal / 2 * math.sqrt(1 + aspect ** 2) / aspect
                fy = focal / 2 * math.sqrt(1 + aspect ** 2)
                intrinsic = p.intrinsics_from_focal_center(
                    fx, fy, torch.tensor(0.5, dtype=points.dtype), torch.tensor(0.5, dtype=points.dtype))[0]
                result["recovery"]["focal_relative_to_half_diagonal"] = _finite_scalar(focal[0])
            else:
                y, x = torch.meshgrid(torch.arange(height, dtype=points.dtype),
                                      torch.arange(width, dtype=points.dtype), indexing="ij")
                rays = torch.stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]), dim=-1)
                sampled_points = points[0][rows[:, None], cols[None, :]]
                sampled_rays = rays[rows[:, None], cols[None, :]]
                result["recovery"]["fixed_focal"] = 1.0
                if sample_count < 2:
                    # Match the author's insufficient-sampled-support fallback,
                    # recording it explicitly rather than claiming a solved fit.
                    shift = torch.zeros(1, dtype=points.dtype)
                else:
                    value = p.solve_optimal_shift(sampled_rays[sampled_mask].numpy().copy(),
                                                  sampled_points[sampled_mask].numpy().copy(), focal=1.0)
                    shift = torch.as_tensor(value, dtype=points.dtype).reshape(1)
                intrinsic = normalized_intrinsics(K, height, width, torch_module=torch)
            shifted = points[..., 2] + shift[:, None, None]
            mask = original_mask & (shifted > 0)
            depth = shifted * scale[:, None, None]
            result.update(depth=depth[0], normal=normal[0].clone(), mask=mask[0],
                          intrinsics=intrinsic,
                          K_pixels=K.clone() if mode == "C" else _pixel_intrinsics(intrinsic, height, width))
            result["recovery"]["shift_raw_point_units"] = _finite_scalar(shift[0])
            result["validity"] = _validity(depth[0], normal[0], mask[0], intrinsic, torch)
            finite = all(result["validity"][key] for key in
                         ("depth_all_finite", "normal_all_finite", "intrinsics_finite"))
            positive_focal = float(intrinsic[0, 0]) > 0 and float(intrinsic[1, 1]) > 0
            result["status"] = "completed" if finite and positive_focal else "numerical_failure"
    except Exception as exc:
        result.update(status="helper_error", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        result["warnings"] = _warning_records(caught)
    return result


def postprocess_all(raw, K_pixels, *, provider=None):
    p = _provider(provider)
    return {mode: postprocess(raw, K_pixels, mode, provider=p) for mode in MODES}


def refine_with_normal(depth, normal, K_pixels, *, provider=None):
    """CPU Torch/NumPy input; float32 tensor output, exact body on private copies.

    Call with_window_compatibility(load_provider()) explicitly to opt into the
    documented temporary missing-window-name alias. Without it, author API
    failures remain visible. This function does not apply any validity mask.
    """
    p = _provider(provider)
    torch = p.torch
    if any(isinstance(x, torch.Tensor) and x.device.type != "cpu" for x in (depth, normal)):
        raise ContractError("refinement_requires_CPU")
    depth = torch.as_tensor(depth, dtype=torch.float32, device="cpu")
    normal = torch.as_tensor(normal, dtype=torch.float32, device="cpu")
    if depth.ndim != 2 or tuple(normal.shape) != tuple(depth.shape) + (3,):
        raise ContractError("refinement_tensor_shape")
    height, width = depth.shape
    intrinsic = normalized_intrinsics(K_pixels, height, width, torch_module=torch)
    d = depth.detach().to(dtype=torch.float32).clone()
    n = normal.detach().to(dtype=torch.float32).clone()
    result = {"schema": "evr0028.normal-helper.v1", "status": "started",
              "provenance": dict(p.provenance), "options": dict(REFINE_OPTIONS),
              "intrinsics": intrinsic.clone(), "warnings": [], "compatibility": {},
              "input_depth_nonpositive_count": int((d <= 0).sum()),
              "input_depth_nonfinite_count": int((~torch.isfinite(d)).sum())}
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            with _window_scope(p, result["compatibility"]):
                out = p.refine_depth_with_normal(d, n, intrinsic.clone(), **REFINE_OPTIONS)
            if not isinstance(out, torch.Tensor) or tuple(out.shape) != tuple(depth.shape):
                raise ContractError("helper_output_shape")
            result["depth"] = out.detach().clone()
            result["finite"] = bool(torch.isfinite(out).all())
            result["positive"] = bool((out > 0).all())
            result["nonfinite_count"] = int((~torch.isfinite(out)).sum())
            result["status"] = "completed" if result["finite"] and result["positive"] else "numerical_failure"
    except Exception as exc:
        result.update(status="helper_error", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        result["warnings"] = _warning_records(caught)
    return result
