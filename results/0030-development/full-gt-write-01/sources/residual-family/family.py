"""Pure causal channel-affine memory residuals; no Torch import or work on import.

Torch is passed explicitly. Statistics are detached float64, the stored basis
and coefficients are float32. The measured basis Gram defines pre-BF16 energy.
No claim of bit-exact LS, post-quantization bounds, or segmentation improvement.
"""
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, torch, message):
    _require(bool(torch.isfinite(value.detach()).all().item()), message)


def prepare(memory, rho, *, torch):
    """Build a family from one [1,C,H,W] float32/BF16 causal memory."""
    _require(memory.ndim == 4 and memory.shape[0] == 1
             and all(n > 0 for n in memory.shape), "expected nonempty [1,C,H,W]")
    _require(memory.dtype in (torch.float32, torch.bfloat16), "memory dtype")
    _require(type(rho) in (int, float) and math.isfinite(rho) and rho >= 0, "rho")
    _finite(memory, torch, "nonfinite memory")
    values = memory.detach().to(torch.float64).flatten(2)[0]
    mean = values.mean(dim=1)
    centered = values - mean[:, None]
    variance = centered.square().mean(dim=1)
    sigma = variance.sqrt()
    active = variance > 0
    safe_sigma = torch.where(active, sigma, torch.ones_like(sigma))
    basis = (centered / safe_sigma[:, None]).to(torch.float32)
    basis = basis * active[:, None]
    measured = basis.to(torch.float64)
    g00, g01 = measured.square().mean(dim=1), measured.mean(dim=1)
    g11 = active.to(torch.float64)
    gram = torch.stack((g00, g01, g01, g11), dim=1).reshape(-1, 2, 2)
    determinant = g00 * g11 - g01.square()
    _require(bool((determinant[active] > 0).all().item()), "singular active basis")
    radius = math.sqrt(float(variance.sum().item())) * rho
    _require(math.isfinite(radius) and math.isfinite(radius * radius)
             and (radius == 0 or radius * radius > 0), "unrepresentable energy radius")
    return dict(shape=tuple(memory.shape), basis=basis, mean=mean, sigma=sigma,
                active=active, gram=gram, radius=radius, rho=float(rho))


def _coefficients(context, coefficients, torch):
    _require(tuple(coefficients.shape) == (context["shape"][1], 2)
             and coefficients.dtype == torch.float32, "expected float32 [C,2]")
    _require(coefficients.device == context["basis"].device, "coefficient device")
    _finite(coefficients, torch, "nonfinite coefficients")
    return coefficients * context["active"][:, None]


def coefficient_energy(context, coefficients, *, torch):
    """Sum of per-channel spatial mean-square residuals, using measured Gram."""
    q = _coefficients(context, coefficients, torch).to(torch.float64)
    energy = torch.einsum("ci,cij,cj->", q, context["gram"], q)
    _finite(energy, torch, "nonfinite energy")
    _require(float(energy.detach().item()) >= 0, "negative basis energy")
    return energy


def project(context, coefficients, *, torch):
    """Radial measured-energy projection. Identity derivative at active zero."""
    q = _coefficients(context, coefficients, torch)
    radius = context["radius"]
    if radius == 0:
        return q * 0
    energy = coefficient_energy(context, q, torch=torch)
    denominator = torch.maximum(energy, energy.new_tensor(radius * radius)).sqrt()
    # The floor is BEFORE sqrt: no hidden sqrt(0) backward at zero initialization.
    result = (q.to(torch.float64) * (radius / denominator)).to(torch.float32)
    _finite(result, torch, "nonfinite projection")
    return result


def expand(context, coefficients, *, torch):
    """Expand already-chosen coefficients; does not silently project them."""
    q = _coefficients(context, coefficients, torch)
    delta = q[:, :1] * context["basis"] + q[:, 1:2]
    _finite(delta, torch, "nonfinite expanded residual")
    return delta.reshape(context["shape"])


def least_squares_target(context, target_delta, *, torch):
    """Detached GT-difference LS in the measured family, then the same bound."""
    _require(tuple(target_delta.shape) == context["shape"]
             and target_delta.is_floating_point(), "target shape/dtype")
    _require(target_delta.device == context["basis"].device, "target device")
    _finite(target_delta, torch, "nonfinite target")
    target = target_delta.detach().to(torch.float64).flatten(2)[0]
    basis, active, gram = context["basis"].double(), context["active"], context["gram"]
    r0, r1 = (basis * target).mean(dim=1), target.mean(dim=1)
    g00, g01, g11 = gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1]
    determinant = g00 * g11 - g01.square()
    safe = torch.where(active, determinant, torch.ones_like(determinant))
    q = torch.stack(((g11 * r0 - g01 * r1) / safe,
                     (g00 * r1 - g01 * r0) / safe), dim=1)
    q = (q * active[:, None]).to(torch.float32)
    _finite(q, torch, "unrepresentable target coefficients")
    bounded = project(context, q, torch=torch)
    return dict(unconstrained=q, coefficients=bounded,
                diagnostics=describe(context, bounded, torch=torch))


def describe(context, coefficients=None, *, torch):
    """Small JSON-safe statistics; energy refers to float32 pre-BF16 expansion."""
    result = dict(shape=list(context["shape"]), active=context["active"].tolist(),
                  sigma=context["sigma"].tolist(),
                  basis_means=context["gram"][:, 0, 1].tolist(),
                  measured_gram=context["gram"].tolist(), radius=context["radius"],
                  radius_squared=context["radius"] ** 2, rho=context["rho"],
                  scope="pre-BF16; finite-precision LS/energy, not bit-exact identities")
    if coefficients is not None:
        theoretical = float(coefficient_energy(context, coefficients, torch=torch).detach())
        actual = float(expand(context, coefficients, torch=torch).detach().double()
                       .flatten(2).square().mean(dim=2).sum())
        result.update(gram_energy=theoretical, actual_energy=actual,
                      actual_residual_rms=math.sqrt(actual / context["shape"][1]),
                      rounding_energy_difference=actual - theoretical,
                      actual_energy_excess=max(0., actual - context["radius"] ** 2))
    return result
