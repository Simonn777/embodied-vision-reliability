"""Causal coefficient head. No imports, model loading, or work on import.

Torch and the pure family module are explicit dependencies. Callers must supply
current causal inputs; this interface cannot establish their external provenance.
All input statistics are detached. No future targets enter this module.
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration():
    return dict(input_features=131, hidden_features=128, output_features=128,
                layers=["Linear(131,128)", "GELU", "Linear(128,128)"],
                parameter_count=33408, output_initialization="all zeros",
                memory_mean="sign(mean)*log1p(abs(mean))",
                memory_std="log1p(population_std)",
                tail_order=["sigmoid(current_object_score_logit)",
                            "current_selected_predicted_iou",
                            "mean(current_mask_probability > 0.5)"],
                input_statistics="detached float64 then float32",
                batch_normalization=False, coefficient_order="[channel,(a,b)]",
                scaling="dimensionless output * channel_sigma * rho",
                final_bound="family.project measured pre-BF16 energy",
                gradient_scope="head only; causal input statistics detached")


def build_head(*, torch, device="cpu"):
    """Use the caller's RNG state; do not reseed or create an optimizer."""
    head = torch.nn.Sequential(
        torch.nn.Linear(131, 128, device=device, dtype=torch.float32),
        torch.nn.GELU(),
        torch.nn.Linear(128, 128, device=device, dtype=torch.float32))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def _scalar(value, *, torch, device, label):
    value = torch.as_tensor(value, device=device)
    _require(value.numel() == 1 and not value.is_complex()
             and value.dtype != torch.bool, label + " scalar")
    value = value.detach().to(torch.float64).reshape(1)
    _require(bool(torch.isfinite(value).all().item()), label + " nonfinite")
    return value


def extract_input(context, object_score_logit, predicted_iou, mask_probability, *, torch):
    """Return detached [1,131], with exactly the ordering in configuration()."""
    _require(context["shape"][0:2] == (1, 64), "head requires 64 memory channels")
    mean, sigma = context["mean"].detach(), context["sigma"].detach()
    _require(tuple(mean.shape) == tuple(sigma.shape) == (64,), "statistic shape")
    _require(bool(torch.isfinite(mean).all().item())
             and bool(torch.isfinite(sigma).all().item())
             and bool((sigma >= 0).all().item()), "invalid causal statistics")
    _require(mask_probability.ndim == 4 and tuple(mask_probability.shape[:2]) == (1, 1)
             and mask_probability.numel() > 0 and mask_probability.is_floating_point(),
             "current mask probability must be [1,1,H,W]")
    _require(mask_probability.device == mean.device, "current mask device")
    probability = mask_probability.detach()
    _require(bool(torch.isfinite(probability).all().item())
             and bool(((probability >= 0) & (probability <= 1)).all().item()),
             "invalid current mask probability")
    logit = _scalar(object_score_logit, torch=torch, device=mean.device, label="object score")
    iou = _scalar(predicted_iou, torch=torch, device=mean.device, label="predicted IoU")
    area = (probability > .5).to(torch.float64).mean().reshape(1)
    features = torch.cat((mean.sign() * mean.abs().log1p(), sigma.log1p(),
                          logit.sigmoid(), iou, area)).to(torch.float32).reshape(1, 131)
    _require(bool(torch.isfinite(features).all().item()), "unrepresentable head input")
    return features


def predict_coefficients(head, context, object_score_logit, predicted_iou,
                         mask_probability, *, torch, family):
    """Predict dimensionless coordinates, scale by sigma*rho, then project.

    Returns graph-carrying coefficient tensors, not an implicit KD loss. Raw
    selected predicted-IoU is retained without clipping. Exact-zero variance
    channels produce zero coefficients regardless of the head output.
    """
    features = extract_input(context, object_score_logit, predicted_iou,
                             mask_probability, torch=torch)
    raw = head(features)
    _require(tuple(raw.shape) == (1, 128) and raw.dtype == torch.float32
             and raw.device == context["basis"].device, "head output contract")
    _require(bool(torch.isfinite(raw.detach()).all().item()), "nonfinite head output")
    dimensionless = raw.reshape(64, 2)
    scale = (context["sigma"] * context["rho"]).to(torch.float32)
    _require(bool(torch.isfinite(scale).all().item()), "unrepresentable channel scale")
    scaled = dimensionless * scale[:, None]
    coefficients = family.project(context, scaled, torch=torch)
    return dict(features=features, dimensionless=dimensionless,
                unprojected=scaled, coefficients=coefficients)
