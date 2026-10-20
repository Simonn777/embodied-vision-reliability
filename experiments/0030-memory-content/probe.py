"""Manufactured SAM2 spatial-memory gradient qualification; no work on import.

Private script: reviewed --source-sha256 binds its exact bytes. Existing 0029
native admission and supervision are reused. No scientific inputs are accepted.
BF16-cast backward is PyTorch's surrogate, not the quantizer's mathematical
derivative. Raw-mask/presence BCE is explicitly distinct from native hard gating.
"""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
ATOL, RTOL = 1e-4, 1e-5


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(args):
    root, native = Path(args.repo_root).resolve(), Path(args.native_root).resolve()
    out = Path(args.output_dir).resolve()
    if out.exists() or any(p == out or p in out.parents for p in (root, native)):
        raise ValueError("fresh private output outside repository/native installation required")
    out.mkdir(parents=True)
    started = time.perf_counter()
    record = dict(schema="evr0030.gradient-probe.v1", status="started", eligible=False,
        manufactured_only=True, scientific_input_access=False, expected_commit=args.expected_commit,
        native_commit=args.native_commit, offline_attempts=[], compatibility_probes=[], warnings=[],
        timings=[], calls=[], outputs=[], checks={}, baseline_frames=[], image_encoder_calls=[])

    def save():
        record["wall_seconds"] = time.perf_counter() - started
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")

    @contextlib.contextmanager
    def timed(name):
        entry = dict(stage=name, status="started")
        record["timings"].append(entry)
        save()
        began = time.perf_counter()
        try:
            yield
            entry["status"] = "completed"
        except BaseException:
            entry["status"] = "failed"
            raise
        finally:
            entry["seconds"] = time.perf_counter() - began
            save()

    save()
    try:
        with timed("source_and_native_admission"):
            h = module("probe_install", root / OWN / "install_runtime.py")
            old = module("probe_admission", root / OWN / "run_evidence.py")
            h.require(h.sha(Path(__file__)) == args.source_sha256, "wrong reviewed probe source")
            paths = [OWN + n for n in ("install_runtime.py", "run_evidence.py", "native_fixture.py",
                     "supervise_exit_checked.py")] + ["experiments/0022-frontend/qualify.py", old.MONITOR]
            if Path(__file__).resolve().is_relative_to(root):
                paths.append(str(Path(__file__).resolve().relative_to(root)))
            record["source_sha256"] = h.frozen(root, args.expected_commit, paths)
            record["executing_source_sha256"] = args.source_sha256
            installation, installed, record["native_binding"] = old.native_contract(
                h, root, native, args.native_commit)
            h.require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1 and
                      Path(sys.executable).absolute() == installation / "venv/bin/python", "guarded CP312 required")
            h.require(all(os.environ.get(k) == "4" for k in h.THREAD_KEYS), "four-thread environment")
            q = module("probe_guard", root / "experiments/0022-frontend/qualify.py")
            nf = module("probe_native_fixture", root / OWN / "native_fixture.py")
            q.install_guard(record["offline_attempts"], record["compatibility_probes"])
            sys.dont_write_bytecode = True
            os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
            import site
            site.main()
            h.require(Path(sys.prefix).resolve() == installation / "venv", "runtime prefix")
            sys.path.insert(0, str(installation / "sources/sam2"))
        with timed("imports_and_exact_loader"), q.capture_warnings(record):
            import numpy as np
            import torch
            import cv2
            from PIL import Image
            from torch.nn import functional as F
            from sam2.build_sam import build_sam2_video_predictor
            for name, prefix in (("sam2", installation / "sources/sam2"),
                    ("torch", installation / "venv"), ("numpy", installation / "venv"),
                    ("PIL", installation / "venv"), ("cv2", installation / "venv")):
                h.require(Path(sys.modules[name].__file__).resolve().is_relative_to(prefix), "module origin: " + name)
            torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(0)
            cv2.setNumThreads(0)
            h.require((torch.get_num_threads(), torch.get_num_interop_threads(), cv2.getNumThreads()) == (4, 1, 1), "threads")
            model = build_sam2_video_predictor(nf.CONFIG, ckpt_path=None, device="cpu", mode="eval",
                apply_postprocessing=False, hydra_overrides_extra=list(nf.OVERRIDES), vos_optimized=False)
            checkpoint = torch.load(installation / "weights/sam2.1_hiera_tiny.pt", map_location="cpu", weights_only=True)
            record["load"] = q.strict_state(model, checkpoint["model"], torch)
            del checkpoint
            model.eval().to(device="cpu", dtype=torch.float32).requires_grad_(False)
            h.require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()), "model dtype/device")
            h.require(model.num_maskmem == 7 and model.memory_temporal_stride_for_eval == 1 and
                      model.fill_hole_area == 0 and model.mem_dim == 64, "native model configuration")
            record["config"] = dict(config=nf.CONFIG, overrides=nf.OVERRIDES, torch_version=torch.__version__,
                device="cpu", parameter_dtype="float32", autocast=False, threads=4, interop=1,
                frozen_eval=True, grad_forward_atol=ATOL, grad_forward_rtol=RTOL,
                surrogate="native-pIoU-selected raw pre-presence-gate mask BCE + presence BCE(target=1)",
                bf16_backward="PyTorch cast surrogate; not mathematical quantizer derivative",
                head="single 64-to-64 1x1 Conv; zero weights/bias; residual=0.01*tanh; SGD lr=0.001; one step",
                head_is_final_campaign_architecture=False)
            model_hash = nf.state_hash(model.state_dict(), torch)
            encoder_hook = model.image_encoder.register_forward_hook(
                lambda m, a, v: record["image_encoder_calls"].append(list(a[0].shape)))
        with timed("manufactured_fixture"):
            images, description = q.fixture(np)
            base = images[0]; frames = out / "frames"; frames.mkdir()
            record["fixture"] = dict(base=description, frames=9, probe_frame=7, written_memory_frame=6,
                shift_per_frame=12, jpeg_quality=95, jpeg_subsampling=0, padding="zero",
                quantization="floor(255*clip(image,0,1)+0.5)", initial_mask_yx=[118, 182, 207, 271])
            for i in range(9):
                image = np.zeros_like(base); shift = 12 * i
                image[:, shift:] = base[:, :512-shift]
                path = frames / ("%05d.jpg" % i)
                with path.open("xb") as stream:
                    Image.fromarray(np.floor(255*np.clip(image, 0, 1)+.5).astype(np.uint8)).save(
                        stream, format="JPEG", quality=95, subsampling=0)
                record["outputs"].append(dict(filename=str(path.relative_to(out)), sha256=h.sha(path), bytes=path.stat().st_size))
            initial = np.zeros((384, 512), dtype=np.bool_); initial[118:182, 207:271] = True
            target = np.zeros_like(initial); target[:, 84:] = initial[:, :-84]
            record["outputs"].append(q.save_arrays(out / "manufactured-masks.npz", {"initial": initial, "future": target}, np))
        with timed("native_history_and_one_future_baseline"), torch.inference_mode(), q.capture_warnings(record):
            state = model.init_state(str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=False)
            model.add_new_mask(state, frame_idx=0, obj_id=1, mask=initial)
            baseline_decoder = {}
            hook = model.sam_mask_decoder.register_forward_hook(lambda m, a, values: baseline_decoder.update(values=values))
            try:
                iterator = model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=7)
                for expected in range(8):
                    frame, ids, _ = next(iterator)
                    h.require(frame == expected and ids == [1], "native baseline identity")
                    record["baseline_frames"].append(frame)
                h.require(next(iterator, None) is None, "unexpected baseline frame")
            finally:
                hook.remove()
            _, _, vision0, pos0, sizes = model._get_image_feature(state, 7, 1)
            history0 = state["output_dict_per_obj"][0]
            baseline_low = history0["non_cond_frame_outputs"][7]["pred_masks"]
        def ordinary(value):
            if isinstance(value, torch.Tensor):
                result = value.detach().clone()
                h.require(not torch.is_inference(result), "inference tensor escaped clone boundary")
                return result
            if isinstance(value, dict): return {k: ordinary(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)): return type(value)(ordinary(v) for v in value)
            return value
        state_hash = nf.state_hash(state, torch)
        history = ordinary(dict(cond_frame_outputs={0: history0["cond_frame_outputs"][0]},
            non_cond_frame_outputs={i: history0["non_cond_frame_outputs"][i] for i in range(1, 7)}))
        vision, pos, reference = ordinary(vision0), ordinary(pos0), ordinary(baseline_low)
        raw_reference = ordinary(baseline_decoder["values"])
        cache_hash = nf.state_hash((history, vision, pos), torch)
        stored = history["non_cond_frame_outputs"][6]["maskmem_features"]
        h.require(stored.shape == (1, 64, 64, 64) and stored.dtype == torch.bfloat16, "memory shape/dtype")
        base_memory = stored.float().clone()
        target_low = F.interpolate(torch.from_numpy(target.copy()).float()[None, None], (256, 256), mode="nearest")

        def forward(memory, label):
            selected = dict(cond_frame_outputs=history["cond_frame_outputs"], non_cond_frame_outputs=dict(history["non_cond_frame_outputs"]))
            selected["non_cond_frame_outputs"][6] = dict(history["non_cond_frame_outputs"][6], maskmem_features=memory.to(torch.bfloat16))
            traces, captured = [], {}
            carrier = model.memory_attention.register_forward_pre_hook(nf.carrier_hook(model, selected, traces, torch), with_kwargs=True)
            decoder = model.sam_mask_decoder.register_forward_hook(lambda m, a, values: captured.update(values=values))
            try:
                _, sam, _, _ = model._track_step(7, False, vision, pos, sizes, None, None, selected, 9, False, None)
            finally:
                carrier.remove(); decoder.remove()
            h.require(len(traces) == 1 and traces[0]["spatial_frames"] == list(range(7)) and
                      traces[0]["pointer_frames"] == [0, 6, 5, 4, 3, 2, 1] and traces[0]["num_obj_ptr_tokens"] == 28, "full native carrier context")
            raw, quality, _, presence = captured["values"]
            index = int(quality[0].argmax().item())
            raw_selected = raw[:, index:index+1]
            h.require(all(torch.isfinite(x).all().item() for x in (*sam, raw, presence)), "nonfinite native output")
            mask_loss = F.binary_cross_entropy_with_logits(raw_selected, target_low)
            presence_loss = F.binary_cross_entropy_with_logits(presence, torch.ones_like(presence))
            h.require(all(torch.isfinite(x).item() for x in (mask_loss, presence_loss, mask_loss+presence_loss)), "nonfinite surrogate loss")
            numerical = []
            for actual, expected in zip((raw, quality, presence), (raw_reference[0], raw_reference[1], raw_reference[3])):
                difference = (actual.detach().double() - expected.double()).abs()
                numerical.append(dict(max_abs=float(difference.max()), max_rel=float(
                    (difference / expected.double().abs().clamp_min(torch.finfo(torch.float32).tiny)).max())))
            record["calls"].append(dict(label=label, grad_enabled=torch.is_grad_enabled(), selected_index=index,
                native_present=bool((presence > 0).item()), presence_logit=float(presence.detach().item()),
                mask_bce=float(mask_loss.detach()), presence_bce=float(presence_loss.detach()),
                native_selected_sha256=nf.tensor_hash(sam[3], torch), trace=traces[0],
                raw_quality_presence_parity=numerical))
            compare = torch.equal if label == "zero_no_grad" else lambda a, b: torch.allclose(a, b, atol=ATOL, rtol=RTOL)
            if label != "post_step":
                h.require(all(compare(a.detach(), b) for a, b in zip(
                    (raw, quality, presence), (raw_reference[0], raw_reference[1], raw_reference[3]))), "raw decoder forward parity")
            return mask_loss + presence_loss, sam[3]

        with timed("exact_zero_residual_native_replay"), torch.no_grad(), q.capture_warnings(record):
            zero_memory = base_memory + torch.zeros_like(base_memory)
            h.require(torch.equal(zero_memory.to(torch.bfloat16), stored), "BF16 zero roundtrip")
            zero_loss, zero_low = forward(zero_memory, "zero_no_grad")
            h.require(torch.equal(zero_low, reference), "native zero-residual exact parity")
            record["checks"]["zero_residual_native_exact"] = True
        with timed("leaf_forward"), q.capture_warnings(record):
            leaf = base_memory.clone().requires_grad_(True)
            loss, grad_low = forward(leaf, "leaf_grad")
            delta = (grad_low.detach().double()-reference.double()).abs()
            record["grad_forward_parity"] = dict(max_abs=float(delta.max()),
                max_rel=float((delta / reference.double().abs().clamp_min(torch.finfo(torch.float32).tiny)).max()),
                binary_differences=int(((grad_low.detach() > 0) != (reference > 0)).sum()))
            h.require(torch.allclose(grad_low.detach(), reference, atol=ATOL, rtol=RTOL), "grad forward tolerance")
        with timed("leaf_backward"), q.capture_warnings(record):
            loss.backward()
            h.require(leaf.grad is not None and torch.isfinite(leaf.grad).all().item() and leaf.grad.abs().max().item() > 0, "leaf gradient")
            record["leaf_gradient"] = dict(l2=float(leaf.grad.double().norm()), max_abs=float(leaf.grad.abs().max()), nonzero=int(torch.count_nonzero(leaf.grad)))
            h.require(all(p.grad is None for p in model.parameters()), "frozen model accumulated gradient")
            del loss, grad_low, leaf
        with timed("tiny_head_forward"), q.capture_warnings(record):
            head = torch.nn.Conv2d(64, 64, 1)
            torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
            optimizer = torch.optim.SGD(head.parameters(), lr=.001)
            head_before = nf.state_hash(head.state_dict(), torch)
            residual = .01 * torch.tanh(head(base_memory))
            h.require(torch.count_nonzero(residual).item() == 0, "head initialization")
            head_loss, head_low = forward(base_memory + residual, "head_grad")
            h.require(torch.allclose(head_low.detach(), reference, atol=ATOL, rtol=RTOL), "head zero forward parity")
        with timed("tiny_head_backward_and_step"), q.capture_warnings(record):
            head_loss.backward()
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            h.require(grads and all(torch.isfinite(g).all().item() for g in grads) and any(g.abs().max().item() > 0 for g in grads), "head overall gradient")
            record["head_gradient"] = [dict(l2=float(g.double().norm()), nonzero=int(torch.count_nonzero(g))) for g in grads]
            optimizer.step()
            h.require(all(torch.isfinite(p).all().item() for p in head.parameters()) and
                      nf.state_hash(head.state_dict(), torch) != head_before, "optimizer did not update finite head")
            with torch.no_grad():
                updated = .01 * torch.tanh(head(base_memory))
                record["head_step"] = dict(parameters=sum(p.numel() for p in head.parameters()), steps=1,
                    before_sha256=head_before, after_sha256=nf.state_hash(head.state_dict(), torch),
                    coefficient_change_l2=float(sum(p.detach().double().square().sum() for p in head.parameters()).sqrt()),
                    residual_l2=float(updated.norm()), residual_max_abs=float(updated.abs().max()),
                    bf16_changed_entries=int(torch.count_nonzero((base_memory+updated).to(torch.bfloat16) != stored)),
                    no_post_update_accuracy_claim=True)
            del head_loss, head_low, residual
        with timed("post_step_forward"), torch.no_grad(), q.capture_warnings(record):
            post_loss, post_low = forward(base_memory+updated, "post_step")
            record["post_step"] = dict(native_selected_max_abs_change=float((post_low-reference).abs().max()),
                native_binary_differences=int(((post_low > 0) != (reference > 0)).sum()),
                surrogate_loss_change=float(post_loss-zero_loss), improvement_required=False,
                pre_presence_bank_max_abs_change=record["calls"][-1]["raw_quality_presence_parity"][0]["max_abs"])
        with timed("unchanged_state_source_runtime_checks"):
            encoder_hook.remove()
            h.require(record["baseline_frames"] == list(range(8)) and len(record["image_encoder_calls"]) == 8
                      and len(record["calls"]) == 4, "unexpected encoder/replay invocation")
            h.require(all(not m.training for m in model.modules()) and all(not p.requires_grad and p.grad is None for p in model.parameters()), "frozen eval state")
            before = dict(model=model_hash, native_state=state_hash, cache_history=cache_hash)
            after = dict(model=nf.state_hash(model.state_dict(), torch), native_state=nf.state_hash(state, torch),
                         cache_history=nf.state_hash((history, vision, pos), torch))
            h.require(before == after, "model weights/buffers or cache/history/pointers changed")
            h.require(installed["installed_inventory"] == h.inventory(installation), "runtime files changed")
            h.require(all(h.sha(root / rel) == sha for rel, sha in record["source_sha256"].items()) and
                      h.sha(Path(__file__)) == args.source_sha256, "source changed")
            old.unchanged(h, record["native_binding"]["file_sha256"])
            h.require(not record["offline_attempts"], "offline operation attempted")
            record["checks"].update(model_unchanged=True, cache_history_pointer_unchanged=True,
                source_runtime_unchanged=True, finite_nonzero_leaf_and_head_gradient=True, optimizer_step_completed=True)
            record["before_after_hashes"] = dict(before=before, after=after)
        record.update(status="completed", eligible=True, qualification_scope="one fixed future context, no training or segmentation gain")
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
        print(json.dumps({k: record[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo-root", "expected-commit", "native-root", "native-commit", "source-sha256", "output-dir"):
        parser.add_argument("--" + key, required=True)
    run(parser.parse_args())
