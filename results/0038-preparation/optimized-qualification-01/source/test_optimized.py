"""Source-prepared manufactured tests; no SAM/source/data/model loading."""
import types
import unittest
from unittest.mock import patch

import operators
import optimized as o
import runtime_adapter as adapter
from test_runtime_adapter import fixture


def apply_rotary_enc(q, k, *, freqs_cis, repeat_freqs_k):
    """Manufactured nontrivial periodic rotation, not SAM's implementation."""
    import torch
    def rotate(x):
        angle = freqs_cis.repeat((x.shape[-2] + len(freqs_cis) - 1) // len(freqs_cis))[:x.shape[-2]]
        angle = angle.reshape(1, 1, -1, 1)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack([even * angle.cos() - odd * angle.sin(),
                            even * angle.sin() + odd * angle.cos()], -1).flatten(-2)
    return rotate(q), rotate(k)


def contexts(torch):
    result = []
    for delta in (-.4, 0., .6):
        memory = torch.arange(48).float().reshape(12, 1, 4) / 50
        memory[4:8] += delta  # latest of two complete four-token grids
        memory[10:12] += delta  # latest pointer after initial two-token pointer
        result.append(dict(curr=[torch.arange(16).float().reshape(4, 1, 4) / 20],
            curr_pos=[torch.zeros(4, 1, 4)], memory=memory,
            memory_pos=torch.arange(48).float().reshape(12, 1, 4) / 1000,
            num_obj_ptr_tokens=4))
    return result


def tiny_stack(torch):
    class RoPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads, self.rope_k_repeat, self.dropout_p = 1, True, 0.
            self.freqs_cis = torch.arange(1, 5).float() / 10
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                layer = torch.nn.Linear(4, 4, bias=False)
                with torch.no_grad():
                    layer.weight.copy_(torch.eye(4))
                setattr(self, name, layer)
        def _separate_heads(self, x, count):
            return x.unsqueeze(1)
        def _recombine_heads(self, x):
            return x.squeeze(1)
        def forward(self, q, k, v, num_k_exclude_rope=0):
            q = self._separate_heads(self.q_proj(q), 1)
            k = self._separate_heads(self.k_proj(k), 1)
            v = self._separate_heads(self.v_proj(v), 1)
            n = k.shape[-2] - num_k_exclude_rope
            q, k[:, :, :n] = apply_rotary_enc(q, k[:, :, :n],
                freqs_cis=self.freqs_cis, repeat_freqs_k=True)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.)
            return self.out_proj(self._recombine_heads(out))
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn_image = RoPE()
    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])
            self.fail = False
        def forward(self, *, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens):
            value = (curr[0] + curr_pos[0]).transpose(0, 1)
            for index, layer in enumerate(self.layers):
                read = layer.cross_attn_image(q=value,
                    k=(memory + memory_pos).transpose(0, 1), v=memory.transpose(0, 1),
                    num_k_exclude_rope=num_obj_ptr_tokens)
                value = torch.tanh(value + read)  # changing queries in later layers
                if self.fail and index == 1:
                    raise RuntimeError("manufactured stack failure")
            return value.transpose(0, 1)
    return types.SimpleNamespace(training=False, memory_attention=Stack().eval())


class OptimizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_multiplicity_normalization_matches_explicit_flat(self):
        t = self.torch
        cs = contexts(t)
        compressed, bias, info = o.compress_contexts(cs, spatial_grid_tokens=4,
            pointer_width=2, torch=t, operators=operators)
        flat, _ = operators.flatten_contexts(cs, torch=t)
        # Independent float64 explicit softmax oracle. No repeated-common weak control.
        q = cs[0]["curr"][0][:, 0].double()
        def attention(context, extra):
            k = (context["memory"] + context["memory_pos"])[:, 0].double()
            v = context["memory"][:, 0].double()
            scores = q @ k.T / 2
            return t.softmax(scores + extra, -1) @ v
        expected = attention(flat, 0)
        actual = attention(compressed, bias.double())
        t.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        self.assertEqual(info["original_flat_spatial_tokens"], 24)
        self.assertEqual(info["compressed_spatial_tokens"], 16)
        self.assertEqual(info["compressed_pointer_tokens"], 8)
        self.assertEqual(int((bias == 0).sum()), 18)
        # Omitting multiplicity must be detected by this fixture.
        self.assertGreater(float((attention(compressed, 0) - expected).abs().max()), 1e-3)

    def test_common_values_and_complete_grids_are_enforced(self):
        t = self.torch
        cs = contexts(t)
        cs[2]["memory"][0, 0, 0] += .5
        with self.assertRaisesRegex(ValueError, "nonvariable"):
            o.compress_contexts(cs, spatial_grid_tokens=4, pointer_width=2, torch=t, operators=operators)
        with self.assertRaisesRegex(ValueError, "complete initial/latest"):
            o.compress_contexts(contexts(t), spatial_grid_tokens=3, pointer_width=2, torch=t, operators=operators)

    def test_periodic_rope_and_all_four_nonlinear_layers(self):
        t = self.torch
        cs, model = contexts(t), tiny_stack(t)
        flat, _ = operators.flatten_contexts(cs, torch=t)
        compact, bias, _ = o.compress_contexts(cs, spatial_grid_tokens=4,
            pointer_width=2, torch=t, operators=operators)
        expected = model.memory_attention(**flat)
        actual = o.compressed_stack(model, t, compact, bias)
        t.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(all("forward" not in layer.cross_attn_image.__dict__
                            for layer in model.memory_attention.layers))

    def test_compressed_forward_restoration_on_exception(self):
        t = self.torch
        model = tiny_stack(t)
        compact, bias, _ = o.compress_contexts(contexts(t), spatial_grid_tokens=4,
            pointer_width=2, torch=t, operators=operators)
        model.memory_attention.fail = True
        with self.assertRaisesRegex(RuntimeError, "manufactured"):
            o.compressed_stack(model, t, compact, bias)
        self.assertTrue(all("forward" not in layer.cross_attn_image.__dict__
                            for layer in model.memory_attention.layers))

    def test_batched_contexts_preserve_independent_results(self):
        t = self.torch
        cs, model = contexts(t), tiny_stack(t)
        singles = [model.memory_attention(**c) for c in cs]
        batched = model.memory_attention(**o.batch_contexts(cs, t))
        for i, single in enumerate(singles):
            t.testing.assert_close(batched[:, i:i + 1], single, atol=2e-6, rtol=2e-6)

    def test_cow_context_is_equal_and_does_not_mutate_borrowed_inputs(self):
        t = self.torch
        model, capture, writes = fixture(t)
        before = adapter.clone_tree((capture, writes), t)
        expected = adapter.context_for_write(model, t, capture, writes[0])
        actual = o.cow_context(model, t, capture, writes[0])
        self.assertTrue(adapter.equal_tree(actual, expected, t))
        self.assertTrue(adapter.equal_tree((capture, writes), before, t))
        self.assertEqual(model.memory_attention.real_calls, 0)
        self.assertNotIn("forward", model.memory_attention.__dict__)

    def test_full_orchestration_and_all_seven_audit_fields(self):
        t = self.torch
        model, capture, writes = fixture(t)
        model.memory_temporal_stride_for_eval = 1
        model.max_cond_frames_in_attn = -1
        model.use_obj_ptrs_in_encoder = model.only_obj_ptrs_in_the_past_for_eval = True
        model.num_maskmem, model.max_obj_ptrs_in_encoder, model.mem_dim = 7, 16, 4
        def decode(*, backbone_features, **_kwargs):
            low = backbone_features.mean(1, keepdim=True)
            bank = t.cat([low - 1, low, low + 1], 1)
            n = low.shape[0]
            return (bank, bank.clone(), t.tensor([[.2, .8, .4]]).repeat(n, 1),
                    low, low.clone(), t.ones(n, 4), t.ones(n, 1))
        model._forward_sam_heads = decode
        reference, _ = adapter.readouts(model, t, capture, writes, operators)
        before = adapter.clone_tree((capture, writes), t)
        # Fixture's invented stack uses a memory mean. Compression's weighted
        # mean is its exact equivalent; real RoPE/fullstack is tested separately.
        def compact_mean(_model, torch, context, bias):
            weights = bias.exp().reshape(-1, 1, 1)
            mean = ((context["memory"] + context["memory_pos"]) * weights).sum(0) / weights.sum()
            return torch.tanh(context["curr"][0] + mean.unsqueeze(0))
        audit = {}
        with patch.object(o, "compressed_stack", compact_mean):
            result, meta = o.optimized_readouts(model, t, capture, writes, operators, adapter,
                                               audit_tensors=audit)
        for name in o.ARMS:
            t.testing.assert_close(result[name], reference[name], atol=2e-6, rtol=2e-6)
        self.assertEqual(meta["calls"], dict(context_materializations=4,
            attention_stack_calls=2, attention_stack_examples=4, sam_head_calls=1, sam_head_examples=5))
        expected_names = {"slot0", "slot1", "slot2", "early", "flat", "late"}
        self.assertEqual(set(audit["stack_by_name"]), expected_names)
        self.assertEqual(set(audit["sam_by_name"]), expected_names)
        self.assertTrue(all(len(sam) == 7 for sam in audit["sam_by_name"].values()))
        self.assertTrue(adapter.equal_tree((capture, writes), before, t))
        self.assertTrue(t.equal(audit["sam_by_name"]["slot1"][3], capture["sam_outputs"][3]))


if __name__ == "__main__":
    unittest.main()
