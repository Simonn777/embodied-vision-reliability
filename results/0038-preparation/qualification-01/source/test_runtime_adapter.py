"""Manufactured adapter tests only. No SAM import, loading, files, RGB or labels."""
import types
import unittest

import runtime_adapter as a
import operators


def fixture(torch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.real_calls = 0

        def forward(self, *, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens):
            self.real_calls += 1
            # Invented nonlinear complete stack, not an implementation of SAM.
            return torch.tanh(curr[0] + (memory + memory_pos).mean(0).unsqueeze(0))

    class Model:
        training = False
        hidden_dim = 4

        def __init__(self):
            self.memory_attention = Attention()
            self.head_calls = 0

        def _prepare_memory_conditioned_features(self, *, frame_idx,
                is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                feat_sizes, output_dict, num_frames, track_in_reverse):
            rows = [output_dict["cond_frame_outputs"][0],
                    output_dict["non_cond_frame_outputs"][frame_idx - 1]]
            spatial = [r["maskmem_features"].flatten(2).permute(2, 0, 1).float() for r in rows]
            pointers = [r["obj_ptr"].unsqueeze(0) for r in rows]
            memory = torch.cat(spatial + pointers)
            out = self.memory_attention(curr=current_vision_feats,
                curr_pos=current_vision_pos_embeds, memory=memory,
                memory_pos=torch.zeros_like(memory), num_obj_ptr_tokens=2)
            return out.permute(1, 2, 0).reshape(1, 4, 2, 2)

        def _forward_sam_heads(self, *, backbone_features, point_inputs,
                mask_inputs, high_res_features, multimask_output):
            self.head_calls += 1
            low = backbone_features.mean(1, keepdim=True)
            bank = torch.cat([low - 1, low, low + 1], 1)
            quality = torch.tensor([[0.2, 0.8, 0.4]])
            return (bank, bank.clone(), quality, low, low.clone(),
                    torch.ones(1, 4), torch.ones(1, 1))

    def write(value):
        return dict(maskmem_features=torch.full((1, 4, 2, 2), value, dtype=torch.bfloat16),
                    maskmem_pos_enc=[torch.zeros(1, 4, 2, 2)],
                    pred_masks=torch.zeros(1, 1, 2, 2),
                    obj_ptr=torch.full((1, 4), value), object_score_logits=torch.ones(1, 1))

    model = Model()
    history = dict(cond_frame_outputs={0: write(0.)}, non_cond_frame_outputs={1: write(.25)})
    capture = dict(frame=2, num_frames=4, is_init_cond_frame=False,
                   history=history, current_vision_feats=[torch.arange(16).float().reshape(4, 1, 4)/32],
                   current_vision_pos_embeds=[torch.zeros(4, 1, 4)], feat_sizes=[(2, 2)],
                   high_res_features=None)
    kwargs = []
    def hook(_module, _args, kw, out):
        kwargs.append((a.clone_tree(kw, torch), out.clone()))
    handle = model.memory_attention.register_forward_hook(hook, with_kwargs=True)
    pix = model._prepare_memory_conditioned_features(frame_idx=2,
        is_init_cond_frame=False, current_vision_feats=capture["current_vision_feats"],
        current_vision_pos_embeds=capture["current_vision_pos_embeds"],
        feat_sizes=[(2, 2)], output_dict=history, num_frames=4, track_in_reverse=False)
    handle.remove()
    capture.update(attention_context=kwargs[0][0], attention_output=kwargs[0][1], pix_feat=pix,
        sam_outputs=model._forward_sam_heads(backbone_features=pix, point_inputs=None,
            mask_inputs=None, high_res_features=None, multimask_output=True), selected_index=1)
    writes = [write(-.5), a.clone_tree(history["non_cond_frame_outputs"][1], torch), write(.75)]
    for i, w in enumerate(writes):
        w.update(candidate_index=i, is_native_selected=(i == 1))
    model.memory_attention.real_calls = model.head_calls = 0
    return model, capture, writes


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_context_collection_does_not_execute_or_mutate_history(self):
        t = self.torch
        model, capture, writes = fixture(t)
        before = a.clone_tree(capture, t)
        context = a.context_for_write(model, t, capture, writes[0])
        self.assertEqual(model.memory_attention.real_calls, 0)
        self.assertTrue(a.equal_tree(capture, before, t))
        self.assertFalse(t.equal(context["memory"], capture["attention_context"]["memory"]))
        self.assertNotIn("forward", model.memory_attention.__dict__)
        # Failure must restore the original bound method too.
        old = model._prepare_memory_conditioned_features
        def fail(**kwargs):
            raise RuntimeError("manufactured native layout failure")
        model._prepare_memory_conditioned_features = fail
        with self.assertRaisesRegex(RuntimeError, "manufactured"):
            a.context_for_write(model, t, capture, writes[0])
        self.assertNotIn("forward", model.memory_attention.__dict__)
        model._prepare_memory_conditioned_features = old

    def test_full_readout_counts_parity_and_no_alias(self):
        t = self.torch
        model, capture, writes = fixture(t)
        before = a.clone_tree((capture, writes), t)
        result, meta = a.readouts(model, t, capture, writes, operators)
        self.assertEqual(set(result), set(a.ARMS))
        self.assertEqual(meta["calls"], dict(context_materializations=4,
                                           attention_stacks=5, sam_heads=6))
        self.assertEqual(model.memory_attention.real_calls, 5)
        self.assertEqual(model.head_calls, 6)
        self.assertTrue(meta["selected_context_exact"] and meta["selected_stack_exact"]
                        and meta["selected_sam_exact"])
        self.assertTrue(a.equal_tree((capture, writes), before, t))
        result["late_mean"].zero_()
        self.assertTrue(a.equal_tree((capture, writes), before, t))

    def test_selected_parity_rejects_tampered_write_and_reuse_needs_qualification(self):
        t = self.torch
        model, capture, writes = fixture(t)
        with self.assertRaisesRegex(ValueError, "prior exact qualification"):
            a.readouts(model, t, capture, writes, operators, reuse_selected=True)
        writes[1]["obj_ptr"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "complete context differs"):
            a.readouts(model, t, capture, writes, operators)

    def test_identity_fastpath_is_explicit_and_initial_is_noop(self):
        t = self.torch
        model, capture, writes = fixture(t)
        for i in (0, 2):
            writes[i] = a.clone_tree(writes[1], t)
            writes[i].update(candidate_index=i, is_native_selected=False)
        out, meta = a.readouts(model, t, capture, writes, operators,
            selected_branch_qualified=True, identity_fastpath=True)
        self.assertTrue(meta["identity_fastpath_used"])
        self.assertEqual(meta["calls"]["attention_stacks"], 0)
        self.assertTrue(all(t.equal(v, capture["sam_outputs"][3]) for v in out.values()))
        capture["frame"] = 0
        out, meta = a.readouts(model, t, capture, None, operators)
        self.assertEqual(meta["noop_reason"], "initial_or_first_noninitial")
        self.assertEqual(meta["calls"]["sam_heads"], 0)

    def test_candidate_writes_native_reuse_absence_and_bf16(self):
        t = self.torch
        model, capture, writes = fixture(t)
        model.use_multimask_token_for_obj_ptr = model.use_obj_ptrs_in_encoder = True
        model.non_overlap_masks_for_mem_enc = False
        model.pred_obj_scores = model.fixed_no_obj_ptr = True
        model.soft_no_obj_ptr = False
        model.no_obj_ptr = t.full((1, 4), -3.)
        calls = dict(encode=0, project=0)
        def project(token):
            calls["project"] += 1
            return token.square() + 1
        model.obj_ptr_proj = project
        def encode(**kwargs):
            calls["encode"] += 1
            return t.full((1, 4, 2, 2), .12345), [t.zeros(1, 4, 2, 2)]
        model._encode_new_memory = encode
        native = a.clone_tree(writes[1], t)
        native["object_score_logits"] = t.tensor([[-2.]])
        native["obj_ptr"] = model.no_obj_ptr.clone()
        sam = list(capture["sam_outputs"])
        sam[0] = t.full((1, 3, 2, 2), -1024.)
        sam[1] = sam[0].clone()
        sam[3] = native["pred_masks"] = sam[0][:, 1:2].clone()
        sam[5], sam[6] = native["obj_ptr"].clone(), native["object_score_logits"].clone()
        capture.update(frame=1, selected_index=1, sam_outputs=tuple(sam),
                       decoder_tokens=t.arange(12).float().reshape(1, 3, 4))
        before = a.clone_tree((capture, native), t)
        new, meta = a.candidate_writes(model, t, capture, native)
        self.assertEqual(calls, dict(encode=2, project=2))
        self.assertEqual(meta["selected_index"], 1)
        self.assertTrue(all(x["maskmem_features"].dtype == t.bfloat16 for x in new))
        self.assertTrue(all(t.equal(x["obj_ptr"], model.no_obj_ptr) for x in new))
        self.assertTrue(all(a.equal_tree(new[1][k], native[k], t) for k in a.COMPACT))
        self.assertTrue(a.equal_tree((capture, native), before, t))
        # Soft-presence path uses the same native affine mixture, not hard zero.
        model.soft_no_obj_ptr = True
        pointer = a.project_pointer(model, t, t.ones(1, 4), t.zeros(1, 1))
        self.assertTrue(t.equal(pointer, t.full((1, 4), -.5)))

    def test_initial_observer_transparent_tokens_and_restoration(self):
        t = self.torch
        class Decoder(t.nn.Module):
            def forward(self, **kwargs):
                return (t.ones(1, 1, 256, 256), t.ones(1, 1),
                        t.ones(1, 1, 256), t.ones(1, 1))
        class Attention(t.nn.Module):
            def forward(self, **kwargs):
                raise AssertionError("initial mask route must not attend")
        class Model:
            training = False
            def __init__(self):
                self.sam_mask_decoder, self.memory_attention = Decoder(), Attention()
                self.returned = None
            def _track_step(self, *args):
                low, quality, _tokens, presence = self.sam_mask_decoder(dummy=True)
                high = t.ones(1, 1, 1024, 1024)
                sam = (low, high, quality, low, high, t.ones(1, 256), presence)
                self.returned = ({}, sam, None, t.ones(1, 256, 1, 1))
                return self.returned
        model = Model()
        with a.NativeObserver(model, t) as observer:
            args = (0, True, [t.ones(1, 1, 256)], [t.zeros(1, 1, 256)],
                    [(1, 1)], None, t.ones(1, 1, 4, 4),
                    {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                    2, False, None)
            answer = model._track_step(*args)
            self.assertIs(answer, model.returned)
            captured = observer.pop(0)
            self.assertIsNone(captured["attention_context"])
            self.assertEqual(tuple(captured["decoder_tokens"].shape), (1, 1, 256))
            self.assertEqual(observer.pending, ())
            # A side decoder dispatch cannot enter the baseline capture.
            model.sam_mask_decoder(dummy=True)
            self.assertEqual(observer.pending, ())
        self.assertNotIn("_track_step", model.__dict__)
        self.assertFalse(hasattr(model, "_d38_observer"))
        self.assertFalse(model.sam_mask_decoder._forward_hooks)
        self.assertFalse(model.memory_attention._forward_hooks)


if __name__ == "__main__":
    unittest.main()
