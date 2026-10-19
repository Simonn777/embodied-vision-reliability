# 0011 source and cache audit before new predictions

Observed in the original Work executor on 2026-10-19 UTC (October18 New York).
Read-only checks, no model import, inference, array decode, new markers, asset
acquisition or fees. The original 0002 allocation is unchanged.

The five native sources pinned in `scripts/tapnext_native_adapter.py` match their
cached SHA256 values. Core returns two 256-bin coordinate heads, ordered y then
x; native decode is temperature2, argmax +/-20 truncated expectation plus .5.
The wrapper reverses y/x into display x/y. Its native next step consumes the
original queries and recurrent states; coordinate predictions are not explicit
feedback. Existing 0005 saved NPZ outputs contain raw coordinate logits according
to the saved receipt and saving code; those arrays were not reopened here.

Source audit turn `01a151aa-e697-7024-afaf-5a667af58bdd`: commands
`exec-5420e7f9-ac5b-409e-a8fc-c1bf454a5742`,
`exec-149fd927-28dd-4435-bc83-c59f67e7e0a8`,
`exec-855f1644-8b14-49ac-91de-765239dec9be`,
`exec-52517eee-b4cc-4412-abec-549b0c24f81f`,
`exec-4a9d5a7c-a6a0-4b8a-9cb7-4c9d54d11d67` all exited0.
The second source-grep output was truncated in parent retrieval; the third's
five hashes and selected source fragments were not truncated. No claim depends
on unseen text in the truncated output. Revision
730cda1c730877cfedbe01bf87fb1cadb78a565d and source details are in the prior note.

Cache audit turn `01a151ae-832e-7770-92e4-c22a571a42e7`: actual commands
`exec-aa9e0d15-d7a1-49d8-81fd-d01f15314e00` (0ms),
`exec-631b4238-9dd6-4c7b-95b9-5271b4941107` (1462ms), and
`exec-5827bbc3-5415-4bcb-9d98-9c3653d3e749` (0ms) all exited0 with untruncated
retrieved output. Zero-ms values are tool reporting granularity, not measured
zero work. Full turn elapsed67.860s includes orchestration.

Cached `bootstapir-cache/rgb.zip` is187291581 bytes, SHA256
ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc, with stable
before/after stat. The first two reserve entries are g-036/video36 and g-002/video2.
No prepared NPZ for these videos was found; `frame-002.npz` in old capability
outputs refers to a frame of video0, not video2.

Directory inspection covered28411 entries within depth8 of named project cache,
scratch and research-result trees. The initial small-JSON scan used16777206 of
16777216 allowed bytes; thus it was incomplete for later JSON content. A focused
follow-up read57 distinct receipt/launch/marker files (298897 bytes) in the stated
cache/scratch/calibration/prediction paths and found no target group mention.
The30 existing published prediction group directories contain neither group.
This is bounded visible-workspace evidence, not proof of no execution in every
historical environment. No archive members or NPZ headers/arrays were opened.

These observations admit planning a cache-only developmental allocation. They
do not admit inference until the new protocol/source freeze is reviewed and
published, and do not establish scene independence or any quality improvement.
