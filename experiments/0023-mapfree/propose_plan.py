"""Create a PRIVATE, unadopted Map-free graph proposal from ZIP metadata only.

No HTTP implementation, image decoding, calibration parsing or pose access is
used here. This proposal is not a scientific allocation or acquisition command.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

SPEC = importlib.util.spec_from_file_location("evr0023_range_planner", Path(__file__).with_name("range_inputs.py"))
ranges = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ranges
SPEC.loader.exec_module(ranges)

SALT = "evr-mapfree-development-draft-v1\0"
EXPECTED_SCENES = ("s00513", "s00475", "s00478")
N_SEQ1 = 16
REFERENCE_NEIGHBORS = 3
REPO = Path(__file__).resolve().parents[2]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def uniform_ranks(count):
    if type(count) is not int or count < N_SEQ1:
        raise ValueError("a candidate scene needs at least 16 seq1 members; no replacement")
    # floor(k*(N-1)/15 + 1/2), exactly in integer arithmetic.
    ranks = [(2 * k * (count - 1) + 15) // 30 for k in range(N_SEQ1)]
    if len(set(ranks)) != N_SEQ1:
        raise ValueError("uniform selection produced repeated ranks")
    return ranks


def nearest_references(query, references):
    return sorted(references, key=lambda row: (
        abs(row["frame_number"] - query["frame_number"]), row["frame_number"], row["filename"]
    ))[:REFERENCE_NEIGHBORS]


def member_summary(index, name):
    entry = index.entries[name]
    return {k: entry[k] for k in ranges.CORE_FIELDS}


def byte_budget(entries):
    """Maximum local header extra-field length is 65535; no payload is opened."""
    packed = sum(e["compressed_bytes"] for e in entries)
    header = sum(30 + len(e["filename"].encode("ascii")) + 65535 for e in entries)
    requests = 3 * len(entries)  # local header, name/extra, deflated member
    return dict(members=len(entries), compressed_bytes=packed,
                uncompressed_bytes=sum(e["uncompressed_bytes"] for e in entries),
                local_header_and_name_upper_bound=header,
                range_requests=len(entries) * 3,
                response_extra_byte_probe_reserve=requests,
                response_body_upper_bound=packed + header + requests,
                maximum_member_uncompressed_bytes=max(e["uncompressed_bytes"] for e in entries))


def build_plan(index, prior_draft_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", prior_draft_sha256):
        raise ValueError("prior draft fingerprint required")
    by_scene = {}
    for name in index.entries:
        match = re.fullmatch(r"val/(s[0-9]{5})/seq1/frame_([0-9]{5})\.jpg", name)
        if match:
            by_scene.setdefault(match[1], []).append((int(match[2]), name))
    scenes = sorted(by_scene, key=lambda scene: (digest(SALT + scene), scene))[:3]
    if tuple(scenes) != EXPECTED_SCENES:
        raise ValueError("candidate scene selection differs from the preserved resource draft")
    rows, input_names, future_pose_names = [], [], []
    for scene in scenes:
        chronological = sorted(by_scene[scene])
        selected = []
        for j, rank in enumerate(uniform_ranks(len(chronological)), start=1):
            frame, name = chronological[rank]
            selected.append(dict(filename=name, frame_number=frame, chronological_rank_zero_based=rank,
                                 selected_index_one_based=j, role="B_query" if j % 2 else "C_reference"))
        a = "val/%s/seq0/frame_00000.jpg" % scene
        k = "val/%s/intrinsics.txt" % scene
        pose = "val/%s/poses.txt" % scene
        if any(name not in index.entries for name in (a, k, pose)):
            raise ValueError("candidate scene lacks anchor/intrinsics/future pose metadata; no replacement")
        b = [row for row in selected if row["role"] == "B_query"]
        c = [row for row in selected if row["role"] == "C_reference"]
        reference_edges = [dict(source=a, target=row["filename"], role="A_to_C") for row in c]
        for query in b:
            for reference in nearest_references(query, c):
                reference_edges.append(dict(source=query["filename"], target=reference["filename"],
                                            role="B_to_C", frame_number_distance=abs(
                                                query["frame_number"] - reference["frame_number"])))
        query_edges = [dict(source=a, target=row["filename"], role="A_to_B") for row in b]
        image_names = [a] + [row["filename"] for row in selected]
        frontend = image_names + [k]
        for name in frontend:
            ranges.frontend_kind(name)
        rows.append(dict(scene=scene, candidate_only=True, available_seq1_members=len(chronological),
                         anchor=a, selected_seq1_chronological=selected,
                         query_images=[row["filename"] for row in b],
                         reference_images=[row["filename"] for row in c],
                         image_members=[member_summary(index, name) for name in image_names],
                         intrinsics_member=member_summary(index, k),
                         frontend_budget=byte_budget([index.entries[name] for name in frontend]),
                         reference_edges=reference_edges, query_edges=query_edges))
        input_names.extend(frontend)
        future_pose_names.append(pose)
    images = [name for name in input_names if name.endswith(".jpg")]
    return dict(
        schema="evr0023.proposed-plan.v1", status="candidate_only", allocated=False,
        scientific_member_payloads_read=0, http_requests=0,
        source=dict(url=index.url, archive_bytes=index.archive_bytes, etag=index.etag,
                    cached_index_sha256=index.fingerprints),
        prior_resource_draft_sha256=prior_draft_sha256,
        prospective_change=dict(
            original_draft_preserved=True,
            unchanged="Original scene salt and three candidate places.",
            previous_seq1_selection="16 smallest SHA256(salt + full_filename), hash order.",
            new_seq1_selection="16 uniform chronological member ranks; round-half-up(k*(N-1)/15), k=0..15.",
            rationale="Prospective connected multiview coverage before any image, K, pose or estimator outcome.",
            quality_or_overlap_guaranteed=False),
        selection=dict(scene_salt=SALT, scene_sort="(SHA256(salt + scene_id), scene_id)",
                       candidate_scene_ids=scenes, seq1_sort="(actual frame number, filename)",
                       role_rule="1-based odd selected positions B; even selected positions C.",
                       reference_rule="A->each C; each B->3 C nearest by absolute actual frame-number difference.",
                       distance_ties="Lower actual C frame number, then filename.",
                       query_rule="A->each B; query edges are excluded from the reference edge set."),
        scenes=rows,
        frontend_acquisition=dict(member_names=input_names, image_names=images,
                                  intrinsics_names=[name for name in input_names if name.endswith("intrinsics.txt")],
                                  image_compressed_bytes=sum(index.entries[name]["compressed_bytes"] for name in images),
                                  **byte_budget([index.entries[name] for name in input_names])),
        future_evaluation_only=dict(pose_member_names=future_pose_names,
                                    acquisition_authorized=False, read_authorized=False,
                                    prerequisite="All scientific predictions sealed before any pose download or read.",
                                    included_in_frontend_acquisition=False),
        totals=dict(places=3, images=51, query_images=24, reference_images=24, anchor_images=3,
                    reference_edges=96, query_edges=24, all_edges=120),
        interpretation="Unadopted development proposal only. Graph adjacency does not establish image overlap, parallax, motion diversity or method efficacy. If adopted, the three places become development forever; other 62 cached scene IDs remain unallocated.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members-index", required=True)
    parser.add_argument("--central-directory", required=True)
    parser.add_argument("--index-receipt", required=True)
    for kind in ("members", "directory", "receipt"):
        parser.add_argument("--" + kind + "-sha256", required=True)
    parser.add_argument("--prior-draft", required=True)
    parser.add_argument("--prior-draft-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("proposal output must stay in private work outside the public repository")
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError("fresh private proposal output required")
    prior_bytes = Path(args.prior_draft).read_bytes()
    if ranges.sha(prior_bytes) != args.prior_draft_sha256:
        raise ValueError("preserved prior draft fingerprint mismatch")
    index = ranges.load_index(args.members_index, args.central_directory, args.index_receipt,
                              {kind: getattr(args, kind + "_sha256") for kind in ("members", "directory", "receipt")})
    plan = build_plan(index, args.prior_draft_sha256)
    plan["planning_source_sha256"] = {"propose_plan.py": ranges.sha(Path(__file__).read_bytes()),
                                     "range_inputs.py": ranges.sha(Path(__file__).with_name("range_inputs.py").read_bytes())}
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": plan["status"], "totals": plan["totals"],
                      "frontend_budget": {k: plan["frontend_acquisition"][k] for k in (
                          "members", "compressed_bytes", "response_body_upper_bound")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
