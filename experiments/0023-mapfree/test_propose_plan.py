"""Artificial metadata tests; no scientific images/K/poses, HTTP or estimators."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location("evr0023_plan_test", Path(__file__).with_name("propose_plan.py"))
plan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plan
spec.loader.exec_module(plan)


def artificial_index():
    entries = {}
    for scene in ("s00513", "s00475", "s00478"):
        names = ["val/%s/seq1/frame_%05d.jpg" % (scene, i * 3) for i in range(31)]
        names += ["val/%s/%s" % (scene, n) for n in (
            "seq0/frame_00000.jpg", "intrinsics.txt", "poses.txt")]
        for name in names:
            entries[name] = dict(filename=name, flags=0, compression_method=8, crc32_hex="00000000",
                                 compressed_bytes=13, uncompressed_bytes=29, local_header_offset=0)
    return plan.ranges.Index(plan.ranges.URL, '"artificial"', 1000000, 900000, entries,
                             {k: "0" * 64 for k in ("members", "directory", "receipt")})


class PlanTests(unittest.TestCase):
    def test_round_half_up_all_distinct_and_chronological(self):
        self.assertEqual(plan.uniform_ranks(16), list(range(16)))
        self.assertEqual(plan.uniform_ranks(31), list(range(0, 31, 2)))
        self.assertEqual(plan.uniform_ranks(17), [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16])
        for n in (16, 17, 31, 243, 581):
            r = plan.uniform_ranks(n)
            self.assertEqual((r[0], r[-1], len(set(r))), (0, n - 1, 16))
        with self.assertRaises(ValueError):
            plan.uniform_ranks(15)

    def test_graph_roles_pair_disjointness_and_fixed_counts(self):
        result = plan.build_plan(artificial_index(), "1" * 64)
        self.assertFalse(result["allocated"])
        self.assertEqual(result["selection"]["candidate_scene_ids"], list(plan.EXPECTED_SCENES))
        for row in result["scenes"]:
            self.assertEqual(len(row["image_members"]), 17)
            self.assertEqual(len(row["reference_edges"]), 32)
            self.assertEqual(len(row["query_edges"]), 8)
            self.assertEqual([r["frame_number"] for r in row["selected_seq1_chronological"]], list(range(0, 91, 6)))
            for item in row["selected_seq1_chronological"]:
                self.assertEqual(item["role"], "B_query" if item["selected_index_one_based"] % 2 else "C_reference")
            reference = {(e["source"], e["target"]) for e in row["reference_edges"]}
            query = {(e["source"], e["target"]) for e in row["query_edges"]}
            self.assertEqual(len(reference), 32)
            self.assertFalse(reference & query)
            for b in row["query_images"]:
                self.assertEqual(sum(e["source"] == b for e in row["reference_edges"]), 3)

    def test_actual_frame_distance_and_lower_frame_ties(self):
        query = dict(frame_number=5, filename="b")
        refs = [dict(frame_number=n, filename=str(n)) for n in (30, 6, 4, 2, 0)]
        self.assertEqual([r["frame_number"] for r in plan.nearest_references(query, refs)], [4, 6, 2])

    def test_no_gt_in_frontend_and_range_byte_bound(self):
        result = plan.build_plan(artificial_index(), "1" * 64)
        frontend = result["frontend_acquisition"]
        self.assertEqual((len(frontend["image_names"]), len(frontend["intrinsics_names"])), (51, 3))
        self.assertEqual(frontend["members"], 54)
        self.assertEqual(frontend["image_compressed_bytes"], 51 * 13)
        self.assertEqual(frontend["compressed_bytes"], 54 * 13)
        expected = sum(13 + 30 + len(name) + 65535 + 3 for name in frontend["member_names"])
        self.assertEqual(frontend["response_body_upper_bound"], expected)
        future = result["future_evaluation_only"]
        self.assertEqual(len(future["pose_member_names"]), 3)
        self.assertFalse(set(future["pose_member_names"]) & set(frontend["member_names"]))
        self.assertFalse(future["acquisition_authorized"])
        self.assertFalse(future["read_authorized"])
        json.dumps(result, allow_nan=False)

    def test_directory_order_does_not_change_proposal(self):
        index = artificial_index()
        first = plan.build_plan(index, "1" * 64)
        reversed_index = plan.ranges.Index(index.url, index.etag, index.archive_bytes,
                                           index.directory_offset, dict(reversed(list(index.entries.items()))), index.fingerprints)
        self.assertEqual(first, plan.build_plan(reversed_index, "1" * 64))

    def test_missing_inputs_or_changed_scene_set_fail_without_replacement(self):
        index = artificial_index()
        del index.entries["val/s00513/intrinsics.txt"]
        with self.assertRaisesRegex(ValueError, "no replacement"):
            plan.build_plan(index, "1" * 64)
        index = artificial_index()
        index.entries.clear()
        with self.assertRaisesRegex(ValueError, "scene selection differs"):
            plan.build_plan(index, "1" * 64)


if __name__ == "__main__":
    unittest.main()
