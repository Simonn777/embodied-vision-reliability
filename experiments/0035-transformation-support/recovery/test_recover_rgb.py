"""Temporary opaque bytes only: no real receipts, images, network or model."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zlib

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


r = module("_test_recover", HERE / "recover_rgb.py")
a = module("_test_recover_admit", HERE.parent / "admit.py")


class Fixture:
    def __init__(self, root):
        self.root, self.parent = root, root / "failed"
        self.parent.mkdir()
        self.sources = {"inert.py": "a" * 64}
        self.index = types.SimpleNamespace(entries={}, fingerprints={"manifest": "f" * 64},
                                           etag='"fake"', archive_bytes=10000)
        self.rows, requests = [], []
        for i in range(3):
            raw = bytes([65+i])
            name = f"fake/{i}.jpg"
            entry = dict(filename=name, phase="rgb", sequence="fake", frame_index=i, uncompressed_bytes=1,
                compressed_bytes=1, crc32_hex=f"{zlib.crc32(raw):08x}", local_header_offset=100*i,
                response_body_bytes_cap=100)
            self.index.entries[name] = entry
            if i == 2:
                continue
            path = self.parent / "members" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            row = dict(entry, bytes=1, sha256=r.sha(raw), saved_relative="members/"+name,
                data_offset=100*i+40, compressed_sha256="b"*64, local_header_sha256="c"*64,
                local_extra_bytes=0, local_version_needed=20,
                data_descriptor_read=False, data_descriptor_integrity_claimed=False)
            self.rows.append(row)
            for start, end in ((100*i, 100*i+29), (100*i+30, 100*i+39), (100*i+40, 100*i+40)):
                requests.append(dict(start=start, end=end, expected_bytes=end-start+1,
                    body_bytes=end-start+1, status=206, etag=self.index.etag,
                    content_range=f"bytes {start}-{end}/10000", content_length=str(end-start+1),
                    body_sha256="b"*64))
        self.spec = dict(repo_root=str(root/"repo"), metadata_root=str(root),
                         initial_dir=str(root/"initial"), initial_receipt={"sha256":"d"*64})
        self.desc = {}
        inspection = self.write("inspection/receipt.json", {"status":"completed","eligible":True})
        self.desc["initial_inspection"] = inspection
        self.desc["initial_inspection_outer"] = self.outer("inspection-outer", inspection["sha256"], "completed")
        self.record = dict(schema="evr0035.vost-acquisition.v1", status="failed", phase="rgb",
            expected_commit=r.OLD_COMMIT, source_sha256=self.sources, index_sha256=self.index.fingerprints,
            media_decoded=False, gt_admission=None, retry_count=0, hidden_retries=False,
            full_archive_fallback=False, error_type="URLError", members=self.rows, requests=requests,
            current_member="fake/2.jpg", response_body_bytes=sum(x["body_bytes"] for x in requests),
            requested_body_upper_bound=sum(x["expected_bytes"]+1 for x in requests),
            response_body_bytes_exact=True, initial_admission=dict(receipt_sha256="d"*64,
                inspection_sha256=inspection["sha256"],
                inspection_outer_sha256=self.desc["initial_inspection_outer"]["sha256"]))
        self.desc["receipt"] = self.write("failed/receipt.json", self.record)
        command = ["/fake/python", "-B", "-S", "-W", "error", str(root/"repo"/(r.BASE+"acquire.py")),
            "--repo-root", str(root/"repo"), "--expected-commit", r.OLD_COMMIT,
            "--metadata-root", str(root), "--manifest-relative", r.BASE+"proposed-manifest-v2.json",
            "--admission-relative", r.BASE+"acquisition-admission.json",
            "--output-dir", str(self.parent), "--phase", "rgb",
            "--initial-dir", self.spec["initial_dir"], "--initial-inspection", inspection["path"],
            "--initial-inspection-outer", self.desc["initial_inspection_outer"]["path"]]
        operation = dict(command=command, repo=self.spec["repo_root"], commit=r.OLD_COMMIT, timeout=1800,
                         result_file=self.desc["receipt"]["path"], operation_output=str(root/"failed-outer"))
        self.desc["operation"] = self.write("operation.json", operation)
        normalized = [v.replace(command[0], "$PYTHON").replace(self.spec["repo_root"], "$REPOSITORY")
                      .replace(str(root), "$PRIVATE") for v in command]
        self.desc["outer"] = self.outer("failed-outer", self.desc["receipt"]["sha256"], "failed",
                                       spec_sha256=self.desc["operation"]["sha256"], command=normalized)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value)+"\n").encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=r.sha(raw))

    def outer(self, name, output_sha, status, **extra):
        root = self.root/name
        root.mkdir(exist_ok=True)
        for log in ("stdout.log","stderr.log"):
            (root/log).write_bytes(b"fake\n")
        value = dict(schema="evr0029.exit-checked-operation.v1", status=status,
            exit_code=int(status=="failed"), termination=None, remaining_descendant_pids=[], cleanup_errors=[],
            supervisor_sha256=a.SUPERVISOR_SHA256, monitor_sha256=a.MONITOR_SHA256,
            thread_environment={k:"4" for k in a.THREAD_KEYS}, hard_timeout_seconds=1800,
            group_rss_limit_kib=8388608, expected_commit=r.OLD_COMMIT, wall_seconds=.1,
            peak_group_rss_kib=1000, output_sha256=output_sha,
            log_sha256={k:r.sha((root/k).read_bytes()) for k in ("stdout.log","stderr.log")}, **extra)
        return self.write(name+"/receipt.json", value)

    def verify(self):
        return r.check_parent(self.desc, self.index, self.sources, self.spec, a, {})


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.f = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_parent_metadata_and_all_raw_bytes_verified(self):
        result = self.f.verify()
        self.assertEqual(len(result["members"]), 2)
        self.assertEqual(result["status"], "failed")

    def test_corrupted_reused_sha_and_crc_rejected(self):
        row = self.f.rows[0]
        path = self.f.parent/row["saved_relative"]
        path.write_bytes(b"X")
        with self.assertRaisesRegex(ValueError, "SHA/CRC/size"):
            self.f.verify()
        path.write_bytes(b"A")
        with self.assertRaisesRegex(ValueError, "SHA/CRC/size"):
            r.raw_member(self.f.parent, dict(row, crc32_hex="00000000"))

    def test_overlap_requires_agreement_and_earliest_parent(self):
        parent2 = copy.deepcopy(self.f.record)
        selected, overlap = r.select_reuse([(self.f.desc,self.f.record),(self.f.desc,parent2)])
        self.assertEqual(overlap,2)
        self.assertTrue(all(v[0]==0 for v in selected.values()))
        parent2["members"][0]["sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"conflicting"):
            r.select_reuse([(self.f.desc,self.f.record),(self.f.desc,parent2)])

    def test_successful_parent_and_wrong_outer_binding_rejected(self):
        self.f.record["status"]="completed"
        self.f.desc["receipt"]=self.f.write("failed/receipt.json",self.f.record)
        with self.assertRaisesRegex(ValueError,"failed RGB"):
            self.f.verify()
        self.f.record["status"]="failed"
        self.f.desc["receipt"]=self.f.write("failed/receipt.json",self.f.record)
        outer=json.loads(Path(self.f.desc["outer"]["path"]).read_text())
        outer["output_sha256"]="0"*64
        self.f.desc["outer"]=self.f.write("failed-outer/receipt.json",outer)
        with self.assertRaisesRegex(ValueError,"outer provenance"):
            self.f.verify()

    def test_wrong_missing_population_fails_before_reader_or_network(self):
        helper=types.SimpleNamespace(RangeReader=Mock())
        with self.assertRaisesRegex(ValueError,"coverage/budget"):
            r.materialize(None,helper,{},self.f.index,{},self.root,{},lambda:None,0)
        helper.RangeReader.assert_not_called()

    def test_complete_extension_rejects_wrong_mode_and_missing_inventory(self):
        record=dict(schema="evr0035.vost-acquisition.v1",transport_mode="network_only",phase="rgb",
            status="completed",reused_members=408,fetched_members=38,members=[{}]*446,
            parent_members_verified=[408,308],overlap_verified=308,retry_count=0,
            media_decoded=False,parent_rechecks_complete=True)
        with self.assertRaisesRegex(ValueError,"recovery extension"):
            r.verify_recovery_receipt(record,self.f.index,self.root)
        record["transport_mode"]="verified_complete_recovery"
        record["members"]=[dict(filename=f"fake/{i}.jpg") for i in range(446)]
        with self.assertRaisesRegex(ValueError,"ordered recovered inventory"):
            r.verify_recovery_receipt(record,self.f.index,self.root)

    def test_original_admission_accepts_explicit_recovery_extension(self):
        repo=self.root/"repo"
        repo.mkdir()
        (repo/"inert.py").write_bytes(b"# artificial\n")
        record=dict(self.f.record,status="completed",transport_mode="verified_complete_recovery",
                    parent_failures=[self.f.desc],
                    source_sha256={"inert.py":a.sha(repo/"inert.py")})
        desc=self.f.write("failed/receipt.json",record)
        outer=self.f.outer("new-complete-outer",desc["sha256"],"completed")
        manifest={"members":list(self.f.index.entries.values())[:2]}
        with patch.object(a,"MANIFEST_SHA256",self.f.index.fingerprints["manifest"]):
            saved,files=a.acquisition(desc,outer,"rgb",manifest,{},repo)
        self.assertEqual(saved["transport_mode"],"verified_complete_recovery")
        self.assertEqual(len(files),2)

    def test_fresh_output_and_preflight_failure_retained(self):
        output=self.root/"recovery"
        spec=dict(schema="evr0035.rgb-recovery-spec.v1", output=str(output),
                  repo_root=str(self.root/"repo"), recovery_source_commit="a"*40,parents=[])
        p=self.root/"recovery-spec.json"
        p.write_text(json.dumps(spec))
        with patch.object(r,"source_binding",side_effect=ValueError("manufactured source rejection")) as mocked:
            with self.assertRaisesRegex(ValueError,"manufactured source"):
                r.run(p)
            receipt=json.loads((output/"receipt.json").read_text())
            self.assertEqual(receipt["status"],"failed")
            self.assertEqual(receipt["requests"],[])
            with self.assertRaisesRegex(ValueError,"fresh private"):
                r.run(p)
            self.assertEqual(mocked.call_count,1)


if __name__ == "__main__":
    unittest.main()
