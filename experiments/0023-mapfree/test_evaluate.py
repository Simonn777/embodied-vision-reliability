"""Synthetic sealed predictions, pose text and ZIP transport only; no real GT."""
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

spec = importlib.util.spec_from_file_location("evr0023_evaluate_test", Path(__file__).with_name("evaluate.py"))
ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
IDENTITY = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def rotation(degrees):
    a = math.radians(degrees); c,s = math.cos(a), math.sin(a)
    return [[1.,0.,0.],[0.,c,-s],[0.,s,c]]


def artificial_index():
    entries = {}
    for scene in ev.SCENES:
        names = ["val/%s/seq1/frame_%05d.jpg" % (scene,i) for i in range(16)]
        names += ["val/%s/%s" % (scene,n) for n in ("seq0/frame_00000.jpg","intrinsics.txt","poses.txt")]
        for name in names:
            entries[name] = dict(filename=name, flags=0, compression_method=8, crc32_hex="00000000",
                                 compressed_bytes=13, uncompressed_bytes=29, local_header_offset=0)
    return ev.ranges.Index(ev.ranges.URL, '"artificial"', 1000000, 900000, entries,
                           {k:"0"*64 for k in ("members","directory","receipt")})


def scene_predictions():
    result = {}
    for scene in ev.SCENES:
        queries = []
        for k in range(1,9):
            good = dict(status="completed", pose=[IDENTITY,[1.,0.,0.]])
            queries.append(dict(scene=scene, query="B"+str(k), native_E=copy.deepcopy(good),
                                 arms={method:copy.deepcopy(good) for method in ev.METHODS[1:]}))
        result[scene] = dict(schema="evr0023.scene-prediction.v1", status="completed", scene=scene,
                             ground_truth_access=False, queries=queries)
    return result


def ground_truth(plan):
    result = {}
    for scene in plan["scenes"]:
        prefix = "val/"+scene["scene"]+"/"
        names = [scene["anchor"][len(prefix):]] + [n[len(prefix):] for n in scene["query_images"]]
        text = "\n".join(name+" 1 0 0 0 "+("0 0 0" if i == 0 else "1 0 0") for i,name in enumerate(names))
        result[scene["scene"]] = ev.parse_poses(text,names)[0]
    return result


class FakeResponse:
    status = 206
    def __init__(self, body, start, end, size, etag):
        self.data = io.BytesIO(body)
        self.headers = {"ETag":etag,"Content-Range":"bytes %d-%d/%d"%(start,end,size),"Content-Length":str(end-start+1)}
    def __enter__(self): return self
    def __exit__(self,*_): self.data.close()
    def read(self,n): return self.data.read(n)


def synthetic_zip(plan):
    buffer=io.BytesIO(); content={}
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        for scene in plan["scenes"]:
            prefix="val/"+scene["scene"]+"/"
            names=[scene["anchor"][len(prefix):]]+[n[len(prefix):] for n in scene["query_images"]]
            data=("\n".join(n+" 1 0 0 0 "+("0 0 0" if i==0 else "1 0 0") for i,n in enumerate(names))+"\n").encode()
            content[prefix+"poses.txt"]=data; z.writestr(prefix+"poses.txt",data)
    blob=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(blob)) as z: offset=z.start_dir
    entries=ev.ranges.parse_directory(blob[offset:-22],len(blob),offset)
    index=ev.ranges.Index(ev.ranges.URL,'"artificial"',len(blob),offset,entries,{k:"0"*64 for k in ("members","directory","receipt")})
    calls=[]
    def transport(request,timeout):
        assert request.full_url==ev.ranges.URL and request.get_header("If-match")==index.etag
        assert 0 < timeout <= 30
        start,end=map(int,request.get_header("Range")[6:].split("-")); calls.append((start,end))
        return FakeResponse(blob[start:end+1],start,end,len(blob),index.etag)
    return index,content,calls,transport


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="evr0023-evaluation-fake-")
        self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.index=artificial_index(); self.plan=ev.planner.build_plan(self.index,"1"*64)
        self.plan["planning_source_sha256"]={name:ev.sha(ev.HERE/name) for name in ("propose_plan.py","range_inputs.py")}
        self.scenes=scene_predictions()
        self.args=SimpleNamespace(expected_commit="a"*40,protocol=self.root/"protocol.md",plan=self.root/"plan.json",
                                  predictions=self.root/"predictions",prediction_supervision=self.root/"outer"/"receipt.json",
                                  members_index=self.root/"members.json",central_directory=self.root/"directory.bin",
                                  index_receipt=self.root/"index.json",output_dir=self.root/"evaluation")
        self.args.predictions.mkdir(); self.args.prediction_supervision.parent.mkdir()
        write(self.args.plan,self.plan); self.args.plan_sha256=ev.sha(self.args.plan)
        self.args.protocol.write_text("Artificial frozen plan "+self.args.plan_sha256)
        self.network=patch.object(ev.ranges.urllib.request,"build_opener",side_effect=AssertionError("real HTTP prohibited"))
        self.network.start(); self.addCleanup(self.network.stop)
        real_module=ev.module
        predictor=real_module("_evr0023_fixture_predictor_sources",ev.HERE/"predict.py")
        self.native_identity=dict(version="artificial",files={"fake-native":"0"*64})
        self.frontend_record=dict(frontend_bindings={"artificial_model_admission":True},
                                  source_sha256={"artificial_frontend":"0"*64},frontend_freeze="b"*40,wall_seconds=.25)
        self.frontend_bindings={}
        self.input_freeze=self.args.expected_commit
        def admit(args):
            self.assertEqual(args.expected_commit,self.args.expected_commit)
            self.assertEqual(args.input_freeze,self.input_freeze)
            return SimpleNamespace(q=object()),None,self.frontend_record,self.frontend_bindings,{}
        self.admit_frontend=Mock(side_effect=admit)
        self.admit_native=Mock(return_value=self.native_identity)
        fake_predictor=SimpleNamespace(source_paths=predictor.source_paths,frontend_admission=self.admit_frontend,
                                       native_binding=self.admit_native)
        self.modules=patch.object(ev,"module",side_effect=lambda name,path:fake_predictor if name=="_evr0023_evaluation_predictor" else real_module(name,path))
        self.modules.start();self.addCleanup(self.modules.stop)

    def frozen(self,commit,paths):
        self.assertEqual(commit,self.args.expected_commit)
        result={}
        for path in paths:
            try: key=Path(path).relative_to(ev.ROOT).as_posix()
            except ValueError: key="protocols/artificial0023.md"
            result[key]=ev.sha(path)
        return result

    def seal_fixture(self):
        predictor=ev.module("_evr0023_evaluation_fixture_predictor",ev.HERE/"predict.py")
        sources=self.frozen(self.args.expected_commit,predictor.source_paths(self.args.protocol))
        outputs={}
        for name,value in {"selection.json":{"artificial":True},**{"scene-"+s+".json":r for s,r in self.scenes.items()}}.items():
            path=self.args.predictions/name; write(path,value)
            outputs[name]=dict(sha256=ev.sha(path),bytes=path.stat().st_size)
        frontend=self.root/"frontend";frontend.mkdir();write(frontend/"receipt.json",{"artificial":True})
        fs=self.root/"frontend-supervision.json";write(fs,{"artificial":True})
        self.frontend_bindings.update(frontend_receipt_sha256=ev.sha(frontend/"receipt.json"),frontend_supervision_sha256=ev.sha(fs),
                                      frontend_supervision_log_sha256={"stdout":"0"*64,"stderr":"0"*64})
        receipt=dict(schema="evr0023.prediction.v1",status="completed",eligible=True,expected_commit=self.args.expected_commit,
                     input_freeze=self.input_freeze,
                     plan_sha256=self.args.plan_sha256,query_count=24,method_names=list(ev.METHODS),ground_truth_access=False,
                     offline_attempts=[],source_sha256=sources,outputs=outputs,wall_seconds=.5,
                     **self.frontend_bindings,installation_binding=self.frontend_record["frontend_bindings"],
                     frontend_source_sha256=self.frontend_record["source_sha256"],frontend_freeze=self.frontend_record["frontend_freeze"],
                     frontend_complete_wall_seconds=self.frontend_record["wall_seconds"],native_binding=self.native_identity)
        write(self.args.predictions/"receipt.json",receipt)
        self.reseal()
        command=["$PYTHON","$REPOSITORY/experiments/0023-mapfree/predict.py","--expected-commit",self.args.expected_commit,
                 "--plan-sha256",self.args.plan_sha256]
        for flag,path in {"--protocol":self.args.protocol,"--plan":self.args.plan,"--frontend-cache":frontend,
                          "--frontend-supervision":fs,"--native-runtime":self.root/"fake-native","--output-dir":self.args.predictions}.items():
            command.extend([flag,str(path)])
        outer_paths=[ev.HERE/"supervise.py",ev.HERE/"predict.py"]+[ev.HERE/n for n in ev.supervisor.HELPERS["prediction"]]
        out=self.args.prediction_supervision.parent
        (out/"stdout.log").write_text(json.dumps({k:receipt[k] for k in ("status","eligible","wall_seconds")})+"\n")
        (out/"stderr.log").write_text("")
        outer=dict(schema="evr0023.supervision.v1",operation="prediction",status="completed",exit_code=0,
                   expected_commit=self.args.expected_commit,termination=None,remaining_descendant_pids=[],cleanup_errors=[],
                   source_sha256=self.frozen(self.args.expected_commit,outer_paths),wall_seconds=.6,hard_timeout_seconds=1800,
                   group_rss_limit_kib=8*1024**2,peak_group_rss_kib=100,
                   thread_environment={k:"4" for k in ev.supervisor.THREAD_KEYS},command=command,
                   stdout_sha256=ev.sha(out/"stdout.log"),stderr_sha256=ev.sha(out/"stderr.log"))
        write(self.args.prediction_supervision,outer)

    def reseal(self):
        receipt=ev.read_json(self.args.predictions/"receipt.json")
        write(self.args.predictions/"seal.json",dict(schema="evr0023.prediction-seal.v1",complete=True,query_count=24,
              receipt_sha256=ev.sha(self.args.predictions/"receipt.json"),outputs=receipt["outputs"]))

    def test_directed_sign_and_rotation_errors(self):
        truth=[IDENTITY,[1.,0.,0.]]
        err=ev.pose_error([IDENTITY,[-2.,0.,0.]],truth)
        self.assertEqual((err["max_degrees"],err["signfolded_max_degrees"]),(180.,0.))
        self.assertAlmostEqual(ev.pose_error([rotation(90),[1.,0.,0.]],truth)["rotation_degrees"],90.)
        for pose in ([IDENTITY,[0.,0.,0.]], [[[1,0,0],[0,1,0],[0,0,-1]],[1,0,0]], [IDENTITY,[float('nan'),0,0]]):
            with self.assertRaises(ValueError):ev.checked_pose(pose)

    def test_auc_known_values_threshold_ties_and_frozen_reference_equivalence(self):
        self.assertEqual(ev.dms_auc([5.,5.,5.])["5"],0.)
        self.assertEqual(ev.dms_auc([0.,0.]),{"5":1.,"10":1.,"20":1.})
        auc=ev.dms_auc([0.,2.5,5.,180.]); self.assertEqual(auc["5"],.4375);self.assertEqual(auc["10"],.625)
        metrics=ev.module("_evr0023_original_auc_test",ev.HERE.parent/"0013-pose-headroom/evaluate_sealed.py")
        for errors in ([0,2.5,5,180],[1,2,4,8,16],[5,10,20],[0]*24):
            for ours,old in zip(ev.dms_auc(errors).values(),metrics.dms_auc(errors)):
                self.assertAlmostEqual(ours,old,places=15)

    def test_quaternion_normalization_and_world_to_camera_relative_pose(self):
        text="seq0/frame_00000.jpg 1.0005 0 0 0 1 2 3\nseq1/frame_00000.jpg 1 0 0 0 4 2 3\n"
        poses,meta=ev.parse_poses(text,["seq0/frame_00000.jpg","seq1/frame_00000.jpg"])
        self.assertEqual(meta["renormalized_selected_rows"],1)
        self.assertEqual(ev.relative_pose(poses["seq0/frame_00000.jpg"],poses["seq1/frame_00000.jpg"]),[IDENTITY,[3.,0.,0.]])
        q=math.sqrt(.5)
        rotated,_=ev.parse_poses("seq1/frame_00001.jpg %s 0 0 %s 0 2 0\n"%(q,q),["seq1/frame_00001.jpg"])
        a=dict(R=IDENTITY,t=[1.,0.,0.]);r,t=ev.relative_pose(a,rotated["seq1/frame_00001.jpg"])
        self.assertAlmostEqual(t[0],0.,places=12);self.assertAlmostEqual(t[1],1.,places=12)

    def test_gt_text_duplicate_missing_nonfinite_and_wrong_norm_fail(self):
        good="seq0/frame_00000.jpg 1 0 0 0 0 0 0\n"
        for text in (good+good,good.replace("1 0 0 0","1.01 0 0 0"),good.replace("1 0 0 0","nan 0 0 0"),"",good.replace("0 0 0\n","inf 0 0\n")):
            with self.assertRaises(ValueError):ev.parse_poses(text,["seq0/frame_00000.jpg"])
        parsed,meta=ev.parse_poses(good+"seq1/frame_00000.jpg malformed nan inf unseen numeric columns ignored\n",["seq0/frame_00000.jpg"])
        self.assertEqual((meta["filename_rows_scanned"],meta["numeric_rows_parsed"]),(2,1))
        self.assertEqual(set(parsed),{"seq0/frame_00000.jpg"})
        path=self.root/"bad.json"
        for text in ('{"x":1e999}','{"x":NaN}','{"x":1,"x":2}'):
            path.write_text(text)
            with self.assertRaises(ValueError):ev.read_json(path)

    def test_all24_failure_denominators_and_fallbacks_retained(self):
        first=self.scenes[ev.SCENES[0]]["queries"][0]
        first["arms"]["directional"]=dict(status="failed",pose=None,reason="artificial failure")
        first["arms"]["hessian"]=dict(status="fallback",pose=first["native_E"]["pose"],fallback="native_E_reference_filtered_query")
        prepared=ev.prepare_predictions(self.plan,self.scenes)
        result=ev.evaluate_predictions(prepared,ground_truth(self.plan))
        self.assertEqual(len(result["rows"]),24)
        self.assertEqual(result["overall"]["directional"]["failed_count"],1)
        self.assertEqual(result["overall"]["hessian"]["fallback_count"],1)
        self.assertEqual(result["overall"]["directional"]["count"],24)
        self.assertEqual(result["rows"][0]["methods"]["directional"]["error"]["max_degrees"],180.)
        self.assertFalse(result["gate"]["passed"])

    def test_gate_uses_all_six_controls_and_per_scene_best(self):
        for scene in self.scenes.values():
            for row in scene["queries"]:
                row["native_E"]["pose"]=[rotation(4),[1.,0.,0.]]
                for name,outcome in row["arms"].items():outcome["pose"]=[rotation(1 if name=="directional" else 4),[1.,0.,0.]]
        result=ev.evaluate_predictions(ev.prepare_predictions(self.plan,self.scenes),ground_truth(self.plan))
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(set(result["gate"]["margins_pp"]),set(ev.METHODS[:-1]))
        self.assertTrue(result["gate"]["auc20_requirement"])
        self.assertEqual(result["method_display_names"]["native_E"],"native_E_reference_filtered_query")

    def test_complete_seal_admission_has_no_gt_calls(self):
        self.seal_fixture()
        with patch.object(ev.supervisor,"verify_frozen",side_effect=self.frozen), patch.object(ev.ranges,"load_index",return_value=self.index), \
             patch.object(ev.ranges,"RangeReader",side_effect=AssertionError("GT forbidden at barrier")):
            admitted=ev.verify_prediction_seal(self.args)
        self.assertTrue(admitted["barrier_complete"]);self.assertEqual(len(admitted["prepared"]),24)
        self.admit_frontend.assert_called_once();self.admit_native.assert_called_once()

    def test_explicit_old_input_freeze_is_reconstructed_from_sealed_command(self):
        self.input_freeze='c'*40;self.seal_fixture()
        outer=ev.read_json(self.args.prediction_supervision)
        outer['command'].extend(['--input-freeze',self.input_freeze]);write(self.args.prediction_supervision,outer)
        with patch.object(ev.supervisor,'verify_frozen',side_effect=self.frozen), patch.object(ev.ranges,'load_index',return_value=self.index), \
             patch.object(ev.ranges,'RangeReader',side_effect=AssertionError('GT forbidden at barrier')):
            admitted=ev.verify_prediction_seal(self.args)
        self.assertEqual(admitted['bindings']['input_freeze'],'c'*40)
        self.assertEqual(self.admit_frontend.call_args.args[0].expected_commit,'a'*40)
        self.assertEqual(self.admit_frontend.call_args.args[0].input_freeze,'c'*40)

    def test_missing_or_mismatched_input_freeze_blocks_before_gt(self):
        self.input_freeze='c'*40;self.seal_fixture()
        outer=ev.read_json(self.args.prediction_supervision)
        with patch.object(ev.supervisor,'verify_frozen',side_effect=self.frozen), patch.object(ev.ranges,'load_index',return_value=self.index), \
             patch.object(ev,'acquire_ground_truth') as acquire:
            for i,value in enumerate((None,'d'*40,'main','')):
                altered=copy.deepcopy(outer)
                if value is not None:altered['command'].extend(['--input-freeze',value])
                write(self.args.prediction_supervision,altered)
                self.args.output_dir=self.root/('input-freeze-failure-'+str(i))
                with self.assertRaisesRegex(ValueError,'input freeze binding'):ev.run(self.args)
                self.assertFalse(ev.read_json(self.args.output_dir/'receipt.json')['gt_access_started'])
            self.admit_frontend.assert_not_called();acquire.assert_not_called()

    def test_tampered_seal_blocks_before_any_gt_and_preserves_failed_receipt(self):
        self.seal_fixture()
        (self.args.predictions/"scene-s00513.json").write_text("{}")
        with patch.object(ev.supervisor,"verify_frozen",side_effect=self.frozen), patch.object(ev.ranges,"load_index",return_value=self.index), \
             patch.object(ev,"acquire_ground_truth") as acquire:
            with self.assertRaisesRegex(ValueError,"sealed output hash/size"):
                ev.run(self.args)
            acquire.assert_not_called()
        record=ev.read_json(self.args.output_dir/"receipt.json")
        self.assertFalse(record["gt_access_started"]);self.assertEqual(record["status"],"failed")

    def test_missing_method_invalid_pose_and_failed_outer_block_before_gt(self):
        self.seal_fixture()
        original=ev.read_json(self.args.prediction_supervision)
        with patch.object(ev.supervisor,"verify_frozen",side_effect=self.frozen), patch.object(ev.ranges,"load_index",return_value=self.index):
            for change in ({"status":"failed"},{"termination":"timeout"},{"source_sha256":{}},{"stdout_sha256":"0"*64}):
                write(self.args.prediction_supervision,dict(original,**change))
                with self.assertRaises(ValueError):ev.verify_prediction_seal(self.args)
        del self.scenes[ev.SCENES[0]]["queries"][0]["arms"]["directional"]
        with self.assertRaisesRegex(ValueError,"all six"):ev.prepare_predictions(self.plan,self.scenes)
        self.scenes=scene_predictions();self.scenes[ev.SCENES[0]]["queries"][0]["arms"]["directional"]["pose"][1]=[0,0,0]
        with self.assertRaisesRegex(ValueError,"undefined translation"):ev.prepare_predictions(self.plan,self.scenes)

    def test_failed_deep_frontend_or_changed_native_binding_blocks_gt(self):
        self.seal_fixture()
        with patch.object(ev.supervisor,"verify_frozen",side_effect=self.frozen), patch.object(ev.ranges,"load_index",return_value=self.index), \
             patch.object(ev,"acquire_ground_truth") as acquire:
            self.admit_frontend.side_effect=ValueError("failed frontend supervision")
            with self.assertRaisesRegex(ValueError,"failed frontend supervision"):ev.run(self.args)
            acquire.assert_not_called()
            self.admit_frontend.side_effect=None
            self.admit_frontend.return_value=(SimpleNamespace(q=object()),None,self.frontend_record,self.frontend_bindings,{})
            self.admit_native.return_value={"mutated_binary":True}
            self.args.output_dir=self.root/"native-failure"
            with self.assertRaisesRegex(ValueError,"native source/binary"):ev.run(self.args)
            acquire.assert_not_called()

    def test_gt_ranges_use_nine_requests_crc_etag_and_keep_frontend_gt_forbidden(self):
        index,contents,calls,transport=synthetic_zip(self.plan)
        admission=dict(barrier_complete=True,prepared=ev.prepare_predictions(self.plan,self.scenes),index=index,plan=self.plan,bindings={"artificial":True})
        record=ev.acquire_ground_truth(admission,self.root/"gt",self.root/"gt.json",transport)
        self.assertEqual((record["status"],len(calls)),("completed",9))
        self.assertEqual(record["output_bytes"],sum(map(len,contents.values())))
        for name,data in contents.items():self.assertEqual((self.root/"gt"/name).read_bytes(),data)
        with self.assertRaises(ValueError):ev.ranges.frontend_kind("val/s00513/poses.txt")
        admission["barrier_complete"]=False
        with self.assertRaisesRegex(ValueError,"prediction barrier"):
            ev.acquire_ground_truth(admission,self.root/"forbidden",self.root/"forbidden.json",transport)
        self.assertEqual(len(calls),9)

    def test_interrupted_gt_retains_partial_range_accounting(self):
        index,_,calls,transport=synthetic_zip(self.plan)
        admission=dict(barrier_complete=True,prepared=ev.prepare_predictions(self.plan,self.scenes),index=index,plan=self.plan,bindings={"artificial":True})
        def interrupted(request,timeout):
            if len(calls)==3:raise KeyboardInterrupt("artificial interruption after first member")
            return transport(request,timeout)
        with self.assertRaises(KeyboardInterrupt):ev.acquire_ground_truth(admission,self.root/"gt",self.root/"gt.json",interrupted)
        record=ev.read_json(self.root/"gt.json")
        self.assertEqual(record["status"],"failed");self.assertEqual(len(record["members"]),1)
        self.assertEqual(len(record["requests"]),4);self.assertEqual(record["error"]["type"],"KeyboardInterrupt")

    def test_full_artificial_evaluation_is_exclusive_and_all_queries_scored(self):
        index,_,calls,transport=synthetic_zip(self.plan)
        admission=dict(barrier_complete=True,prepared=ev.prepare_predictions(self.plan,self.scenes),index=index,plan=self.plan,
                       bindings={"artificial":True},source_sha256={"artificial":"0"*64})
        original=ev.acquire_ground_truth
        def acquire(a,out,receipt):return original(a,out,receipt,transport)
        with patch.object(ev,"verify_prediction_seal",return_value=admission),patch.object(ev,"acquire_ground_truth",side_effect=acquire):
            record=ev.run(self.args)
        result=ev.read_json(self.args.output_dir/"evaluation.json")
        self.assertEqual((record["status"],result["query_count"],len(calls)),("completed",24,9))
        self.assertEqual(result["overall"]["directional"]["auc_fraction"]["5"],1.)
        self.assertFalse(result["gate"]["passed"])
        with self.assertRaisesRegex(ValueError,"fresh evaluation"):ev.run(self.args)


if __name__=="__main__":unittest.main()
