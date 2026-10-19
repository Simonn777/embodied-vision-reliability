"""Artificial inputs only; no HDF5, images, native library or scientific records."""
import copy
import importlib.util
import math
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("diagnose_trace_0018_test", Path(__file__).with_name("diagnose_trace.py"))
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)


def pair(n=2):
    K = np.array([[500., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    camera = {"model": "PINHOLE", "params": [500., 600., 320., 240.], "width": 640, "height": 480}
    return {"x1": np.tile([320., 240.], (n, 1)), "x2": np.tile([320., 240.], (n, 1)),
            "depth1": np.arange(1, n+1, dtype=float), "depth2": np.arange(1, n+1, dtype=float),
            "calibration": {"K1": K, "K2": K.copy(), "camera1": camera, "camera2": copy.deepcopy(camera)}}


def state():
    return {"q": [1., 0., 0., 0.], "t": [.2, 0., 0.], "scale": 1., "shift1": 0., "shift2": 0.}


def effective(loss="TRUNCATED"):
    return {"scale_reproj": 1., "weight_sampson": 1., "loss_scale": .2, "loss_type": loss, "weights": "uniform"}


def artificial_blocks(losses, active):
    losses = np.array(losses, dtype=float).reshape(-1, 3)
    active = np.array(active, dtype=bool).reshape(-1, 3)
    losses[~active] = 0.
    total = sum(float(v) for v in losses.ravel())
    n = int(active.sum())
    return d.Blocks(tuple(range(len(losses))), active, losses, total, n, total/max(1,n), n)


def trace_fixture(arm="depth_scale",rejected=True,n=2):
    p = d.prepare_inputs(pair(n))
    s0 = state(); s0["t"] = [.001,0.,0.]; s0["R"] = np.eye(3).tolist()
    s1 = copy.deepcopy(s0); s1["t"] = [.0005,0.,0.]
    opt = d.expected_effective(p,arm,"proposal_lo")
    def ev(kind,**kw):
        return {"kind":kind,"lo_id":0,"stage":"proposal_lo","finite":True,**kw}
    def cost_event(kind,s,**kw):
        b = d.reconstruct_blocks(p,s,opt,list(range(n)))
        return ev(kind,state=s,native_cost={"numerator":b.numerator,"count":b.count,"mean":b.mean},
            blocks={"sampson_blocks":n,"forward_active":n,"backward_active":n,"forward_nonfinite_z":0,"backward_nonfinite_z":0},**kw)
    events = [ev("pipeline",lo_id=-1,phase="begin"),
        ev("proposal",lo_id=-1,proposal_id=1,state=s0),
        ev("incumbent",lo_id=-1,source_kind="proposal",source_id=1,previous_source_kind="none",previous_source_id=-1,state=s0),
        ev("lo_begin",support=list(range(n)),effective=opt,parent_kind="proposal",parent_id=1,state=s0),
        cost_event("initial",s0),
        cost_event("attempt",s1,iteration=0,accepted=True,jacobian_count=3*n,solve_count=3*n,lambda_before=.001,lambda_after=.0001,grad_norm=1.,step_norm=.0005)]
    if rejected:
        bad = copy.deepcopy(s1); bad["scale"] = None
        events.append(ev("attempt",finite=False,iteration=1,accepted=False,state=bad,
            native_cost={"numerator":None,"count":6,"mean":None},jacobian_count=5,solve_count=6))
    events += [cost_event("exit",s1,iterations=2 if rejected else 1,invalid_steps=int(rejected),reason="step_tol",live_count=5,jacobian_count=5),
        ev("lo_end",state=s1),
        ev("incumbent",lo_id=-1,source_kind="lo",source_id=0,previous_source_kind="proposal",previous_source_id=1,state=s1),
        ev("pipeline",lo_id=-1,phase="pre_final",source_kind="lo",source_id=0,state=s1,final_refinement_enabled=False),
        ev("pipeline",lo_id=-1,phase="returned",source_kind="lo",source_id=0,state=s1,final_refinement_performed=False)]
    return {"schema":"evr0018.native-trace.v1","complete":True,"capacity":50000,
        "event_count":len(events),"dropped_events":0,"proposal_trace_hash_hex":"1234567890abcdef",
        "sampled_draws":1,"scored_proposals":1,"events":events}


METRICS_PATH = Path(__file__).resolve().parents[1]/"0013-pose-headroom/evaluate_sealed.py"
metrics = d._module("evr0018_test_metrics",METRICS_PATH)


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(metrics.json_safe(obj),allow_nan=False))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_fixture(root):
    data = root/"data"; data.mkdir()
    folder = root/"predictions"; folder.mkdir()
    mp,bp,sp = root/"manifest.json",root/"baseline.json",folder/"seal.json"
    bp.write_text("artificial baseline"); sp.write_text("artificial sealed traces")
    selected=[]
    for scene in range(5):
        path = data/f"scene{scene}.h5"; path.write_bytes(b"NOT HDF5 "+str(scene).encode())
        for j in range(16):
            selected.append({"pair_index":16*scene+j,"scene":f"scene{scene}","key":f"key{j}",
                "filename":path.name,"dataset_bytes":path.stat().st_size,"dataset_sha256":file_hash(path)})
    manifest={"selected_pairs":selected}
    write_json(mp,manifest)
    p=pair(); p["input_diagnostics"]={"artificial":True}
    def hashes(p):
        return {k:hashlib.sha256(np.ascontiguousarray(p[k]).tobytes()).hexdigest() for k in ("x1","x2","depth1","depth2")}
    records=[]
    for row in selected:
        for arm in d.ARMS:
            trace=trace_fixture(arm)
            tp=folder/f"{row['pair_index']}-{arm}.trace.json"
            write_json(tp,{"native":trace})
            s=trace["events"][-1]["state"]
            records.append({**row,"arm":arm,"trace_path":tp.name,"trace_sha256":file_hash(tp),
                "mask_length":2,"kept_count":2,"native_inliers_packbits_big_hex":"c0",
                "mask_sha256":hashlib.sha256(np.ones(2,dtype=bool).tobytes()).hexdigest(),
                "input_arrays":hashes(p),"calibration":metrics.json_safe(p["calibration"]),"input_diagnostics":p["input_diagnostics"],
                "geometry":{k:s[k] for k in ("R","t","scale","shift1","shift2")},
                "timings":{"call_seconds":.1,"charged_total_seconds":.2}})
    refs=[]
    for row in selected:
        err={k:0. for k in ("rotation_degrees","translation_degrees","max_degrees","translation_signfolded_degrees","max_signfolded_degrees")}
        refs.append({**{k:row[k] for k in ("pair_index","scene","key")},
            "ground_truth":{"R_original":np.eye(3).tolist(),"t_original":[1.,0.,0.]},
            "outcomes":{a:{"status":"completed","error":err.copy()} for a in d.ARMS}})
    rp=root/"references/evaluation.json"
    write_json(rp,{"schema":"evr0017.evaluation.v1","status":"completed","pair_count":80,
        "manifest_sha256":file_hash(mp),"seal_sha256":file_hash(bp),"rows":refs})
    target=root/"experiments/0013-pose-headroom/evaluate_sealed.py"
    target.parent.mkdir(parents=True); shutil.copyfile(METRICS_PATH,target)
    config={"baseline_evaluation":{"path":"references/evaluation.json","sha256":file_hash(rp)}}
    def bound(folder,name):
        path=Path(folder)/name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("unsafe fixture path")
        return path
    def new(path,value):
        with Path(path).open("x") as stream:
            json.dump(metrics.json_safe(value),stream,allow_nan=False)
    c=SimpleNamespace(ROOT=root,json_safe=metrics.json_safe,sha256=file_hash,write_new=new,
        source_hashes=lambda:{"artificial":"fixed"},library_config=lambda:config,bound_file=bound,input_hashes=hashes)
    verify=mock.Mock(return_value=(manifest,{"status":"completed"},records))
    def check(folder,record):
        if file_hash(bound(folder,record["trace_path"])) != record["trace_sha256"]:
            raise ValueError("trace changed")
    runner=SimpleNamespace(verify_seal=verify,check_trace=check)
    loader=mock.Mock(side_effect=lambda c,path,key:copy.deepcopy(p))
    # No HDF5/native library imported or called: the five assets are opaque text.
    return SimpleNamespace(data=data,manifest=mp,baseline=bp,seal=sp,records=records,
        c=c,runner=runner,loader=loader,reference=rp,config=config,folder=folder)


class DiagnoseTests(unittest.TestCase):
    def test_normalized_units_and_loss_are_native_block_mean(self):
        p = d.prepare_inputs(pair())
        self.assertEqual(p["f1"], 550.)
        np.testing.assert_array_equal(p["h1"], [[0,0,1], [0,0,1]])
        b = d.reconstruct_blocks(p, state(), effective(), [0,1])
        self.assertEqual(b.count, 6)  # Two vector reprojections count once each.
        self.assertAlmostEqual(b.numerator, .1)
        self.assertAlmostEqual(b.mean, .1/6)
        self.assertEqual(b.cost_formula_nonzero_weight_count, 4)  # threshold equality -> zero robust weight
        self.assertEqual(b.summary()["kind_counts"], {"sampson":2,"forward":2,"backward":2})

    def test_cauchy_and_residual_scaling_before_loss(self):
        p = d.prepare_inputs(pair())
        e = effective("CAUCHY")
        e["scale_reproj"] = .25
        b = d.reconstruct_blocks(p, state(), e, [0,1])
        expected = 2*.04*math.log1p(.01/.04)+2*.04*math.log1p(.0025/.04)
        self.assertAlmostEqual(b.numerator, expected, places=16)
        self.assertEqual(b.cost_formula_nonzero_weight_count, 6)

    def test_raw_translation_gauge_not_normalized(self):
        p = d.prepare_inputs(pair())
        e = effective("CAUCHY")
        s = state()
        b = d.reconstruct_blocks(p,s,e,[0,1])
        s["t"] = [2.,0.,0.]
        changed = d.reconstruct_blocks(p,s,e,[0,1])
        self.assertGreater(changed.numerator, b.numerator)
        p["d1"] *= 10
        p["d2"] *= 10
        matched = d.reconstruct_blocks(p,s,e,[0,1])
        self.assertAlmostEqual(matched.numerator,b.numerator,places=15)

    def test_literal_transformed_depth_predicate_not_source_sign(self):
        p = d.prepare_inputs(pair(1))
        p["d1"][:] = -1.
        p["d2"][:] = 3.
        s = state(); s["t"] = [0.,0.,2.]
        e = effective(); e["weight_sampson"] = 0.
        b = d.reconstruct_blocks(p,s,e,[0])
        self.assertEqual(b.count,2)
        s["t"] = [0.,0.,3.]
        b = d.reconstruct_blocks(p,s,e,[0])
        self.assertEqual(b.count,1)  # z==0 is inactive

    def test_dropping_low_cost_raises_mean_and_high_cost_lowers_mean(self):
        old = artificial_blocks([[1,9,0]], [[1,1,0]])
        lose_low = artificial_blocks([[0,9,0]], [[0,1,0]])
        r = d.decompose_transition(old,lose_low)
        self.assertEqual(r["L"],1)
        self.assertEqual(r["mean_components"]["lost"],-1)
        self.assertEqual(r["mean_components"]["denominator_change"],5)
        self.assertEqual(r["mean_components_sum"],4)
        self.assertFalse(r["resolved_decrease"])
        lose_high = artificial_blocks([[1,0,0]], [[1,0,0]])
        self.assertTrue(d.decompose_transition(old,lose_high)["resolved_decrease"])

    def test_common_lost_gain_and_zero_count_identity(self):
        old = artificial_blocks([[1,4,0]], [[1,1,0]])
        new = artificial_blocks([[2,0,7]], [[1,0,1]])
        r = d.decompose_transition(old,new)
        self.assertEqual((r["D"],r["L"],r["G"]),(1,4,7))
        self.assertEqual(r["numerator_delta"],4)
        self.assertEqual(r["mean_components_sum"],2)
        empty = artificial_blocks([[0,0,0]], [[0,0,0]])
        r = d.decompose_transition(old,empty)
        self.assertEqual(r["new_denominator"],1)
        self.assertEqual(r["mean_components_sum"],-2.5)
        self.assertEqual(d.decompose_transition(empty,new)["mean_components_sum"],4.5)

    def test_native_reconciliation_strict_count_and_frozen_tolerance(self):
        b = d.reconstruct_blocks(d.prepare_inputs(pair()),state(),effective(),[0,1])
        native = {"numerator":b.numerator,"mean":b.mean,"count":b.count}
        self.assertTrue(d.reconcile(b,native)["count_exact"])
        bad = dict(native,count=7)
        with self.assertRaises(d.TraceMismatch): d.reconcile(b,bad)
        bad = dict(native,numerator=b.numerator+1e-7)
        with self.assertRaises(d.TraceMismatch): d.reconcile(b,bad)
        bad = dict(native,mean=float("nan"))
        with self.assertRaises(d.TraceMismatch): d.reconcile(b,bad)

    def test_known_stages_and_settings_exact_support(self):
        p = d.prepare_inputs(pair())
        lo = d.expected_effective(p,"depth_affine","proposal_lo")
        final = d.expected_effective(p,"depth_affine","final_refinement")
        self.assertEqual((lo["loss_type"],lo["max_iterations"]),("TRUNCATED",25))
        self.assertEqual((final["loss_type"],final["max_iterations"]),("CAUCHY",100))
        self.assertEqual(lo["loss_scale"],2*final["loss_scale"])
        d.validate_effective(lo,lo)
        with self.assertRaises(d.TraceMismatch): d.validate_effective(final,lo)
        for indices in ([0,0],[2],[-1],[True],np.array([0])):
            with self.assertRaises(d.TraceMismatch): d.reconstruct_blocks(p,state(),lo,indices)

    def test_nonfinite_true_overflow_and_degenerate_sampson_fail_closed(self):
        p = d.prepare_inputs(pair())
        s = state(); s["t"] = [0.,0.,0.]
        with self.assertRaises(d.TraceMismatch): d.reconstruct_blocks(p,s,effective(),[0,1])
        s = state(); s["scale"] = float("nan")
        with self.assertRaises(d.TraceMismatch): d.reconstruct_blocks(p,s,effective(),[0,1])
        s = state(); s["scale"] = 1e308
        with self.assertRaises((d.TraceMismatch,FloatingPointError)):
            d.reconstruct_blocks(p,s,effective(),[0,1])
        bad = pair(); bad["calibration"]["K1"] = np.eye(3)
        with self.assertRaises(d.TraceMismatch): d.prepare_inputs(bad)

    def test_quaternion_is_not_renormalized_and_r_mismatch_rejected(self):
        q = [math.sqrt(.5),0.,0.,math.sqrt(.5)]
        np.testing.assert_allclose(d.quaternion_matrix(q),[[0,-1,0],[1,0,0],[0,0,1]],atol=5e-16)
        self.assertFalse(np.allclose(d.quaternion_matrix(np.array(q)*2),d.quaternion_matrix(q)))
        s = state(); s["R"] = np.diag([-1.,-1.,1.])
        with self.assertRaises(d.TraceMismatch):
            d.reconstruct_blocks(d.prepare_inputs(pair()),s,effective(),[0,1])

    def test_tiny_decrease_is_not_interpreted(self):
        old = artificial_blocks([[1,1,0]], [[1,1,0]])
        new = artificial_blocks([[1-1e-12,1,0]], [[1,1,0]])
        r = d.decompose_transition(old,new)
        self.assertLess(r["native_mean_delta_reconstructed"],0)
        self.assertFalse(r["resolved_decrease"])


class TraceTests(unittest.TestCase):
    def diagnose(self,trace):
        return d.diagnose_native_trace(d.prepare_inputs(pair()),trace,"depth_scale",[0,1])

    def test_all_accepted_states_exit_snapshot_and_rejected_invalid_retained(self):
        result=self.diagnose(trace_fixture())
        self.assertEqual(result["final_ancestry_lo_ids"],[0])
        lo=result["los"][0]
        self.assertEqual((len(lo["states"]),lo["rejected_count"],lo["nonfinite_rejected_count"]),(2,1,1))
        self.assertEqual(lo["exit"]["native_cost"]["count"],6)
        self.assertEqual(lo["termination"]["live_count"],5)
        self.assertEqual(lo["attempt_counts"][1]["jacobian_count"],5)
        self.assertEqual(lo["attempt_counts"][1]["solve_count"],6)
        self.assertTrue(lo["transitions"][0]["resolved_decrease"])

    def test_invalid_state_cost_counts_or_coverage_block_whole_case(self):
        for mutate in (
            lambda t:t["events"][4]["native_cost"].update(count=5),
            lambda t:t["events"][5]["state"].update(scale=None),
            lambda t:t["events"][5]["blocks"].update(forward_active=1),
            lambda t:t["events"][5].update(stage="terminal_lo"),
            lambda t:t["events"][-1].update(source_id=99),
            lambda t:t["events"][-2].update(final_refinement_enabled=True),
            lambda t:t.update(dropped_events=1),
        ):
            t=copy.deepcopy(trace_fixture()); mutate(t)
            with self.assertRaises((ValueError,TypeError)):
                self.diagnose(t)

    def test_auxiliary_nonfinite_does_not_hide_finite_cost_reconstruction(self):
        t=trace_fixture(); t["events"][5].update(finite=False,nu=None)
        result=self.diagnose(t)
        s=result["los"][0]["states"][1]
        self.assertTrue(s["auxiliary_nonfinite_with_valid_state_and_cost"])
        self.assertIsNone(s["auxiliary_diagnostics"]["nu"])

    def test_temporarily_adopted_lo_excluded_from_final_ancestry(self):
        t=trace_fixture(); s=t["events"][1]["state"]
        later=[{"kind":"proposal","lo_id":-1,"finite":True,"proposal_id":2,"state":s},
            {"kind":"incumbent","lo_id":-1,"finite":True,"source_kind":"proposal","source_id":2,
             "previous_source_kind":"lo","previous_source_id":0,"state":s}]
        t["events"][-2:-2]=later
        for ev in t["events"][-2:]:
            ev.update(source_kind="proposal",source_id=2,state=s)
        t["event_count"]=len(t["events"]); t["scored_proposals"]=2
        result=self.diagnose(t)
        self.assertEqual(result["final_ancestry_lo_ids"],[])
        self.assertTrue(result["los"][0]["adopted_as_ransac_incumbent"])
        self.assertFalse(result["los"][0]["on_final_ancestry"])

    def test_final_support_branch_and_mask_binding(self):
        r={"mask_length":2,"kept_count":2,"native_inliers_packbits_big_hex":"c0",
           "mask_sha256":hashlib.sha256(np.ones(2,dtype=bool).tobytes()).hexdigest()}
        self.assertEqual(d._mask(r,2),[0,1])
        for key,value in (("native_inliers_packbits_big_hex","c1"),("kept_count",1),("mask_sha256","bad")):
            bad={**r,key:value}
            with self.assertRaises(d.TraceMismatch):d._mask(bad,2)

    def test_actual_final_stage_uses_subset_and_cauchy_not_internal_loss(self):
        t=trace_fixture(n=5,rejected=False); p=d.prepare_inputs(pair(5))
        support=[0,1,3,4]; s=t["events"][-1]["state"]
        e=d.expected_effective(p,"depth_scale","final_refinement")
        b=d.reconstruct_blocks(p,s,e,support)
        cost={"numerator":b.numerator,"count":b.count,"mean":b.mean}
        blocks={"sampson_blocks":4,"forward_active":4,"backward_active":4,"forward_nonfinite_z":0,"backward_nonfinite_z":0}
        shared={"lo_id":1,"stage":"final_refinement","finite":True,"state":s}
        t["events"][-2]["final_refinement_enabled"]=True
        t["events"][-1:-1]=[
            {**shared,"kind":"lo_begin","support":support,"effective":e,"parent_kind":"lo","parent_id":0},
            {**shared,"kind":"initial","native_cost":cost,"blocks":blocks},
            {**shared,"kind":"exit","native_cost":cost,"blocks":blocks,"iterations":0,
             "invalid_steps":0,"reason":"gradient_tol","live_count":12,"jacobian_count":12},
            {**shared,"kind":"lo_end"}]
        t["events"][-1].update(source_id=1,final_refinement_performed=True)
        t["event_count"]=len(t["events"])
        r=d.diagnose_native_trace(p,t,"depth_scale",support)
        self.assertEqual(r["final_ancestry_lo_ids"],[1,0])
        self.assertEqual(r["los"][1]["effective"]["loss_type"],"CAUCHY")
        self.assertEqual(r["los"][1]["exit"]["native_cost"]["count"],12)
        bad=copy.deepcopy(t); bad["events"][-5]["support"]=[0,1,2,3]
        with self.assertRaises(d.TraceMismatch):d.diagnose_native_trace(p,bad,"depth_scale",support)

    def test_strict_same_step_harm_not_merely_same_lo_cooccurrence(self):
        case=self.diagnose(trace_fixture()); case.update(pair_index=0,scene="a",key="b")
        lo=case["los"][0]
        def angled(a):
            s=copy.deepcopy(state()); s["R"]=np.eye(3).tolist()
            s["t"]=[math.cos(math.radians(a)),math.sin(math.radians(a)),0.]
            return s
        lo["states"]=[{"state":angled(a)} for a in (2.,6.,6.1)]
        lo["exit"]["state"]=angled(6.1); case["returned_state"]=angled(6.1)
        tr=lo["transitions"][0]
        lo["transitions"]=[{**tr,"from_state_index":0,"to_state_index":1,"lost_count":0,
                "support_contribution_resolved_negative":False},
            {**tr,"from_state_index":1,"to_state_index":2,"lost_count":1,
                "support_contribution_resolved_negative":True}]
        err,_=metrics.pose_errors(np.eye(3),angled(6.1)["t"],np.eye(3),[1.,0.,0.])
        err.update(translation_signfolded_degrees=err["translation_degrees"],max_signfolded_degrees=err["max_degrees"])
        ref={"ground_truth":{"R_original":np.eye(3),"t_original":[1.,0.,0.]},
             "outcomes":{"depth_scale":{"status":"completed","error":err}}}
        out=d.annotate_case(case,ref,metrics)
        self.assertTrue(out["weaker_cooccurrence_only"]); self.assertFalse(out["strict_condition"])
        lo["transitions"][0].update(lost_count=1,support_contribution_resolved_negative=True)
        self.assertTrue(d.annotate_case(case,ref,metrics)["strict_condition"])
        lo["on_final_ancestry"]=False
        self.assertFalse(d.annotate_case(case,ref,metrics)["strict_condition"])
        self.assertTrue(d.material_harm(4.9999,5.0))
        self.assertTrue(d.material_harm(10.,11.))
        self.assertFalse(d.material_harm(10.,10.99999))

    def test_acos_endpoint_difference_descriptive_after_raw_state_parity(self):
        case=self.diagnose(trace_fixture()); case.update(pair_index=0,scene="a",key="b")
        s=case["returned_state"]
        baseline={k:copy.deepcopy(s[k]) for k in ("R","t","scale","shift1","shift2")}
        s["R"][0][0] -= 1e-14
        d._state_matches_output(s,baseline)  # Mandatory unchanged raw tolerance.
        zero={k:0. for k in ("rotation_degrees","translation_degrees","max_degrees","translation_signfolded_degrees","max_signfolded_degrees")}
        ref={"ground_truth":{"R_original":np.eye(3),"t_original":[1.,0.,0.]},
             "outcomes":{"depth_scale":{"status":"completed","error":zero}}}
        out=d.annotate_case(case,ref,metrics)
        difference=out["returned_error_minus_frozen0017_degrees"]["rotation_degrees"]
        self.assertGreater(difference,1e-7)
        self.assertFalse(d.close(difference,0.))  # Cost tolerance would falsely block it.
        bad=copy.deepcopy(s); bad["t"][0] += 1e-5
        with self.assertRaises(d.TraceMismatch):d._state_matches_output(bad,baseline)


class CohortTests(unittest.TestCase):
    def run_fixture(self,f,out):
        return d.run(f.data,f.manifest,f.baseline,f.seal,out,_modules=(f.c,f.runner),_pair_loader=f.loader)

    def test_complete_160_before_first_saved_label_and_all80_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=cohort_fixture(root); out=root/"diagnosis"
            original=d._load_reference
            def guarded(*args):
                self.assertEqual(len(list(out.glob("*.reconstruction.json"))),160)
                self.assertEqual(f.loader.call_count,80)
                self.assertEqual(f.runner.verify_seal.call_count,2)
                self.assertTrue(json.loads((out/"reconstruction-seal.json").read_text())["all_160_parity_and_cost_checks_passed"])
                return original(*args)
            with mock.patch.object(d,"_load_reference",side_effect=guarded) as labels:
                result=self.run_fixture(f,out)
            receipt=json.loads((out/"operation.json").read_text())
            self.assertEqual(result,0,receipt)
            self.assertEqual(labels.call_count,1)
            evaluation=json.loads((out/"evaluation.json").read_text())
            self.assertEqual(len(evaluation["rows"]),160)
            for arm in d.ARMS:
                self.assertEqual(evaluation["summary"]["all_pairs"][arm]["pair_denominator"],80)
                self.assertEqual(evaluation["summary"]["all_pairs"][arm]["returned_auc_fractions_5_10_20"],[1.,1.,1.])
            self.assertFalse(evaluation["summary"]["primary_gate"]["passed"])
            self.assertAlmostEqual(evaluation["timings"]["replay_outer_process_seconds"],32.)
            with self.assertRaises(FileExistsError):self.run_fixture(f,out)

    def test_bad_parity_seal_blocks_observations_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=cohort_fixture(root)
            f.runner.verify_seal.side_effect=ValueError("bad seal")
            with mock.patch.object(d,"_load_reference") as labels:
                self.assertEqual(self.run_fixture(f,root/"bad"),1)
                labels.assert_not_called(); f.loader.assert_not_called()

    def test_last_case_cost_failure_blocks_all_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=cohort_fixture(root); record=f.records[-1]
            path=f.folder/record["trace_path"]; trace=json.loads(path.read_text())
            trace["native"]["events"][4]["native_cost"]["count"]=5
            write_json(path,trace); record["trace_sha256"]=file_hash(path)
            with mock.patch.object(d,"_load_reference") as labels:
                self.assertEqual(self.run_fixture(f,root/"bad"),1)
                labels.assert_not_called()
            receipt=json.loads((root/"bad/operation.json").read_text())
            self.assertEqual(receipt["completed_reconstructions"],159)
            self.assertFalse((root/"bad/evaluation.json").exists())

    def test_input_hash_change_and_reference_hash_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=cohort_fixture(root)
            (f.data/"scene0.h5").write_bytes(b"changed")
            with mock.patch.object(d,"_load_reference") as labels:
                self.assertEqual(self.run_fixture(f,root/"bad"),1)
                f.loader.assert_not_called(); labels.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=cohort_fixture(root); f.reference.write_text("changed")
            self.assertEqual(self.run_fixture(f,root/"bad"),1)
            receipt=json.loads((root/"bad/operation.json").read_text())
            self.assertEqual(receipt["failed_phase"],"saved_labels")
            self.assertEqual(receipt["completed_reconstructions"],160)

    def test_primary_requires_two_scenes_and_scale_arm(self):
        cases=[]
        for i in range(80):
            for arm in d.ARMS:
                cases.append({"pair_index":i,"scene":str(i//16),"arm":arm,"strict_condition":False,
                    "weaker_cooccurrence_only":False,"returned_error":{"max_degrees":1.,"max_signfolded_degrees":1.},
                    "frozen0017_returned_error":{"max_degrees":1.},
                    "returned_error_minus_frozen0017_degrees":{"max_degrees":0.},"los":[]})
        cases[0]["strict_condition"]=True
        cases[33]["strict_condition"]=True  # affine in a second scene cannot rescue
        self.assertFalse(d.summarize_cases(cases,metrics)["primary_gate"]["passed"])
        cases[32]["strict_condition"]=True
        self.assertTrue(d.summarize_cases(cases,metrics)["primary_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
