"""Manufactured complete cache cohorts; no model, native or scientific input."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

spec = importlib.util.spec_from_file_location("_evr0023_cache_test", Path(__file__).with_name("cache_inputs.py"))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def save_cache(path, arrays):
    np.savez(path, **arrays)
    return dict(filename=path.name, bytes=path.stat().st_size, sha256=c.sha(path), arrays={
        k:dict(shape=list(v.shape), dtype=str(v.dtype), raw_c_order_sha256=c.hashlib.sha256(v.tobytes()).hexdigest())
        for k,v in arrays.items()})


class Fixture:
    def __init__(self, root):
        self.root=root; self.cache=root/"cache"; self.cache.mkdir(); self.plan_path=root/"plan.json"
        scenes=[]; images=[]; edge_list=[]; members=[]
        near=[(1,2,3),(1,2,3),(2,3,1),(3,4,2),(4,5,3),(5,6,4),(6,7,5),(7,8,6)]
        for scene in c.SCENES:
            prefix="val/"+scene+"/"; anchor=prefix+"seq0/frame_00000.jpg"
            rows=[dict(filename=prefix+"seq1/frame_%05d.jpg"%n,frame_number=n,
                       chronological_rank_zero_based=n,selected_index_one_based=n+1,
                       role="B_query" if n%2==0 else "C_reference") for n in range(16)]
            bs=[r["filename"] for r in rows[::2]]; cs=[r["filename"] for r in rows[1::2]]
            refs=[dict(source=anchor,target=name,role="A_to_C") for name in cs]
            for index,neighbors in enumerate(near):
                refs += [dict(source=bs[index],target=cs[n-1],role="B_to_C",
                              frame_number_distance=abs(2*index-(2*n-1))) for n in neighbors]
            qs=[dict(source=anchor,target=name,role="A_to_B") for name in bs]
            names=[anchor]+[r["filename"] for r in rows]
            scenes.append(dict(scene=scene,anchor=anchor,selected_seq1_chronological=rows,
                query_images=bs,reference_images=cs,reference_edges=refs,query_edges=qs,
                image_members=[dict(filename=n) for n in names],intrinsics_member=dict(filename=prefix+"intrinsics.txt")))
            images+=names; members+=names+[prefix+"intrinsics.txt"]
            edge_list += [dict(e,scene=scene) for e in refs+qs]
        self.plan=dict(schema="evr0023.proposed-plan.v1",scenes=scenes,
            frontend_acquisition=dict(image_names=images,member_names=members,
                                     intrinsics_names=["val/"+s+"/intrinsics.txt" for s in c.SCENES]))
        dump(self.plan_path,self.plan)
        source={}
        for path in ["experiments/0023-mapfree/"+n for n in ("frontend.py","range_inputs.py","propose_plan.py",
                    "acquire_operation.py","supervise.py")]+["protocols/artificial.md"]:
            p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("# invented frozen source: "+path)
            source[path]=c.sha(p)
        self.receipt=dict(schema="evr0023.frontend.v1",status="completed",eligible=True,stage="completed",
            offline_attempts=[],expected_commit="a"*40,frontend_freeze="b"*40,
            plan_sha256=c.sha(self.plan_path),input_receipt_sha256="c"*64,source_sha256=source,
            input_acquisition_bindings=dict(input_operation_sha256="1"*64,input_supervision_sha256="2"*64,
                                           input_supervision_log_sha256=dict(stdout="3"*64,stderr="4"*64)),
            frontend_bindings={"artificial_installation": "d"*64},input_sha256={n:c.hashlib.sha256(n.encode()).hexdigest() for n in members},
            settings=copy.deepcopy(c.SETTINGS),images=[],edges=[])
        k=np.array([[50.,0,31.5],[0,51.,31.5],[0,0,1.]])
        self.image_arrays={}
        for index,name in enumerate(images):
            a=dict(xy=np.array([[1.,1.],[9.,1.],[17.,9.]],np.float32),
                   K_original=k.copy(),K_resized=k.copy(),image_size_original=np.array([64,64],np.int64),
                   image_size_resized=np.array([64,64],np.int64),depth_sample_raw=np.array([2.,0.,6.]),
                   depth_sample_normalized=np.array([1.,np.nan,3.]),depth_sample_valid=np.array([True,False,True]),
                   depth_positive_valid_median=np.array(2.,np.float64),points=np.array([np.nan],np.float32))
            self.image_arrays[name]=a
            row=dict(image_index=index,filename=name,status="completed",stage="completed",
                cache=save_cache(self.cache/("image-%03d.npz"%index),a),keypoint_count=3,sampled_depth_valid_count=2,
                positive_valid_depth_median=2.,input=dict(original_encoded_sha256=self.receipt["input_sha256"][name],
                authored_K_original=k.tolist(),authored_K_resized=k.tolist(),original_shape_hwc=[64,64,3],resized_shape_hwc=[64,64,3]))
            self.receipt["images"].append(row);dump(self.cache/("image-%03d.json"%index),row)
        self.edge_arrays={}
        for index,e in enumerate(edge_list):
            # The final completed edge is empty. It must remain an outcome.
            pair=np.array([[2,2],[0,0],[1,1]],np.int64) if index!=119 else np.empty((0,2),np.int64)
            a=dict(match_ids=pair,match_scores=np.array([.6,.8,.7],np.float32) if len(pair) else np.empty(0,np.float32))
            for end,key in enumerate((e["source"],e["target"])):
                im=self.image_arrays[key];ids=pair[:,end]
                for dest,src in [("xy%d"%end,"xy"),("depth%d_raw"%end,"depth_sample_raw"),
                    ("depth%d_normalized"%end,"depth_sample_normalized"),("depth_valid%d"%end,"depth_sample_valid")]:
                    a[dest]=im[src][ids]
            self.edge_arrays[index]=a
            row=dict(edge_index=index,**e,status="completed",stage="completed",geometry_observations_removed_for_depth=0,
                match_count=len(pair),both_depth_valid_count=2 if len(pair) else 0,
                cache=save_cache(self.cache/("edge-%03d.npz"%index),a))
            self.receipt["edges"].append(row);dump(self.cache/("edge-%03d.json"%index),row)
        self.bindings={k:copy.deepcopy(self.receipt[k]) for k in c.BINDING_FIELDS};self.refresh()

    def refresh(self):
        dump(self.cache/"receipt.json",self.receipt)
        self.bindings["frontend_receipt_sha256"]=c.sha(self.cache/"receipt.json")

    def update_edge(self,index):
        row=self.receipt["edges"][index]
        row["cache"]=save_cache(self.cache/("edge-%03d.npz"%index),self.edge_arrays[index])
        dump(self.cache/("edge-%03d.json"%index),row);self.refresh()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.fixture=Fixture(self.root);self.patch=mock.patch.object(c,"ROOT",self.root);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def load(self):
        x=self.fixture;return c.load_inputs(x.plan_path,x.cache,x.bindings)

    def manufactured_resize(self):
        row=copy.deepcopy(self.fixture.receipt["images"][0])
        arrays=copy.deepcopy(self.fixture.image_arrays[row["filename"]])
        ow,oh,w,h=1001,701,512,359
        old=np.array([[731.125,0.,499.123],[0.,729.875,349.2345],[0.,0.,1.]],np.float64)
        sx,sy=w/ow,h/oh
        resized=old.copy();resized[0,0]*=sx;resized[1,1]*=sy
        resized[0,2]=(old[0,2]+.5)*sx-.5;resized[1,2]=(old[1,2]+.5)*sy-.5
        arrays.update(K_original=old,K_resized=resized,image_size_original=np.array([ow,oh],np.int64),
                      image_size_resized=np.array([w,h],np.int64))
        row["input"].update(authored_K_original=old.tolist(),authored_K_resized=resized.tolist(),
                            original_shape_hwc=[oh,ow,3],resized_shape_hwc=[h,w,3])
        return row,arrays

    def test_nontrivial_resize_matches_exact_frozen_producer_operation_order(self):
        row,arrays=self.manufactured_resize()
        old=arrays["K_original"];w,h=map(int,arrays["image_size_resized"]);ow,oh=map(int,arrays["image_size_original"])
        old_adapter=arrays["K_resized"].copy()
        old_adapter[0,2]=(old[0,2]+.5)*w/ow-.5;old_adapter[1,2]=(old[1,2]+.5)*h/oh-.5
        self.assertFalse(np.array_equal(old_adapter,arrays["K_resized"]))
        result=c.image_input(row,arrays)
        np.testing.assert_array_equal(result["K"],arrays["K_resized"])
        self.assertEqual((result["width"],result["height"]),(512,359))

    def test_one_ulp_calibration_tamper_is_still_rejected_even_if_metadata_matches(self):
        row,arrays=self.manufactured_resize()
        arrays["K_resized"][1,2]=np.nextafter(arrays["K_resized"][1,2],np.inf)
        row["input"]["authored_K_resized"]=arrays["K_resized"].tolist()
        with self.assertRaisesRegex(ValueError,"calibration/coordinate metadata mismatch"):
            c.image_input(row,arrays)

    def test_complete51_120_mapping_empty_outcome_and_zero_depth_retained(self):
        result=self.load();self.assertEqual(len(result["scenes"]),3)
        for scene in result["scenes"]:
            self.assertEqual(set(scene["images"]),{"A0"}|{"B%d"%i for i in range(1,9)}|{"C%d"%i for i in range(1,9)})
            self.assertEqual(len(scene["reference_edges"]),32);self.assertEqual(len(scene["query_edges"]),8)
            self.assertTrue(scene["images"]["B1"]["original_path"].endswith("seq1/frame_00000.jpg"))
            self.assertTrue(scene["images"]["C8"]["original_path"].endswith("seq1/frame_00015.jpg"))
            edge=scene["reference_edges"][0]
            np.testing.assert_array_equal(edge["original_match_indices"],[1,2,0])
            np.testing.assert_array_equal(edge["ids_i"],[0,1,2])
            self.assertIn(1,edge["ids_i"]);self.assertFalse(scene["images"]["A0"]["depth_valid"][1])
            self.assertEqual(scene["images"]["A0"]["depth_raw"][1],0.)
        empty=result["scenes"][-1]["query_edges"][-1]
        self.assertEqual(empty["status"],"completed");self.assertEqual(len(empty["ids_i"]),0)
        self.assertEqual(len(result["selection_receipt"]["edges"]),120)

    def test_round_robin_score_ties_and_pixel_boundaries(self):
        xy=np.array([[0,0],[10,0],[1,1],[0,10],[11,1],[2,2]],float)
        pairs=np.column_stack([np.arange(6),np.arange(6)])
        out=c.select_matches(pairs,np.array([.3,.9,.8,.7,.9,.8]),xy,80,80,128)
        np.testing.assert_array_equal(out,[2,1,3,5,4,0])
        xy=np.array([[(i%8)*10+1,(i//8%8)*10+1] for i in range(320)],float)
        pairs=np.column_stack([np.arange(320),np.arange(320)])
        scores=np.array([5-i//64 for i in range(320)],float)
        np.testing.assert_array_equal(c.select_matches(pairs,scores,xy,80,80,128),np.arange(128))
        np.testing.assert_array_equal(c.select_matches(pairs,scores,xy,80,80,256),np.arange(256))
        with self.assertRaises(ValueError):c.select_matches(pairs,scores,xy,80,80,129)

    def test_last_cache_tamper_blocks_before_any_array_read(self):
        path=self.fixture.cache/"edge-119.npz";data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with mock.patch.object(c.np,"load",side_effect=AssertionError("no arrays before complete barrier")) as loader:
            with self.assertRaisesRegex(ValueError,"cache bytes changed"):self.load()
            loader.assert_not_called()

    def test_missing_population_and_source_changes_block_before_arrays(self):
        original=copy.deepcopy(self.fixture.receipt)
        self.fixture.receipt["edges"].pop();self.fixture.refresh()
        with mock.patch.object(c.np,"load") as loader:
            with self.assertRaisesRegex(ValueError,"population"):self.load()
            loader.assert_not_called()
        self.fixture.receipt=original;self.fixture.refresh()
        (self.root/"experiments/0023-mapfree/frontend.py").write_text("changed")
        with mock.patch.object(c.np,"load") as loader:
            with self.assertRaisesRegex(ValueError,"source bytes"):self.load()
            loader.assert_not_called()

    def test_unsafe_cache_path_and_unlisted_pose_file_are_never_read(self):
        row=self.fixture.receipt["images"][0];row["cache"]["filename"]="../../poses.txt"
        dump(self.fixture.cache/"image-000.json",row);self.fixture.refresh()
        with mock.patch.object(c.np,"load") as loader:
            with self.assertRaisesRegex(ValueError,"cache name"):self.load()
            loader.assert_not_called()

    def test_bad_calibration_and_canonical_edge_copy_rejected(self):
        self.fixture.edge_arrays[0]["xy0"][0,0]+=1;self.fixture.update_edge(0)
        with self.assertRaisesRegex(ValueError,"canonical copy"):self.load()
        row=self.fixture.receipt["images"][0]
        a=self.fixture.image_arrays[row["filename"]];a["K_resized"][0,2]+=.5
        row["cache"]=save_cache(self.fixture.cache/"image-000.npz",a)
        dump(self.fixture.cache/"image-000.json",row);self.fixture.refresh()
        with self.assertRaisesRegex(ValueError,"coordinate metadata"):self.load()

    def test_only_canonical_arrays_opened_never_raw_depth_pixels_or_points(self):
        original=np.load;reads=[]
        class Tracked:
            def __init__(self,path,**kwargs):self.saved=original(path,**kwargs);self.files=self.saved.files
            def __enter__(self):return self
            def __exit__(self,*args):self.saved.close()
            def __getitem__(self,key):reads.append(key);return self.saved[key]
        with mock.patch.object(c.np,"load",side_effect=Tracked):self.load()
        self.assertNotIn("points",reads);self.assertNotIn("depth",reads);self.assertNotIn("descriptors",reads)
        self.assertEqual(set(reads),c.IMAGE_READS|c.EDGE_READS)

    def test_unlisted_artifact_and_noncomplete_receipt_are_ineligible(self):
        pose=self.fixture.cache/"poses.txt";pose.write_text("MUST NOT BE READ")
        with mock.patch.object(c.np,"load") as loader:
            with self.assertRaisesRegex(ValueError,"unlisted"):self.load()
            loader.assert_not_called()
        pose.unlink();self.fixture.receipt["eligible"]=False;self.fixture.refresh()
        with self.assertRaisesRegex(ValueError,"outcome not admitted"):self.load()

    def test_acquisition_provenance_mismatch_blocks_before_arrays(self):
        self.fixture.receipt["input_acquisition_bindings"]["input_supervision_sha256"]="9"*64
        self.fixture.refresh()
        with mock.patch.object(c.np,"load") as loader:
            with self.assertRaisesRegex(ValueError,"provenance changed"):self.load()
            loader.assert_not_called()


if __name__=="__main__":unittest.main(verbosity=2)
