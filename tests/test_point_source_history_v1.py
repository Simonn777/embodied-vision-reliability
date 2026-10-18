"""Artificial source fixtures only; no real cache or numerical data access."""
import hashlib,tempfile,unittest
from pathlib import Path
from evr.point_source_history_v1 import read_bounded,source_record

class SourceHistoryTests(unittest.TestCase):
 def test_lexical_schema_and_reader_cast_are_not_conversion_execution(self):
  raw=b"# annotations.npz\nnp.load('annot.npz')\nK = a['intrinsics'].astype(np.float32)\n"
  sha=hashlib.sha256(raw).hexdigest();entry={'type':'blob','size':len(raw),'sha':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
  r=source_record(raw,'fixture.py',sha,entry);hits=r['lexical_occurrence_lines']
  self.assertEqual(hits['annotations_file'],[1]);self.assertEqual(hits['annot_file'],[2]);self.assertEqual(hits['anno_file'],[])
  self.assertEqual(hits['float32'],[3]);self.assertEqual(hits['savez'],[])
  self.assertIn('not proof',r['scope'])
 def test_hash_and_blob_identity_fail_closed(self):
  raw=b'x=1\n';sha=hashlib.sha256(raw).hexdigest();entry={'type':'blob','size':len(raw),'sha':'0'*40}
  with self.assertRaises(ValueError):source_record(raw,'fixture.py',sha,entry)
  with self.assertRaises(ValueError):source_record(raw,'fixture.py','0'*64,entry)
 def test_exact_byte_cap_and_symlink(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'text';p.write_bytes(b'abcd');self.assertEqual(read_bounded(p,4),b'abcd')
   with self.assertRaises(ValueError):read_bounded(p,3)
   q=Path(d)/'link';q.symlink_to(p)
   with self.assertRaises(ValueError):read_bounded(q)

if __name__=='__main__':unittest.main()
