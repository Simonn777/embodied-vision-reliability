"""Fake regular files and /proc only; never opens real weights or real /proc."""
import errno,json,os,tempfile,unittest
from pathlib import Path
from unittest import mock
import guard as g

class Tests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
    def tearDown(self): self.tmp.cleanup()
    def proc(self):
        proc=self.root/'proc'; d=proc/'123';(d/'fd').mkdir(parents=True);(d/'fdinfo').mkdir();(d/'cmdline').write_bytes(b'python\0unrelated.py\0')
        return proc,d
    def test_regular_identity_and_prefix_do_not_modify(self):
        a=self.root/'partial';b=self.root/'final';a.write_bytes(b'abc');b.write_bytes(b'abcdef')
        before=(g.snapshot(a),g.snapshot(b)); r=g.inspect_file(a,{'bytes':3,'sha256':g.sha(b'abc')})
        self.assertEqual(r['before'],r['after']); self.assertEqual(g.prefix_equal(a,b)['bytes_compared'],3)
        self.assertEqual(before,(g.snapshot(a),g.snapshot(b))); self.assertEqual(a.read_bytes(),b'abc')
        a.write_bytes(b'abd')
        with self.assertRaisesRegex(ValueError,'prefix'): g.prefix_equal(a,b)
    def test_symlink_rejected_and_changed_stat_refuses(self):
        f=self.root/'x';f.write_bytes(b'a');link=self.root/'l';link.symlink_to(f)
        with self.assertRaises(ValueError):g.inspect_file(link)
        before=g.snapshot(f);after=dict(before,size=2)
        with mock.patch.object(g,'snapshot',side_effect=[before,after]),self.assertRaisesRegex(ValueError,'changed'):g.inspect_file(f)
    def test_nul_argv_and_open_writer_are_observed(self):
        proc,d=self.proc();(d/'cmdline').write_bytes(b'python\0/work/experiments/0010/cli.py\0probe\0')
        f=self.root/'x';f.write_bytes(b'abc');(d/'fd/5').symlink_to(f);(d/'fdinfo/5').write_text('flags:\t0100002\n')
        r=g.process_snapshot(proc,{'x':g.snapshot(f)})
        self.assertFalse(r['admissible']);self.assertEqual(r['related'][0]['script_basenames'],['cli.py'])
        self.assertTrue(r['matching_fds'][0]['writable']);self.assertEqual(r['counts']['fd_stat']['success'],1)
        self.assertEqual(r['counts']['fd_readlink']['success'],1);self.assertFalse(r['absence_proven'])
    def test_enoent_fd_is_recorded_not_silent_proof(self):
        proc,d=self.proc();(d/'fd/5').symlink_to(self.root/'gone')
        r=g.process_snapshot(proc,{})
        self.assertTrue(r['admissible']);self.assertFalse(r['absence_proven'])
        self.assertEqual(r['counts']['fd_stat']['enoent'],1);self.assertEqual(r['counts']['fd_readlink']['success'],1)
        self.assertEqual(r['errors'][0]['kind'],'enoent')
    def test_readlink_permission_and_other_errors_stop(self):
        for error,kind in [(errno.EACCES,'permission'),(errno.EIO,'other')]:
            with self.subTest(error=error):
                proc=self.root/str(error);d=proc/'1';(d/'fd').mkdir(parents=True);(d/'cmdline').write_bytes(b'python\0')
                f=self.root/('x'+str(error));f.write_bytes(b'x');(d/'fd/1').symlink_to(f)
                with mock.patch.object(os,'readlink',side_effect=OSError(error,'fake')):r=g.process_snapshot(proc,{})
                self.assertFalse(r['admissible']);self.assertEqual(r['counts']['fd_readlink'][kind],1)
                self.assertEqual(r['counts']['fd_stat']['success'],1)
    def test_stat_permission_and_cmdline_race_counted(self):
        proc,d=self.proc();(d/'cmdline').unlink();(d/'fd/1').symlink_to(self.root/'x')
        original=Path.stat
        def patched(p,*a,**k):
            if str(p).endswith('/fd/1'):raise PermissionError(errno.EPERM,'fake')
            return original(p,*a,**k)
        with mock.patch.object(Path,'stat',patched):r=g.process_snapshot(proc,{})
        self.assertFalse(r['admissible']);self.assertEqual(r['counts']['cmdline']['enoent'],1)
        self.assertEqual(r['counts']['fd_stat']['permission'],1);self.assertEqual(r['counts']['fd_readlink']['success'],1)
    def fixture_run(self):
        base=self.root/'base';repo=base/'repo';repo.mkdir(parents=True);ctl=base/'scratch/0010-capability-control';ctl.mkdir(parents=True)
        cfg={'assets':{}}
        for i in range(12):
            name='weights/ViT-B-32.pt' if i==0 else 'metadata/f'+str(i);p=base/'clip-0010-assets'/name;p.parent.mkdir(parents=True,exist_ok=True);raw=('final-'+str(i)).encode();p.write_bytes(raw)
            cfg['assets'][name]={'bytes':len(raw),'sha256':g.sha(raw)}
        (repo/'experiments/0010').mkdir(parents=True);(repo/'experiments/0010/config.json').write_text(json.dumps(cfg))
        downloads=base/'clip-0010-downloads';downloads.mkdir();(downloads/'ViT-B-32.pt').write_bytes(b'final-0');(downloads/'ViT-B-32.pt.partial').write_bytes(b'final')
        (downloads/'.temp').write_bytes(b'temp')
        pins={}
        for label,code in [('wheels',0),('offline-install',0),('imports',0),('assets',0),('before-probe',1)]:
            e={'actual_exit':code,'raw_wait_status':code*256}
            for suffix,key in [('.command.json','command_sha256'),('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')]:
                p=ctl/(label+suffix);p.write_bytes(b'artificial');e[key]=g.sha(b'artificial')
            (ctl/(label+'.execution.json')).write_text(json.dumps(e))
        for p in ctl.iterdir():pins[p.relative_to(base).as_posix()]={'bytes':p.stat().st_size,'sha256':g.sha(p.read_bytes())}
        pinfile=self.root/'preservation-pins.json';pinfile.write_text(json.dumps({'files':pins}));cg=self.root/'cg';cg.mkdir()
        for name,value in [('memory.max','8589934592'),('memory.current','123'),('memory.stat','anon 10'),('memory.events','oom 0')]: (cg/name).write_text(value)
        proc,_=self.proc();return base,repo,cg,proc,pinfile
    def test_complete_guard_fake_inputs_retains_failed_exit(self):
        base,repo,cg,proc,pin=self.fixture_run();out=self.root/'out'
        with mock.patch.object(g,'__file__',str(self.root/'guard.py')),mock.patch.object(g,'PINS_SHA',g.sha(pin.read_bytes())),mock.patch.object(g,'PARTIAL',(5,g.sha(b'final'))),mock.patch.object(g,'TEMPS',{'.temp':4}),mock.patch.object(g,'git_gate',return_value={}),mock.patch.object(os,'sched_getaffinity',return_value={0,1,2,3},create=True):
            r=g.run(base,repo,out,proc=proc,cgroup=cg)
        self.assertEqual(r['status'],'PASS_CACHED_GUARD_ONLY',r);self.assertEqual(r['old_failure_exit'],1);self.assertFalse(r['first_probe_executed'])
        self.assertEqual(r['cause'],'UNKNOWN');self.assertEqual(r['prefix']['bytes_compared'],5)
        self.assertEqual(r['excluded_temporary_copies']['.temp']['sha256'],g.sha(b'temp'))
        self.assertTrue((base/'clip-0010-downloads/ViT-B-32.pt.partial').is_file())
    def test_existing_probe_stops_before_payload_read(self):
        base,repo,cg,proc,pin=self.fixture_run();(base/'scratch/0010-capability-output').mkdir()
        with mock.patch.object(os,'sched_getaffinity',return_value={0,1,2,3},create=True),mock.patch.object(g,'inspect_file',side_effect=AssertionError('must not read')):
            r=g.run(base,repo,self.root/'out',proc=proc,cgroup=cg)
        self.assertEqual(r['status'],'BLOCKED');self.assertIn('prior probe',r['reason'])

if __name__=='__main__':unittest.main(verbosity=2)
