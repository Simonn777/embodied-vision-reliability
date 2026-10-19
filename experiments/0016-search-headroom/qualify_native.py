"""Tiny manufactured scene checks native option routing, not dataset quality."""
import argparse,json,time
import common as c
import predict_pair as w

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=c.BACKENDS,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import poselib
    np=c.np
    rng=np.random.default_rng(160016)
    X=np.column_stack((rng.uniform(-2,2,48),rng.uniform(-1,1,48),rng.uniform(4,12,48)))
    t=np.array([.5,.1,.05]); X2=X+t
    K=np.array([[700.,0.,320.],[0.,620.,240.],[0.,0.,1.]])
    x1=X[:,:2]/X[:,2,None]*np.diag(K)[:2]+K[:2,2]
    x2=X2[:,:2]/X2[:,2,None]*np.diag(K)[:2]+K[:2,2]
    cam={'model':'PINHOLE','width':640,'height':480,'params':[700.,620.,320.,240.]}
    rows=[]
    for n in (2,4):
        opt=c.options(a.kind,0)
        if a.kind=='old_norm': opt.update(min_iterations=n,max_iterations=n)
        else: opt['ransac'].update(min_iterations=n,max_iterations=n)
        start=time.perf_counter()
        if a.kind=='old_norm': pose,stats=poselib.estimate_relative_pose(x1,x2,cam,cam,opt,{},initial_pose=None)
        else: pose,stats=poselib.estimate_relative_pose(x1,x2,cam,cam,opt,initial_pose=None)
        elapsed=time.perf_counter()-start
        c.validate_pose({'R':pose.R,'t':pose.t})
        if stats['iterations']!=n: raise AssertionError('Requested iteration budget ignored')
        rows.append({'requested_iterations':n,'returned_iterations':int(stats['iterations']),'seconds':elapsed,
            'finite_pose':True,'rotation_identity_max_abs_error':float(np.max(np.abs(pose.R-np.eye(3))))})
    c.write_new(a.output,{'status':'passed','kind':a.kind,'manufactured_points':48,'dataset_access':False,
        'library':w.library_metadata(poselib),'checks':rows,'bundle_defaults':poselib.BundleOptions({})})
    print(json.dumps({'kind':a.kind,'status':'passed','native_calls':2,'dataset_access':False}))
if __name__=='__main__':main()
