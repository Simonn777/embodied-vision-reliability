// Artificial observer mechanics only. Root builds/executes after source review.
// Link against the fresh instrumented PoseLib library and its existing Eigen.
#include "PoseLib/robust/optim/lm_impl.h"
#include "PoseLib/robust/optim/monodepth_relpose.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace poselib;
static void require(bool test,const char *message){if(!test)throw std::runtime_error(message);}

// A deliberately controlled 1-parameter problem isolates the observer from a
// minimal solver. It is not a replacement for the geometric cost or a method.
struct ArtificialProblem {
    using param_t=MonoDepthTwoViewGeometry;
    std::size_t num_params=1;
    double target=1.;
    bool reverse_step=false,extra_if_negative=false,nonfinite_step=false;
    double compute_residual(NormalAccumulator &acc,const param_t &m){
        acc.add_residual(m.scale-target);
        if(extra_if_negative&&m.scale<0)acc.add_residual(2.);
        return acc.get_residual();
    }
    void compute_jacobian(NormalAccumulator &acc,const param_t &m){
        Eigen::Matrix<double,1,1> jac;jac<<1.;
        acc.add_jacobian(m.scale-target,jac);
        if(extra_if_negative&&m.scale<0){jac<<0.;acc.add_jacobian(2.,jac);}
    }
    param_t step(const Eigen::VectorXd &dp,const param_t &m)const{
        param_t next=m;
        next.scale=nonfinite_step?std::numeric_limits<double>::infinity():m.scale+(reverse_step?-dp(0):dp(0));
        return next;
    }
};
struct Outcome {
    MonoDepthTwoViewGeometry state;
    BundleStats stats;
    std::vector<trace0018::Event> events;
    bool complete=true;
    std::size_t dropped=0,callbacks=0;
};
static Outcome run(ArtificialProblem problem,BundleOptions options,double initial,bool enabled,std::size_t cap=50000){
    Outcome out;out.state.scale=initial;
    trace0018::Context ctx(cap,1);
    trace0018::Scope scope(enabled?&ctx:nullptr);
    trace0018::prepare_lo("proposal_lo",{"proposal",0});
    {
        trace0018::LOScope lo(&out.state,options,0.,1.,false,true);
        out.stats=lm_impl<ArtificialProblem>(problem,&out.state,options,
            [&](const BundleStats&,RobustLoss*){++out.callbacks;});
    }
    out.complete=ctx.complete;out.dropped=ctx.dropped_events;out.events=std::move(ctx.events);return out;
}
static void equal(const Outcome &a,const Outcome &b){
    require(a.state.scale==b.state.scale,"observer changed state");
    require(a.stats.cost==b.stats.cost&&a.stats.initial_cost==b.stats.initial_cost,"observer changed cost");
    require(a.stats.iterations==b.stats.iterations&&a.stats.invalid_steps==b.stats.invalid_steps,"observer changed counters");
    require(a.stats.lambda==b.stats.lambda&&a.stats.nu==b.stats.nu,"observer changed damping");
    require(a.stats.step_norm==b.stats.step_norm&&a.stats.grad_norm==b.stats.grad_norm,"observer changed norms");
    require(a.callbacks==b.callbacks,"observer changed callback sequence");
}
static const trace0018::Event &find(const Outcome &out,const char *kind){
    for(const auto &e:out.events)if(std::strcmp(e.kind,kind)==0)return e;
    throw std::runtime_error(std::string("missing event ")+kind);
}
static std::size_t attempts(const Outcome &out){
    std::size_t n=0;for(const auto &e:out.events)if(std::strcmp(e.kind,"attempt")==0)++n;return n;
}
int main(){
    try{
        BundleOptions opt;opt.loss_type=BundleOptions::TRIVIAL;opt.max_iterations=3;
        ArtificialProblem p;
        auto accepted=run(p,opt,0.,true);equal(accepted,run(p,opt,0.,false));
        require(find(accepted,"attempt").accepted,"manufactured accepted step missing");
        require(find(accepted,"exit").cost.mean==accepted.stats.cost,"exit did not retain accepted cost");

        auto zero=run(p,opt,1.,true);equal(zero,run(p,opt,1.,false));
        require(attempts(zero)==0&&std::strcmp(find(zero,"exit").reason,"gradient_tol")==0,"zero-step exit missing");

        auto small_opt=opt;small_opt.step_tol=2.;
        auto small=run(p,small_opt,0.,true);equal(small,run(p,small_opt,0.,false));
        require(attempts(small)==0&&std::strcmp(find(small,"exit").reason,"step_tol")==0,"small-step exit missing");

        auto relative_opt=opt;relative_opt.relative_cost_tol=1e12;
        auto relative=run(p,relative_opt,0.,true);equal(relative,run(p,relative_opt,0.,false));
        require(attempts(relative)==1&&relative.callbacks==0,"relative-cost early break/callback changed");
        require(std::strcmp(find(relative,"exit").reason,"relative_cost_tol")==0,"last accepted early exit missing");

        p.reverse_step=true;p.extra_if_negative=true;
        auto rejected=run(p,opt,0.,true);equal(rejected,run(p,opt,0.,false));
        require(attempts(rejected)==3&&rejected.stats.invalid_steps==3,"manufactured rejected steps missing");
        std::size_t n=0;
        for(const auto &e:rejected.events)if(std::strcmp(e.kind,"attempt")==0){
            require(!e.accepted&&e.jacobian_count==1&&e.cost.count==2,"rejected count roles corrupted");
            require(e.solve_count==(n++==0?1:2),"native shared-count reuse was changed");
        }
        require(find(rejected,"exit").cost.count==1&&find(rejected,"exit").live_count==2,"exit snapshot confused with rejected accumulator");

        auto truncated=opt;truncated.loss_type=BundleOptions::TRUNCATED;truncated.loss_scale=.5;
        auto no_weight=run(ArtificialProblem{},truncated,0.,true);
        equal(no_weight,run(ArtificialProblem{},truncated,0.,false));
        require(find(no_weight,"initial").cost.count==1&&find(no_weight,"exit").jacobian_count==0,"TRUNCATED cost/Jacobian count distinction missing");

        auto overflow=run(p,opt,0.,true,2);equal(overflow,run(p,opt,0.,false));
        require(!overflow.complete&&overflow.events.size()==2&&overflow.dropped>0,"overflow did not fail closed");

        p=ArtificialProblem{};p.nonfinite_step=true;
        auto invalid=run(p,opt,0.,true);equal(invalid,run(p,opt,0.,false));
        require(!find(invalid,"attempt").accepted&&!std::isfinite(find(invalid,"attempt").cost.mean),"nonfinite rejected trial was not retained");
        require(std::isfinite(find(invalid,"exit").cost.mean),"invalid rejected trial replaced exit snapshot");

        // Real refiner branch counts on artificial zero-image-coordinate rays.
        std::vector<Point2D> x(4,Point2D::Zero());std::vector<double> d(4,1.);
        UniformWeightVector weights;
        MonoDepthRelPoseRefiner<> refiner(x,x,d,d,1.,0.,false,weights);
        trace0018::Context ctx(100,4);trace0018::Scope scope(&ctx);ctx.active_lo=0;
        MonoDepthTwoViewGeometry model;model.pose.t<<.1,0.,0.;
        NormalAccumulator acc;acc.initialize(7,std::make_shared<TrivialLoss>());
        for(double z:{0.,2.,-2.,0.}){
            model.pose.t(2)=z;acc.reset_residual();const double cost=refiner.compute_residual(acc,model);
            require(std::isfinite(cost),"manufactured visibility cost nonfinite");
            const std::size_t f=z>=-1?4:0,b=z<=1?4:0;
            require(ctx.last_blocks.forward==f&&ctx.last_blocks.backward==b&&acc.residual_count==f+b,
                    "literal transformed-z branch loss/gain count mismatch");
        }
        std::cout<<"{\"schema\":\"evr0018.observer-qualification.v1\",\"status\":\"passed\","
                   "\"checks\":[\"accepted\",\"zero_gradient\",\"small_step\",\"relative_cost_exit\","
                   "\"rejection_count_reuse\",\"truncated_zero_weight\",\"overflow\",\"nonfinite_trial\",\"visibility_loss_gain\"]}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
