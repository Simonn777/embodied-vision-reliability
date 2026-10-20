// Separate bridge to PoseLib 6a636b1d84273d5e3ba97676463476569611572a.
// PoseLib copyright (c) 2020 Viktor Larsson, BSD-3-Clause; see NOTICE.md.
// No upstream implementation is copied or modified by this file.
#include "PoseLib/robust/bundle.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

extern "C" {
struct NRResult {
    std::uint64_t native_called, completed, iterations, invalid_steps;
    double R[9], q[4], t[3], scale, shift1, shift2;
    double initial_cost, cost, lambda, nu, step_norm, grad_norm;
    double loss_scale;
    std::uint64_t output_positive_depth1, output_positive_depth2;
};
std::uint64_t nr_abi_version() noexcept { return 260001; }
std::uint64_t nr_result_size() noexcept { return sizeof(NRResult); }
}

namespace {
bool all_finite(const double *x, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) if (!std::isfinite(x[i])) return false;
    return true;
}
void require(bool valid, const char *message) {
    if (!valid) throw std::invalid_argument(message);
}
void error_text(char *buffer, std::uint64_t capacity, const char *text) noexcept {
    if (buffer && capacity) std::snprintf(buffer, static_cast<std::size_t>(capacity), "%s", text);
}
}

// Status 0: finite completed native call (physical output status remains explicit).
// Status 1: rejected input/physical seed before native; 2: native/C++ exception;
// 3: nonfinite/invalid native numerical output. No status silently repairs a seed.
// Caller owns all storage; length/shape checks are additionally required at ctypes boundary.
extern "C" int nr_refine(const double *x1, const double *x2, const double *d1,
    const double *d2, std::uint64_t n, const double *rotation, const double *translation,
    double a, double b, double c, double focal1, double focal2,
    std::uint64_t max_iterations, NRResult *out, char *error, std::uint64_t error_capacity) noexcept {
    if (out) std::memset(out, 0, sizeof(*out));
    error_text(error, error_capacity, "");
    try {
        require(out && x1 && x2 && d1 && d2 && rotation && translation, "null input/output pointer");
        require(n >= 4 && n <= 1000000, "support count must be 4..1000000");
        require(max_iterations <= 100, "max_iterations must be 0..100");
        require(all_finite(x1, 2*n) && all_finite(x2, 2*n) && all_finite(d1,n) && all_finite(d2,n)
                && all_finite(rotation,9) && all_finite(translation,3), "nonfinite input array");
        require(std::isfinite(a) && a > 0 && std::isfinite(b) && std::isfinite(c), "invalid affine calibration");
        require(std::isfinite(focal1) && focal1 > 0 && std::isfinite(focal2) && focal2 > 0, "invalid focal");
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        for (int i=0; i<3; ++i) { t(i)=translation[i]; for(int j=0;j<3;++j) R(i,j)=rotation[3*i+j]; }
        require((R.transpose()*R-Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <= 1e-8
                && std::abs(R.determinant()-1.0) <= 1e-8, "rotation must be proper orthogonal");
        require(t.cwiseAbs().maxCoeff() > 0, "translation must be nonzero");
        const double shift2 = c/a;
        require(std::isfinite(shift2), "shift2 overflow");
        for (std::uint64_t i=0;i<n;++i) {
            const double z1=d1[i]+b, z2=a*d2[i]+c, native_z2=a*(d2[i]+shift2);
            require(std::isfinite(z1) && z1>0 && std::isfinite(z2) && z2>0
                    && std::isfinite(native_z2) && native_z2>0, "corrected support depths must be finite positive");
        }
        poselib::BundleOptions options;
        options.loss_type=poselib::BundleOptions::TRUNCATED_CAUCHY;
        options.loss_scale=0.25*2.0*(1.0/focal1+1.0/focal2);
        options.max_iterations=static_cast<std::size_t>(max_iterations);
        require(std::isfinite(options.loss_scale) && options.loss_scale>0
                && std::isfinite(options.loss_scale*options.loss_scale)
                && options.loss_scale*options.loss_scale>0, "invalid normalized loss scale");
        std::vector<poselib::Point2D> xx1,xx2;
        xx1.reserve(n); xx2.reserve(n);
        for(std::uint64_t i=0;i<n;++i) { xx1.emplace_back(x1[2*i],x1[2*i+1]); xx2.emplace_back(x2[2*i],x2[2*i+1]); }
        const std::vector<double> dd1(d1,d1+n),dd2(d2,d2+n),weights;
        poselib::MonoDepthTwoViewGeometry geometry(R,t,a,b,shift2);
        out->native_called=1;
        const auto stats=poselib::refine_monodepth_relpose(xx1,xx2,dd1,dd2,&geometry,
            1.0/64.0,1.0,options,true,weights);
        const Eigen::Matrix3d final_R=geometry.pose.R();
        for(int i=0;i<3;++i) { out->t[i]=geometry.pose.t(i); for(int j=0;j<3;++j) out->R[3*i+j]=final_R(i,j); }
        for(int i=0;i<4;++i) out->q[i]=geometry.pose.q(i);
        out->scale=geometry.scale; out->shift1=geometry.shift1; out->shift2=geometry.shift2;
        out->iterations=stats.iterations; out->invalid_steps=stats.invalid_steps;
        out->initial_cost=stats.initial_cost; out->cost=stats.cost; out->lambda=stats.lambda;
        out->nu=stats.nu; out->step_norm=stats.step_norm; out->grad_norm=stats.grad_norm;
        out->loss_scale=options.loss_scale;
        const double scalars[]={out->scale,out->shift1,out->shift2,out->initial_cost,out->cost,
                                out->lambda,out->nu,out->step_norm,out->grad_norm};
        if (!all_finite(out->R,9) || !all_finite(out->q,4) || !all_finite(out->t,3)
                || !all_finite(scalars,9) || stats.iterations>max_iterations) {
            error_text(error,error_capacity,"native returned nonfinite geometry/stats or invalid iteration count"); return 3;
        }
        for(std::uint64_t i=0;i<n;++i) {
            const double z1=d1[i]+out->shift1, z2=out->scale*(d2[i]+out->shift2);
            out->output_positive_depth1 += std::isfinite(z1) && z1>0;
            out->output_positive_depth2 += std::isfinite(z2) && z2>0;
        }
        out->completed=1;
        return 0;
    } catch(const std::exception &e) {
        error_text(error,error_capacity,e.what()); return out && out->native_called ? 2 : 1;
    } catch(...) {
        error_text(error,error_capacity,"unknown C++ exception"); return 2;
    }
}
