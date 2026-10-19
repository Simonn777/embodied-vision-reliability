#!/usr/bin/env python3
"""Apply deterministic read-only observer edits to a fresh pinned PoseLib copy.

No build, package installation, dataset access, or estimator invocation occurs.
--check-only computes every transformation in memory and writes nothing.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = '6a636b1d84273d5e3ba97676463476569611572a'
EXPECTED = {
    'PoseLib/robust/optim/lm_impl.h': '2e102b704d03e6db1d620b74dc21da2ef733a0ef8bb69ce459ae6eb953b4fba7',
    'PoseLib/robust/optim/monodepth_relpose.h': 'f317614b36ed57b92046246acc3b200d287a94b818c4ecce6cbfd004cb10d1ab',
    'PoseLib/robust/bundle.cc': '5c6aebd4acf1cc41d354912b55eb0b4b675dcd0a5d05b9b8b49e5babc99e7555',
    'PoseLib/robust/ransac_impl.h': 'c8ac809150f9189799960b1aed391d685599fd3e38b97de77d63ffda1e159011',
    'PoseLib/robust/estimators/relative_pose.cc': '7784898ccf5a4ac6769dcde94eda154a3a7d7de35b43ae2e24055969a06f3a6f',
    'PoseLib/robust.cc': '32fc5c04a435ff9590a4fdc844f2f3c6863c768086f3e724773d276273d52e7b',
    'pybind/bindings/estimators/relative_pose.cc': '4d89f5581c03e838669d63dfac1d9b43c76f39ab6911737aad8ebdd8be97b82c',
}
HEADER = 'PoseLib/robust/trace0018.h'
INCLUDE = '#include "PoseLib/robust/trace0018.h"\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f'patch anchor occurs {count} times: {old[:100]!r}')
    return text.replace(old, new, 1)


def include(text: str, anchor: str) -> str:
    return once(text, anchor, anchor + '\n' + INCLUDE)


def patch_lm(s: str) -> str:
    s = include(s, '#include "PoseLib/types.h"')
    s = once(s, '    bool recompute_jac = true;',
             '    trace0018::LMObserver<Model> trace;\n'
             '    trace.initial(*parameters, acc, stats.cost);\n\n    bool recompute_jac = true;')
    s = once(s, '            problem.compute_jacobian(acc, *parameters);',
             '            problem.compute_jacobian(acc, *parameters);\n'
             '            trace.jacobian(acc.residual_count);')
    s = once(s, '            if (stats.grad_norm < opt.gradient_tol) {\n                break;',
             '            if (stats.grad_norm < opt.gradient_tol) {\n'
             '                trace.stopping("gradient_tol");\n                break;')
    s = once(s, '        Eigen::VectorXd sol = acc.solve(stats.lambda, opt.damping);',
             '        trace.before_solve(acc.residual_count, stats.lambda);\n'
             '        Eigen::VectorXd sol = acc.solve(stats.lambda, opt.damping);')
    s = once(s, '        if (stats.step_norm < opt.step_tol) {\n            break;',
             '        if (stats.step_norm < opt.step_tol) {\n'
             '            trace.stopping("step_tol");\n            break;')
    s = once(s, '        double cost_new = problem.compute_residual(acc, parameters_new);',
             '        double cost_new = problem.compute_residual(acc, parameters_new);\n'
             '        trace.trial(parameters_new, acc, cost_new, stats);')
    s = once(s, '            stats.lambda = std::max(opt.min_lambda, stats.lambda);',
             '            stats.lambda = std::max(opt.min_lambda, stats.lambda);\n'
             '            trace.decision(true, stats);')
    s = once(s, '            if (stats.cost > 0 && cost_decrease / stats.cost < opt.relative_cost_tol) {\n                break;',
             '            if (stats.cost > 0 && cost_decrease / stats.cost < opt.relative_cost_tol) {\n'
             '                trace.stopping("relative_cost_tol");\n                break;')
    s = once(s, '            stats.lambda = std::min(opt.max_lambda, stats.lambda);',
             '            stats.lambda = std::min(opt.max_lambda, stats.lambda);\n'
             '            trace.decision(false, stats);')
    s = once(s, '    return stats;\n}', '    trace.exit(*parameters, acc, stats);\n    return stats;\n}')
    return s


def patch_refiner(s: str) -> str:
    s = include(s, '#include "../../types.h"')
    start = s.index('    double compute_residual(Accumulator &acc, const MonoDepthTwoViewGeometry &geometry) {')
    end = s.index('    void compute_jacobian(Accumulator &acc, const MonoDepthTwoViewGeometry &geometry) {', start)
    block = s[start:end]
    block = once(block, '        const double scale = geometry.scale;',
                 '        trace0018::ResidualBlocks trace_blocks;\n        const double scale = geometry.scale;')
    block = once(block, '                acc.add_residual(r, weight_sampson * weights[i]);',
                 '                acc.add_residual(r, weight_sampson * weights[i]);\n                trace_blocks.sampson();')
    block = once(block, '                if (Z1(2) > 0) {',
                 '                trace_blocks.direction(true, Z1(2));\n'
                 '                trace_blocks.direction(false, Z2(2));\n\n                if (Z1(2) > 0) {')
    return s[:start] + block + s[end:]


def patch_bundle(s: str) -> str:
    s = include(s, '#include "PoseLib/robust/optim/lm_impl.h"')
    anchor = ('    MonoDepthRelPoseRefiner<WeightType> refiner(x1, x2, d1, d2, scale_reproj, weight_sampson, refine_shift, weights);\n'
              '    return lm_impl<decltype(refiner)>(refiner, pose, opt, callback);')
    return once(s, anchor,
                '    trace0018::LOScope trace_scope(pose, opt, scale_reproj, weight_sampson, refine_shift,\n'
                '                                    std::is_same_v<WeightType, UniformWeightVector>);\n' + anchor)


def patch_ransac(s: str) -> str:
    s = include(s, '#include "PoseLib/types.h"')
    s = once(s, '    int best_model_ind = -1;',
             '    int best_model_ind = -1;\n    std::int64_t best_trace_proposal_id = -1;')
    s = once(s, '        bool better_score = score_msac < state.best_minimal_msac_score;',
             '        bool better_score = score_msac < state.best_minimal_msac_score;\n'
             '        const std::int64_t trace_proposal_id = trace0018::proposal(models[i], i, score_msac, inlier_count,\n'
             '                                                                        more_inliers, better_score);')
    s = once(s, '            best_model_ind = i;',
             '            best_model_ind = i;\n            best_trace_proposal_id = trace_proposal_id;')
    s = once(s, '                stats.num_inliers = inlier_count;\n            }',
             '                stats.num_inliers = inlier_count;\n'
             '                trace0018::adopt(models[i], {"proposal", trace_proposal_id}, score_msac, inlier_count);\n            }')
    s = once(s, '    Model refined_model = models[best_model_ind];',
             '    Model refined_model = models[best_model_ind];\n'
             '    trace0018::prepare_proposal_lo<Model>(best_trace_proposal_id);')
    s = once(s, '        *best_model = refined_model;\n    }',
             '        *best_model = refined_model;\n'
             '        trace0018::adopt(refined_model, {"lo", trace0018::last_lo()}, refined_msac_score, inlier_count);\n    }')
    s = once(s, '        models.clear();\n        estimator.generate_models(&models);',
             '        models.clear();\n        trace0018::ransac_iteration<Model>(stats.iterations);\n'
             '        estimator.generate_models(&models);')
    s = once(s, '    Model refined_model = *best_model;',
             '    Model refined_model = *best_model;\n    trace0018::prepare_terminal_lo<Model>();')
    s = once(s, '        *best_model = refined_model;\n        stats.num_inliers = inlier_count;',
             '        *best_model = refined_model;\n        stats.num_inliers = inlier_count;\n'
             '        trace0018::adopt(refined_model, {"lo", trace0018::last_lo()}, refined_msac_score, inlier_count);')
    return s


def patch_estimator(s: str) -> str:
    s = include(s, '#include "PoseLib/robust/bundle.h"')
    return once(s,
                'void RelativePoseMonoDepthEstimator::generate_models(std::vector<MonoDepthTwoViewGeometry> *models) {\n'
                '    sampler.generate_sample(&sample);',
                'void RelativePoseMonoDepthEstimator::generate_models(std::vector<MonoDepthTwoViewGeometry> *models) {\n'
                '    sampler.generate_sample(&sample);\n    trace0018::sample(sample);')


def patch_wrapper(s: str) -> str:
    s = include(s, '#include "PoseLib/robust/utils.h"')
    s = once(s,
             '    RansacStats stats = ransac_monodepth_relpose(x1_calib, x2_calib, depth_1, depth_2, opt_scaled, geometry, inliers);',
             '    RansacStats stats = ransac_monodepth_relpose(x1_calib, x2_calib, depth_1, depth_2, opt_scaled, geometry, inliers);\n'
             '    trace0018::pipeline("pre_final", geometry, stats.num_inliers > 3);')
    anchor = ('        refine_monodepth_relpose(x1_inliers, x2_inliers, d1_inliers, d2_inliers, geometry, scale_reproj, weight_sampson,\n'
              '                                 scaled_bundle_opt, opt.estimate_shift);')
    return once(s, anchor, '        trace0018::set_final_support(*inliers);\n' + anchor + '\n        trace0018::final_done();')


def patch_binding(s: str) -> str:
    s = include(s, '#include <PoseLib/poselib.h>')
    s = once(s, INCLUDE, INCLUDE + '#include <stdexcept>\n#include <string>\n#include <tuple>\n')
    anchor = 'std::pair<ImagePair, py::dict>\nestimate_shared_focal_relative_pose_wrapper('
    s = once(s, anchor, (HERE / 'trace_pybind.cc.in').read_text() + '\n' + anchor)
    anchor = 'void register_relative_pose(py::module &m) {'
    registration = '''
    m.def("estimate_monodepth_relative_pose_trace", &estimate_monodepth_relative_pose_trace_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"),
          py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("opt") = py::dict(),
          py::arg("initial_pose") = py::none(), py::arg("trace_capacity") = 50000,
          "0018 log-only calibrated monodepth observer; original algorithm and API remain available.");
'''
    return once(s, anchor, anchor + registration)


TRANSFORMS = dict(zip(EXPECTED, [patch_lm, patch_refiner, patch_bundle, patch_ransac,
                                patch_estimator, patch_wrapper, patch_binding]))


def prepare(source: Path):
    before, after = {}, {}
    for name, digest in EXPECTED.items():
        p = source / name
        if p.is_symlink():
            raise ValueError(f'source symlink refused: {name}')
        data = p.read_bytes()
        if sha(data) != digest:
            raise ValueError(f'original source hash mismatch: {name}')
        before[name] = data
        after[name] = TRANSFORMS[name](data.decode()).encode()
    if (source / HEADER).exists():
        raise ValueError('observer header already exists')
    after[HEADER] = (HERE / 'trace0018.h.in').read_bytes()
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    before, after = prepare(source)
    receipt = {
        'schema': 'evr0018.source-patch.v1', 'base_commit': PIN,
        'source': str(source), 'check_only': args.check_only,
        'generator_sha256': sha(Path(__file__).read_bytes()),
        'template_sha256': {p: sha((HERE / p).read_bytes()) for p in ['trace0018.h.in', 'trace_pybind.cc.in']},
        'files': {p: {'before_sha256': sha(before[p]) if p in before else None,
                      'after_sha256': sha(data)} for p, data in after.items()},
        'patch': ''.join(''.join(difflib.unified_diff(before.get(p,b'').decode().splitlines(True),
                            data.decode().splitlines(True), fromfile='a/' + p, tofile='b/' + p))
                         for p, data in after.items()),
    }
    if args.check_only:
        print(json.dumps({k:v for k,v in receipt.items() if k != 'patch'}, sort_keys=True))
        return
    if '0018-preparation' not in source.parts:
        raise ValueError('mutation requires a fresh copy under 0018-preparation')
    if args.receipt is None:
        raise ValueError('--receipt is required for mutation')
    receipt_path = args.receipt.resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusive receipt before the first mutation; an interrupted apply
    # leaves a visible incomplete receipt and cannot silently be reused.
    with receipt_path.open('x') as f:
        json.dump({'schema': receipt['schema'], 'status': 'started', 'source': str(source)}, f)
    for name, data in after.items():
        (source / name).write_bytes(data)
    for name, data in after.items():
        if (source / name).read_bytes() != data:
            raise RuntimeError(f'post-write source mismatch: {name}')
    receipt['status'] = 'completed'
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'completed', 'receipt': str(receipt_path), 'files': len(after)}))


if __name__ == '__main__':
    main()
