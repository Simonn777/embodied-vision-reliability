// 0012d: prospective fixed-grid screening; includes the frozen common kernel.
// Floating-point padding is conservative engineering, not a formal certificate.
// Modes 0/1 are unchanged full/random partial; 2 grouped partial; 3 geometric
// screening; 4 geometric then Cantelli screening. All setup is charged per call.
#include "native_score.cpp"
#include <memory>
#include <unordered_map>

extern "C" {
struct NGResult {
    NSResult base;
    uint64_t moment_cells, moment_points, grid_cells, eligible_cells;
    uint64_t geometry_calls, cantelli_calls;
};
}

namespace {
constexpr double GRID_WIDTH = 256.0;
constexpr size_t MIN_SCREEN_POINTS = 64;
using GridKey = std::array<int64_t, 4>;

struct GridHash {
    size_t operator()(const GridKey& key) const {
        size_t h = 0;
        for (int64_t v : key) {
            const size_t x = std::hash<int64_t>{}(v);
            h ^= x + size_t{0x9e3779b9} + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct GridMoments {
    std::array<double, 45> value{};
    std::array<double, 9> sqrt_diagonal_upper{};
    double sqrt_n_lower = 0.0;
    bool valid = false;
};

struct GridCell {
    size_t head = 0, tail = 0, count = 0, begin = 0, end = 0;
    std::array<double, 4> lo{}, hi{}, center{}, halfwidth{};
    bool finite_box = true;
    std::unique_ptr<GridMoments> moments;
};

struct Grid {
    const Input& data;
    std::vector<GridCell> cells;
    std::vector<size_t> order;
    size_t eligible_points = 0, eligible_cells = 0;
    bool valid = true;

    explicit Grid(const Input& input) : data(input), order(input.n) {
        std::unordered_map<GridKey, size_t, GridHash> lookup;
        lookup.reserve(input.n);
        std::vector<size_t> next(input.n, input.n);
        cells.reserve(std::min(input.n, size_t{256}));
        for (size_t p = 0; p < input.n; ++p) {
            GridKey key{};
            for (size_t a = 0; a < 4; ++a) {
                const double scaled = std::floor(input.coordinate(p, a) / GRID_WIDTH);
                if (!std::isfinite(scaled) ||
                    scaled < -0x1p63 || scaled >= 0x1p63) {
                    valid = false; break;
                }
                key[a] = static_cast<int64_t>(scaled);
            }
            if (!valid) break;
            const auto inserted = lookup.emplace(key, cells.size());
            if (inserted.second) {
                cells.emplace_back();
                auto& c = cells.back(); c.head = c.tail = p;
                for (size_t a = 0; a < 4; ++a)
                    c.lo[a] = c.hi[a] = input.coordinate(p, a);
            }
            auto& c = cells[inserted.first->second];
            if (c.count) { next[c.tail] = p; c.tail = p; }
            ++c.count;
            for (size_t a = 0; a < 4; ++a) {
                const double v = input.coordinate(p, a);
                c.lo[a] = std::min(c.lo[a], v); c.hi[a] = std::max(c.hi[a], v);
            }
        }
        // Unsupported keys fail to the original-order exact partial selector.
        if (!valid) {
            cells.clear(); std::iota(order.begin(), order.end(), size_t{0}); return;
        }
        size_t cursor = 0;
        for (auto& c : cells) {
            c.begin = cursor;
            for (size_t p = c.head; p != input.n; p = next[p]) order[cursor++] = p;
            c.end = cursor;
            for (size_t a = 0; a < 4; ++a) {
                c.center[a] = c.lo[a] / 2.0 + c.hi[a] / 2.0;
                c.halfwidth[a] = (c.hi[a] - c.lo[a]) / 2.0;
                c.halfwidth[a] += 64.0 * EPS *
                    std::max({std::abs(c.lo[a]), std::abs(c.hi[a]), TINY});
                if (!std::isfinite(c.center[a]) || !std::isfinite(c.halfwidth[a]) ||
                    c.halfwidth[a] < 0.0) c.finite_box = false;
            }
            if (c.count >= MIN_SCREEN_POINTS) {
                eligible_points += c.count; ++eligible_cells;
            }
        }
    }

    GridMoments& summarize(GridCell& c, NGResult& result) {
        if (c.moments) return *c.moments;
        c.moments = std::make_unique<GridMoments>();
        auto& m = *c.moments;
        ++result.moment_cells;
        for (size_t k = c.begin; k < c.end; ++k) {
            ++result.moment_points;
            const size_t p = order[k];
            const double x0 = data.x[2*p], x1 = data.x[2*p+1];
            const double y0 = data.y[2*p], y1 = data.y[2*p+1];
            const std::array<double,9> row = {y0*x0,y0*x1,y0,y1*x0,y1*x1,y1,x0,x1,1.0};
            size_t index = 0;
            for (size_t i = 0; i < 9; ++i) {
                if (!std::isfinite(row[i])) return m;
                for (size_t j = i; j < 9; ++j, ++index) {
                    const double product = row[i] * row[j];
                    if (!std::isfinite(product)) return m;
                    m.value[index] += product;
                    if (!std::isfinite(m.value[index])) return m;
                }
            }
        }
        // Positive diagonal sums also contain rounded feature products/squares.
        // The n-dependent allowance includes these operations and an absolute
        // underflow floor via pad's TINY scale. Never assume rounded M is PSD.
        for (size_t i = 0; i < 9; ++i) {
            const double raw = m.value[moment_index(i,i)];
            if (!std::isfinite(raw) || raw < 0.0) return m;
            const auto diagonal = enclose(raw, raw, raw, c.count + 64);
            if (!std::isfinite(diagonal.hi) || diagonal.hi < 0.0) return m;
            const double root = std::sqrt(diagonal.hi);
            const double upper = enclose(root, root, root, 64).hi;
            if (!std::isfinite(upper) || upper < 0.0) return m;
            m.sqrt_diagonal_upper[i] = upper;
        }
        const double root = std::sqrt(static_cast<double>(c.count));
        m.sqrt_n_lower = enclose(root, root, root, 64).lo;
        m.valid = std::isfinite(m.sqrt_n_lower) && m.sqrt_n_lower > 0.0;
        return m;
    }
};

double abs_upper(const Interval& a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }
bool finite_interval(const Interval& a) {
    return std::isfinite(a.lo) && std::isfinite(a.hi) && a.lo <= a.hi;
}

Interval center_affine(double a, double b, double offset, double x, double y) {
    const double value = (a*x + b*y) + offset;
    const double scale = (std::abs(a)*std::abs(x) + std::abs(b)*std::abs(y)) + std::abs(offset);
    return enclose(value, value, scale, 64);
}

struct GeometryLower { double lower = 0.0, denominator_upper = 0.0; bool capped = false, valid = false; };

GeometryLower geometric_lower(const GridCell& c, const Matrix& e, double t) {
    GeometryLower out;
    if (!c.finite_box) return out;
    if (all_zero(e)) return {cap_value(c.count,t).lo, 0.0, true, true};
    std::array<Interval,3> p;
    std::array<Interval,2> q;
    for (size_t i = 0; i < 3; ++i)
        p[i] = center_affine(e[3*i],e[3*i+1],e[3*i+2],c.center[0],c.center[1]);
    for (size_t j = 0; j < 2; ++j)
        q[j] = center_affine(e[j],e[3+j],e[6+j],c.center[2],c.center[3]);
    for (const auto& v : p) if (!finite_interval(v)) return out;
    for (const auto& v : q) if (!finite_interval(v)) return out;
    long double rlo = p[2].lo, rhi = p[2].hi, rs = abs_upper(p[2]);
    for (size_t i = 0; i < 2; ++i) {
        const double y = c.center[2+i];
        const double a = y*p[i].lo, b = y*p[i].hi;
        if (!std::isfinite(a) || !std::isfinite(b)) return out;
        rlo += std::min(a,b); rhi += std::max(a,b); rs += std::max(std::abs(a),std::abs(b));
    }
    const auto central = enclose(static_cast<double>(rlo),static_cast<double>(rhi),static_cast<double>(rs),64);
    if (!finite_interval(central)) return out;
    long double radius = 0.0L;
    for (size_t i = 0; i < 2; ++i) {
        radius += abs_upper(q[i])*c.halfwidth[i];
        radius += abs_upper(p[i])*c.halfwidth[2+i];
        for (size_t j = 0; j < 2; ++j)
            radius += (std::abs(e[3*i+j])*c.halfwidth[2+i])*c.halfwidth[j];
    }
    const double raw_radius = static_cast<double>(radius);
    const double radius_hi = enclose(raw_radius,raw_radius,raw_radius,64).hi;
    if (!std::isfinite(radius_hi)) return out;
    const auto residual = enclose(central.lo-radius_hi,central.hi+radius_hi,abs_upper(central)+radius_hi,64);
    if (!finite_interval(residual)) return out;
    long double denominator = 0.0L;
    for (size_t i = 0; i < 4; ++i) {
        const bool first = i < 2; const size_t j = i % 2;
        const double a = first ? e[3*j] : e[j];
        const double b = first ? e[3*j+1] : e[3+j];
        const size_t start = first ? 0 : 2;
        const double raw = abs_upper(first ? p[j] : q[j]) +
            (std::abs(a)*c.halfwidth[start] + std::abs(b)*c.halfwidth[start+1]);
        const double upper = enclose(raw,raw,raw,64).hi;
        const double square = upper*upper;
        if (!std::isfinite(upper) || !std::isfinite(square)) return out;
        const double squared_upper = enclose(square,square,square,64).hi;
        if (!std::isfinite(squared_upper)) return out;
        denominator += squared_upper;
    }
    const double d = static_cast<double>(denominator);
    out.denominator_upper = enclose(d,d,d,64).hi;
    if (!std::isfinite(out.denominator_upper) || out.denominator_upper < 0.0) return GeometryLower{};
    out.valid = true;
    if (out.denominator_upper == 0.0) return {cap_value(c.count,t).lo,0.0,true,true};
    const double distance = residual.lo <= 0.0 && residual.hi >= 0.0 ? 0.0 :
        std::min(std::abs(residual.lo),std::abs(residual.hi));
    const double square = distance*distance;
    if (!std::isfinite(square)) return GeometryLower{};
    const double numerator = std::max(0.0,enclose(square,square,square,64).lo);
    const double ratio = numerator/out.denominator_upper;
    if (!std::isfinite(ratio)) return GeometryLower{};
    const double ratio_lower = std::max(0.0,enclose(ratio,ratio,ratio,64).lo);
    out.capped = ratio_lower >= t;
    const double raw = static_cast<double>(c.count)*std::min(ratio_lower,t);
    out.lower = std::max(0.0,enclose(raw,raw,static_cast<double>(c.count)*t,c.count+64).lo);
    if (!std::isfinite(out.lower)) return GeometryLower{};
    return out;
}

double moment_lower(Grid& grid, GridCell& cell, const Matrix& e, double t,
                    double dhi, NGResult& result) {
    ++result.cantelli_calls;
    auto& mtx = grid.summarize(cell,result);
    if (!mtx.valid || !std::isfinite(dhi) || dhi <= 0.0) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < 9; ++i) s += std::abs(e[i])*mtx.sqrt_diagonal_upper[i];
    if (!std::isfinite(s)) return 0.0;
    const double shi = enclose(s,s,s,64).hi;
    const double u = shi*shi;
    if (!std::isfinite(shi) || !std::isfinite(u)) return 0.0;
    const double qscale = enclose(u,u,u,64).hi;
    const double mean_scale_raw = shi/mtx.sqrt_n_lower;
    const double mean_scale = enclose(mean_scale_raw,mean_scale_raw,mean_scale_raw,64).hi;
    if (!std::isfinite(qscale) || !std::isfinite(mean_scale)) return 0.0;
    double q = 0.0, mean = 0.0; size_t k = 0;
    for (size_t i = 0; i < 9; ++i) {
        mean += e[i]*mtx.value[moment_index(i,8)];
        for (size_t j = i; j < 9; ++j,++k)
            q += ((e[i]*mtx.value[k])*e[j])*(i == j ? 1.0 : 2.0);
    }
    mean /= static_cast<double>(cell.count);
    if (!std::isfinite(q) || !std::isfinite(mean)) return 0.0;
    const auto qb = enclose(q,q,qscale,cell.count+64);
    const auto mb = enclose(mean,mean,mean_scale,cell.count+64);
    if (!finite_interval(qb) || !finite_interval(mb) || qb.hi < 0.0 ||
        (mb.lo <= 0.0 && mb.hi >= 0.0)) return 0.0;
    const double m = std::nextafter(std::min(std::abs(mb.lo),std::abs(mb.hi)),0.0);
    if (!std::isfinite(m) || m <= 0.0) return 0.0;
    const double m2 = m*m;
    if (!std::isfinite(m2)) return 0.0;
    const double m2lo = std::max(0.0,enclose(m2,m2,m2,64).lo);
    const double second_raw = qb.hi/static_cast<double>(cell.count);
    const double second = enclose(second_raw,second_raw,second_raw,64).hi;
    const double vr = second-m2lo, vs = std::abs(second)+m2lo;
    if (!std::isfinite(vr) || !std::isfinite(vs)) return 0.0;
    const double v = enclose(vr,vr,vs,cell.count+64).hi;
    if (!std::isfinite(v) || v < 0.0) return 0.0;
    // Same capped-cost threshold as 0012b/c; it weakly dominates m/2 alone.
    const double a = std::min(m/2.0,std::sqrt(t)*std::sqrt(dhi));
    if (!std::isfinite(a) || !(a > 0.0 && a < m)) return 0.0;
    const double delta = m-a;
    const double radius = std::hypot(std::sqrt(v),delta);
    if (!std::isfinite(radius) || radius <= 0.0) return 0.0;
    const double z = delta/radius, probability = z*z;
    const double ratio = a/std::sqrt(dhi);
    const double cost = ratio >= std::sqrt(t) ? t : ratio*ratio;
    const double pl = enclose(probability,probability,probability,64).lo;
    const double cl = enclose(cost,cost,cost,64).lo;
    if (!std::isfinite(pl) || !std::isfinite(cl)) return 0.0;
    const double raw = static_cast<double>(cell.count)*std::max(0.0,pl)*std::max(0.0,cl);
    if (!std::isfinite(raw)) return 0.0;
    const double lower = enclose(raw,raw,static_cast<double>(cell.count)*t,cell.count+64).lo;
    return std::isfinite(lower) ? std::max(0.0,lower) : 0.0;
}

double grid_lower(Grid& grid, GridCell& cell, const Matrix& e, int mode, NGResult& result) {
    ++result.base.nodes; ++result.geometry_calls;
    const auto geometric = geometric_lower(cell,e,grid.data.t);
    if (mode == 3 || !geometric.valid || geometric.capped) return geometric.lower;
    return std::max(geometric.lower,moment_lower(grid,cell,e,grid.data.t,geometric.denominator_upper,result));
}

NGResult select_grid(const Input& in, Grid& grid, int mode) {
    NGResult out{}; auto& r = out.base;
    out.grid_cells = grid.cells.size(); out.eligible_cells = grid.eligible_cells;
    r.winner = 0; r.score = in.exact(in.matrices[0]); r.residuals = in.n;
    std::vector<double> costs(in.n);
    const double guard = pad(static_cast<double>(in.n)*in.t,in.n+32);
    for (size_t h = 1; h < in.matrices.size(); ++h) {
        bool rejected = false;
        if (mode >= 3 && cap_ceiling(grid.eligible_points,in.t) > r.score+guard) {
            long double prefix = 0.0L; size_t remaining = grid.eligible_points, visited = 0;
            for (auto& cell : grid.cells) {
                if (cell.count < MIN_SCREEN_POINTS) continue;
                prefix += grid_lower(grid,cell,in.matrices[h],mode,out);
                remaining -= cell.count; ++visited;
                const double raw = static_cast<double>(prefix);
                const double lower = std::max(0.0,enclose(raw,raw,static_cast<double>(in.n)*in.t,visited+64).lo);
                if (lower > r.score+guard) { rejected = true; ++r.pruned; break; }
                // This only abandons screening; a fresh exact partial pass follows.
                if (lower+cap_ceiling(remaining,in.t) <= r.score+guard) break;
            }
        }
        if (!rejected) partial_candidate(in,h,grid.order,costs,r);
    }
    return out;
}
} // namespace

extern "C" int ng_select(const double* x1,const double* x2,uint64_t n,
    const double* matrices,uint64_t models,double tau2,const uint64_t* order,
    int mode,NGResult* out,char* error,uint64_t error_cap) noexcept {
    if ((mode == 0 || mode == 1) && out) {
        *out = NGResult{};
        return ns_select(x1,x2,n,matrices,models,tau2,order,mode,&out->base,error,error_cap);
    }
    return protect(error,error_cap,[&] {
        if (!out) throw std::invalid_argument("output cannot be null");
        if (mode < 0 || mode > 4) throw std::invalid_argument("mode must be in 0..4");
        const auto started = Clock::now();
        Input input(x1,x2,n,matrices,models,tau2); Grid grid(input);
        const auto ready = Clock::now();
        *out = select_grid(input,grid,mode);
        const auto done = Clock::now();
        out->base.setup_seconds = std::chrono::duration<double>(ready-started).count();
        out->base.scoring_seconds = std::chrono::duration<double>(done-ready).count();
    });
}

// Untimed validation checks BOTH geometry-only and geometry+Cantelli lower
// bounds, each eligible cell and their aggregate (small cells contribute zero).
// NSValidation.first_cell is the first-seen grid index; -1 denotes aggregate.
// There is no proposed upper bound; max_upper_violation remains zero.
extern "C" int ng_validate(const double* x1,const double* x2,uint64_t n,
    const double* matrices,uint64_t models,double tau2,NSValidation* out,
    char* error,uint64_t error_cap) noexcept {
    return protect(error,error_cap,[&] {
        if (!out) throw std::invalid_argument("output cannot be null");
        *out = NSValidation{}; out->first_model = out->first_cell = -1;
        Input in(x1,x2,n,matrices,models,tau2); Grid grid(in); NGResult counters{};
        auto check = [&](double lower,double exact,size_t h,int64_t cell) {
            if (!std::isfinite(lower) || lower > exact) {
                ++out->violations;
                if (out->first_model < 0) { out->first_model = h; out->first_cell = cell; }
                out->max_lower_violation = std::max(out->max_lower_violation,lower-exact);
            }
        };
        for (size_t h = 0; h < in.matrices.size(); ++h) {
            std::array<long double,2> totals{};
            for (size_t j = 0; j < grid.cells.size(); ++j) {
                auto& cell = grid.cells[j];
                if (cell.count < MIN_SCREEN_POINTS) continue;
                double exact = 0.0;
                for (size_t k = cell.begin; k < cell.end; ++k) exact += in.residual(in.matrices[h],grid.order[k]);
                for (int mode = 3; mode <= 4; ++mode) {
                    const double lower = grid_lower(grid,cell,in.matrices[h],mode,counters);
                    ++out->cell_checks; check(lower,exact,h,static_cast<int64_t>(j));
                    totals[mode-3] += lower;
                }
            }
            const double exact = in.exact(in.matrices[h]);
            for (long double total : totals) {
                const double raw = static_cast<double>(total);
                const double lower = std::max(0.0,enclose(raw,raw,static_cast<double>(n)*tau2,grid.eligible_cells+64).lo);
                ++out->aggregate_checks; check(lower,exact,h,-1);
            }
        }
    });
}
