// Exploratory native implementation of the frozen 0012b score-bound mechanism.
// Numerical padding is conservative engineering, not a formal interval proof.
// Build without fast-math or contraction; all selectors share one residual.
// Modes: 0 full; 1 original-order partial; 2 moment/full fallback;
// 3 coarse-moment-order partial; 4 moment/partial fallback. No oracle ordering.
// clang++ -std=c++17 -O3 -fPIC -shared -fno-fast-math -ffp-contract=off ...

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
struct NSResult {
    int64_t winner;
    double score;
    uint64_t residuals, nodes, refinements, pruned;
    double setup_seconds, scoring_seconds;
};
struct NSValidation {
    uint64_t cell_checks, aggregate_checks, violations;
    int64_t first_model, first_cell; // -1 cell denotes aggregate; -1 model: none.
    double max_lower_violation, max_upper_violation;
};
}

namespace {
using Clock = std::chrono::steady_clock;
using Matrix = std::array<double, 9>;
constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double TINY = std::numeric_limits<double>::min();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr size_t LEAF_SIZE = 32, INITIAL_CELLS = 128, MAX_REFINEMENTS = 128;
constexpr size_t BLOCK_SIZE = 256;

struct Interval {
    double lo, hi;
    double width() const { return std::max(0.0, hi - lo); }
};

double pad(double scale, size_t operations = 32) {
    return (64.0 * EPS * (static_cast<double>(operations) + 1.0)) *
           std::max(std::abs(scale), TINY);
}

Interval enclose(double lo, double hi, double scale, size_t operations = 32) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(scale))
        return {-INF, INF};
    const double margin = pad(scale, operations);
    if (!std::isfinite(margin)) return {-INF, INF};
    return {std::nextafter(lo - margin, -INF),
            std::nextafter(hi + margin, INF)};
}

double cap_ceiling(size_t n, double t) {
    const double cap = static_cast<double>(n) * t;
    return enclose(cap, cap, cap, n + 32).hi;
}

Interval full_cap(size_t n, double t) {
    return n == 0 ? Interval{0.0, 0.0} : Interval{0.0, cap_ceiling(n, t)};
}

Interval cap_value(size_t n, double t) {
    const double cap = static_cast<double>(n) * t;
    const auto b = enclose(cap, cap, cap, n + 32);
    return {std::max(0.0, b.lo), b.hi};
}

bool all_zero(const Matrix& e) {
    for (double v : e) if (v != 0.0) return false;
    return true;
}

struct Input {
    const double *x, *y;
    size_t n;
    double t;
    std::vector<Matrix> matrices;

    Input(const double* x1, const double* x2, uint64_t count, const double* models,
          uint64_t model_count, double tau2) : x(x1), y(x2), n(count), t(tau2) {
        if (!std::isfinite(t) || t <= 0.0) throw std::invalid_argument("tau2 must be positive and finite");
        if (model_count == 0 || models == nullptr) throw std::invalid_argument("at least one matrix is required");
        if (count > std::numeric_limits<size_t>::max() / (4 * sizeof(double)) ||
            model_count > std::numeric_limits<size_t>::max() / sizeof(Matrix) ||
            count > (std::numeric_limits<uint64_t>::max() / model_count) / 4)
            throw std::invalid_argument("input dimensions exceed supported range");
        if (n && (!x || !y)) throw std::invalid_argument("nonempty points cannot be null");
        if (!std::isfinite(static_cast<double>(n) * t)) throw std::overflow_error("N*tau2 overflows");
        for (size_t i = 0; i < 2 * n; ++i)
            if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
                throw std::invalid_argument("points must be finite");
        matrices.resize(static_cast<size_t>(model_count));
        for (size_t h = 0; h < matrices.size(); ++h) {
            double scale = 0.0;
            for (size_t j = 0; j < 9; ++j) {
                double v = models[9 * h + j];
                if (!std::isfinite(v)) throw std::invalid_argument("matrices must be finite");
                matrices[h][j] = v;
                scale = std::max(scale, std::abs(v));
            }
            if (scale != 0.0) for (double& v : matrices[h]) v /= scale;
        }
    }

    double coordinate(size_t i, size_t axis) const {
        return axis < 2 ? x[2 * i + axis] : y[2 * i + axis - 2];
    }

    double residual(const Matrix& e, size_t i) const {
        const double x0 = x[2 * i], x1 = x[2 * i + 1];
        const double y0 = y[2 * i], y1 = y[2 * i + 1];
        const double ex0 = (e[0] * x0 + e[1] * x1) + e[2];
        const double ex1 = (e[3] * x0 + e[4] * x1) + e[5];
        const double ex2 = (e[6] * x0 + e[7] * x1) + e[8];
        const double ety0 = (e[0] * y0 + e[3] * y1) + e[6];
        const double ety1 = (e[1] * y0 + e[4] * y1) + e[7];
        const double r = (y0 * ex0 + y1 * ex1) + ex2;
        const double q = r * r;
        const double d = (ex0 * ex0 + ety0 * ety0) + (ex1 * ex1 + ety1 * ety1);
        if (!std::isfinite(q) || !std::isfinite(d))
            throw std::overflow_error("residual arithmetic exceeds finite range");
        return d > 0.0 ? std::min(q / d, t) : t;
    }

    double exact(const Matrix& e) const {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i) total += residual(e, i);
        if (!std::isfinite(total)) throw std::overflow_error("score reduction overflows");
        return total;
    }
};

struct Cell {
    size_t begin = 0, end = 0;
    int left = -1, right = -1;
    // Upper triangular entries, row-major: 45 moments instead of 81 copies.
    std::array<double, 45> moment{}, absolute{};
    std::array<double, 9> row_lo{}, row_hi{};
    std::array<double, 4> coord_lo{}, coord_hi{}, center{}, halfwidth{};
    size_t n() const { return end - begin; }
};

constexpr size_t moment_index(size_t i, size_t j) {
    return i * 9 - i * (i + 1) / 2 + j; // i <= j
}

struct Tree {
    const Input& data;
    std::vector<size_t> order;
    std::vector<Cell> cells;
    std::vector<int> initial;

    explicit Tree(const Input& input) : data(input), order(input.n) {
        std::iota(order.begin(), order.end(), size_t{0});
        cells.reserve(2 * (input.n / LEAF_SIZE + 1) + 1);
        build(0, input.n);
        partition();
    }

    int build(size_t begin, size_t end) {
        const int id = static_cast<int>(cells.size());
        cells.emplace_back();
        cells[id].begin = begin;
        cells[id].end = end;
        if (end - begin > LEAF_SIZE) {
            // Geometry alone chooses the same widest-axis stable median split as
            // Python. No A rows or moments are built/scanned at internal nodes.
            std::array<double, 4> lo{}, hi{};
            for (size_t a = 0; a < 4; ++a) lo[a] = hi[a] = data.coordinate(order[begin], a);
            for (size_t k = begin + 1; k < end; ++k) for (size_t a = 0; a < 4; ++a) {
                const double v = data.coordinate(order[k], a);
                lo[a] = std::min(lo[a], v); hi[a] = std::max(hi[a], v);
            }
            size_t axis = 0;
            for (size_t a = 1; a < 4; ++a) if (hi[a] - lo[a] > hi[axis] - lo[axis]) axis = a;
            std::stable_sort(order.begin() + begin, order.begin() + end,
                [&](size_t a, size_t b) { return data.coordinate(a, axis) < data.coordinate(b, axis); });
            const size_t middle = begin + (end - begin) / 2;
            const int left = build(begin, middle), right = build(middle, end);
            Cell& c = cells[id];
            c.left = left; c.right = right;
            const Cell& l = cells[left]; const Cell& r = cells[right];
            for (size_t j = 0; j < 45; ++j) {
                c.moment[j] = l.moment[j] + r.moment[j];
                c.absolute[j] = l.absolute[j] + r.absolute[j];
            }
            for (size_t j = 0; j < 9; ++j) {
                c.row_lo[j] = std::min(l.row_lo[j], r.row_lo[j]);
                c.row_hi[j] = std::max(l.row_hi[j], r.row_hi[j]);
            }
            for (size_t a = 0; a < 4; ++a) {
                c.coord_lo[a] = std::min(l.coord_lo[a], r.coord_lo[a]);
                c.coord_hi[a] = std::max(l.coord_hi[a], r.coord_hi[a]);
            }
        } else {
            Cell& c = cells[id];
            if (begin != end) {
                c.row_lo.fill(INF); c.row_hi.fill(-INF);
                c.coord_lo.fill(INF); c.coord_hi.fill(-INF);
            }
            for (size_t k = begin; k < end; ++k) {
                const size_t p = order[k];
                const double x0 = data.x[2 * p], x1 = data.x[2 * p + 1];
                const double y0 = data.y[2 * p], y1 = data.y[2 * p + 1];
                const std::array<double, 9> row = {y0*x0, y0*x1, y0, y1*x0, y1*x1, y1, x0, x1, 1.0};
                size_t idx = 0;
                for (size_t i = 0; i < 9; ++i) {
                    c.row_lo[i] = std::min(c.row_lo[i], row[i]);
                    c.row_hi[i] = std::max(c.row_hi[i], row[i]);
                    for (size_t j = i; j < 9; ++j, ++idx) {
                        const double product = row[i] * row[j];
                        c.moment[idx] += product;
                        c.absolute[idx] += std::abs(product);
                    }
                }
                for (size_t a = 0; a < 4; ++a) {
                    const double v = data.coordinate(p, a);
                    c.coord_lo[a] = std::min(c.coord_lo[a], v);
                    c.coord_hi[a] = std::max(c.coord_hi[a], v);
                }
            }
        }
        Cell& c = cells[id];
        for (size_t a = 0; a < 4; ++a) {
            c.center[a] = c.coord_lo[a] / 2.0 + c.coord_hi[a] / 2.0;
            c.halfwidth[a] = (c.coord_hi[a] - c.coord_lo[a]) / 2.0;
            c.halfwidth[a] += 64.0 * EPS * std::max({std::abs(c.coord_lo[a]), std::abs(c.coord_hi[a]), TINY});
        }
        return id;
    }

    void partition() {
        struct Entry { size_t n, serial; int cell; };
        struct Compare {
            bool operator()(const Entry& a, const Entry& b) const {
                return a.n != b.n ? a.n < b.n : a.serial > b.serial;
            }
        };
        std::priority_queue<Entry, std::vector<Entry>, Compare> queue;
        std::vector<Entry> done;
        size_t serial = 0;
        queue.push({cells[0].n(), 0, 0});
        while (!queue.empty() && queue.size() + done.size() < INITIAL_CELLS) {
            const auto e = queue.top(); queue.pop();
            const auto& c = cells[e.cell];
            if (c.left < 0) { done.push_back(e); continue; }
            queue.push({cells[c.left].n(), ++serial, c.left});
            queue.push({cells[c.right].n(), ++serial, c.right});
        }
        while (!queue.empty()) { done.push_back(queue.top()); queue.pop(); }
        std::sort(done.begin(), done.end(), [](const Entry& a, const Entry& b) { return a.serial < b.serial; });
        for (const auto& e : done) initial.push_back(e.cell);
    }

    double exact_cell(const Matrix& e, const Cell& c) const {
        double total = 0.0;
        for (size_t i = c.begin; i < c.end; ++i) total += data.residual(e, order[i]);
        if (!std::isfinite(total)) throw std::overflow_error("cell score reduction overflows");
        return total;
    }
};

Interval affine_square(double a, double b, double offset, const Cell& c, size_t start) {
    const double value = (a*c.center[start] + b*c.center[start+1]) + offset;
    const double radius = std::abs(a)*c.halfwidth[start] + std::abs(b)*c.halfwidth[start+1];
    const double scale = (std::abs(a)*std::abs(c.center[start]) + std::abs(b)*std::abs(c.center[start+1])) + std::abs(offset) + radius;
    if (!std::isfinite(value) || !std::isfinite(radius) || !std::isfinite(scale)) return {0.0, INF};
    const auto r = enclose(value-radius, value+radius, scale);
    if (!std::isfinite(r.lo) || !std::isfinite(r.hi)) return {0.0, INF};
    const double l2 = r.lo*r.lo, h2 = r.hi*r.hi;
    if (!std::isfinite(l2) || !std::isfinite(h2)) return {0.0, INF};
    const double lo = r.lo <= 0.0 && r.hi >= 0.0 ? 0.0 : std::min(l2,h2);
    const double hi = std::max(l2,h2);
    const auto out = enclose(lo,hi,hi);
    if (!std::isfinite(out.lo) || !std::isfinite(out.hi)) return {0.0, INF};
    return {std::max(0.0,out.lo), std::max(0.0,out.hi)};
}

double cantelli(const Cell& c, const Matrix& e, double t, double qhi, double dhi) {
    double mean = 0.0, scale = 0.0;
    for (size_t i = 0; i < 9; ++i) {
        mean += e[i]*c.moment[moment_index(i,8)];
        scale += std::abs(e[i])*c.absolute[moment_index(i,8)];
    }
    mean /= static_cast<double>(c.n()); scale /= static_cast<double>(c.n());
    if (!std::isfinite(mean) || !std::isfinite(scale)) return 0.0;
    const auto mb = enclose(mean,mean,scale,c.n()+32);
    if (!std::isfinite(mb.lo) || !std::isfinite(mb.hi) || (mb.lo <= 0.0 && mb.hi >= 0.0)) return 0.0;
    const double m = std::nextafter(std::min(std::abs(mb.lo),std::abs(mb.hi)),0.0);
    if (!std::isfinite(m) || m <= 0.0) return 0.0;
    const double m2 = m*m;
    if (!std::isfinite(m2)) return 0.0;
    const auto m2b = enclose(m2,m2,m2);
    const double second = std::nextafter(qhi/static_cast<double>(c.n()),INF);
    if (!std::isfinite(m2b.lo) || !std::isfinite(second)) return 0.0;
    const double m2lo = std::max(0.0,m2b.lo);
    const double vr = second-m2lo, vs = std::abs(second)+m2lo;
    if (!std::isfinite(vr) || !std::isfinite(vs)) return 0.0;
    const double v = enclose(vr,vr,vs,c.n()+32).hi;
    if (!std::isfinite(v) || v < 0.0 || dhi <= 0.0) return 0.0;
    const double a = std::min(m/2.0,std::sqrt(t)*std::sqrt(dhi));
    if (!std::isfinite(a) || !(a > 0.0 && a < m)) return 0.0;
    const double delta = m-a;
    const double radius = std::hypot(std::sqrt(v),delta);
    if (!std::isfinite(radius) || radius <= 0.0) return 0.0;
    const double z = delta/radius, probability = z*z;
    const double ratio = a/std::sqrt(dhi);
    const double cost = ratio >= std::sqrt(t) ? t : ratio*ratio;
    const double pl = enclose(probability,probability,probability).lo;
    const double cl = enclose(cost,cost,cost).lo;
    if (!std::isfinite(pl) || !std::isfinite(cl)) return 0.0;
    const double raw = static_cast<double>(c.n())*std::max(0.0,pl)*std::max(0.0,cl);
    if (!std::isfinite(raw)) return 0.0;
    const double lower = enclose(raw,raw,static_cast<double>(c.n())*t,c.n()+32).lo;
    return std::isfinite(lower) ? std::max(0.0,lower) : 0.0;
}

Interval cell_bound(const Cell& c, const Matrix& e, double t) {
    const size_t n = c.n(); const double cap = static_cast<double>(n)*t;
    if (n == 0) return {0.0,0.0};
    if (all_zero(e)) return cap_value(n,t);
    double q = 0.0, qs = 0.0;
    size_t k = 0;
    for (size_t i = 0; i < 9; ++i) for (size_t j = i; j < 9; ++j,++k) {
        // Do not form e[i]*e[j] first: that product may underflow while the
        // moment-scaled contribution is representable. E entries have |e|<=1.
        const double factor = i == j ? 1.0 : 2.0;
        q += ((e[i]*c.moment[k])*e[j])*factor;
        qs += ((std::abs(e[i])*c.absolute[k])*std::abs(e[j]))*factor;
    }
    // Inspect raw reductions before std::max can hide NaN, including 0*Inf.
    if (!std::isfinite(q) || !std::isfinite(qs)) return full_cap(n,t);
    auto qb = enclose(q,q,qs,n+32);
    if (!std::isfinite(qb.lo) || !std::isfinite(qb.hi)) return full_cap(n,t);
    qb.lo = std::max(0.0,qb.lo); qb.hi = std::max(0.0,qb.hi);
    double rlo=0.0, rhi=0.0, rs=0.0;
    for (size_t i=0;i<9;++i) {
        const double a=e[i]*c.row_lo[i], b=e[i]*c.row_hi[i];
        if (!std::isfinite(a) || !std::isfinite(b)) return full_cap(n,t);
        rlo += std::min(a,b); rhi += std::max(a,b); rs += std::max(std::abs(a),std::abs(b));
    }
    const auto rb=enclose(rlo,rhi,rs);
    if (!std::isfinite(rb.lo) || !std::isfinite(rb.hi)) return full_cap(n,t);
    const double a2=rb.lo*rb.lo,b2=rb.hi*rb.hi;
    if (!std::isfinite(a2) || !std::isfinite(b2)) return full_cap(n,t);
    const double qmin=rb.lo<=0.0 && rb.hi>=0.0 ? 0.0 : std::min(a2,b2);
    auto qr=enclose(qmin,std::max(a2,b2),std::max(a2,b2));
    if (!std::isfinite(qr.lo) || !std::isfinite(qr.hi)) return full_cap(n,t);
    qr.lo=std::max(0.0,qr.lo); qr.hi=std::max(0.0,qr.hi);
    const std::array<Interval,4> squares = {
        affine_square(e[0],e[1],e[2],c,0), affine_square(e[3],e[4],e[5],c,0),
        affine_square(e[0],e[3],e[6],c,2), affine_square(e[1],e[4],e[7],c,2)};
    long double dl=0.0L,dh=0.0L;
    for (const auto& s:squares) {
        if (!std::isfinite(s.lo) || !std::isfinite(s.hi)) return full_cap(n,t);
        dl+=s.lo;dh+=s.hi;
    }
    auto db=enclose(static_cast<double>(dl),static_cast<double>(dh),static_cast<double>(dh));
    if (!std::isfinite(db.lo) || !std::isfinite(db.hi)) return full_cap(n,t);
    db.lo=std::max(0.0,db.lo);db.hi=std::max(0.0,db.hi);
    if (db.hi==0.0) return cap_value(n,t);
    const double l=qr.lo/db.hi, u=db.lo>0.0 ? qr.hi/db.lo : INF;
    const double sumlo=std::max(qb.lo/db.hi,static_cast<double>(n)*l);
    const double sumhi=db.lo>0.0 ? std::min(qb.hi/db.lo,static_cast<double>(n)*u) : INF;
    const double divisor=std::max(db.hi,qr.hi/t);
    double lower=std::max(divisor>0.0 ? qb.lo/divisor : 0.0,static_cast<double>(n)*std::min(l,t));
    double upper=std::min({sumhi,cap,static_cast<double>(n)*std::min(u,t)});
    if (l>=t) lower=cap;
    else if (u<=t) lower=std::max(lower,sumlo);
    else if (std::isfinite(u) && u>l) {
        const double alpha=(t-l)/(u-l);
        const double secant=alpha*sumlo+static_cast<double>(n)*l*(1.0-alpha);
        if (!std::isfinite(secant)) return full_cap(n,t);
        lower=std::max(lower,secant);
    }
    if (!std::isfinite(lower) || !std::isfinite(upper)) return full_cap(n,t);
    auto base=enclose(lower,upper,cap,n+32);
    if (!std::isfinite(base.lo) || !std::isfinite(base.hi) || base.lo>base.hi) return full_cap(n,t);
    base.lo=std::max(0.0,base.lo);base.hi=std::min(cap_ceiling(n,t),base.hi);
    // Q, absolute scale, and denominator intervals are shared with Cantelli.
    const double extra=cantelli(c,e,t,qb.hi,db.hi);
    if (!std::isfinite(extra) || extra>base.hi) return full_cap(n,t);
    base.lo=std::max(base.lo,extra);
    return base;
}

struct BoundStats {
    Interval bound;
    uint64_t residuals=0,nodes=0,refinements=0;
    std::vector<Interval> initial_bounds;
};

BoundStats score_bound(const Tree& tree,const Matrix& e,bool refine,const BoundStats* cached=nullptr) {
    struct Part { int cell; Interval bound; bool active=true; };
    struct Entry { double width; size_t part; };
    struct Compare {
        bool operator()(const Entry& a,const Entry& b) const {
            return a.width!=b.width ? a.width<b.width : a.part>b.part;
        }
    };
    std::vector<Part> parts; parts.reserve(tree.initial.size()+2*MAX_REFINEMENTS);
    std::priority_queue<Entry,std::vector<Entry>,Compare> queue;
    BoundStats out;
    for (size_t k=0;k<tree.initial.size();++k) {
        const int c=tree.initial[k];
        const auto b=cached ? cached->initial_bounds[k] : cell_bound(tree.cells[c],e,tree.data.t);
        if (!cached) {++out.nodes;out.initial_bounds.push_back(b);}
        parts.push_back({c,b,true});
        if (refine) queue.push({b.width(),parts.size()-1});
    }
    auto sum=[&]() {
        long double lo=0.0L,hi=0.0L;size_t count=0;
        for (const auto& p:parts) if (p.active) {lo+=p.bound.lo;hi+=p.bound.hi;++count;}
        const auto b=enclose(static_cast<double>(lo),static_cast<double>(hi),static_cast<double>(tree.data.n)*tree.data.t,count+32);
        return Interval{std::max(0.0,b.lo),std::min(cap_ceiling(tree.data.n,tree.data.t),b.hi)};
    };
    const auto initial=cached ? cached->bound : sum();
    if (!refine) {out.bound=initial;return out;}
    while (!queue.empty() && out.refinements<MAX_REFINEMENTS) {
        const auto next=queue.top();queue.pop();
        const int cell_id=parts[next.part].cell;
        const auto& c=tree.cells[cell_id];
        if (parts[next.part].bound.width()<=4.0*pad(static_cast<double>(c.n())*tree.data.t,c.n()+32)) continue;
        if (c.left>=0) {
            parts[next.part].active=false;
            for (int child:{c.left,c.right}) {
                const auto b=cell_bound(tree.cells[child],e,tree.data.t);
                parts.push_back({child,b,true});queue.push({b.width(),parts.size()-1});++out.nodes;
            }
        } else {
            const double score=tree.exact_cell(e,c);
            auto b=enclose(score,score,static_cast<double>(c.n())*tree.data.t,c.n()+32);
            b.lo=std::max(0.0,b.lo);b.hi=std::min(cap_ceiling(c.n(),tree.data.t),b.hi);
            parts[next.part].bound=b;out.residuals+=c.n();
        }
        ++out.refinements;
    }
    const auto final=sum();
    out.bound={std::max(initial.lo,final.lo),std::min(initial.hi,final.hi)};
    if (out.bound.lo>out.bound.hi) out.bound=initial;
    return out;
}

bool better(double score,size_t index,const NSResult& r) {
    return score<r.score || (score==r.score && index<static_cast<size_t>(r.winner));
}

NSResult select_full(const Input& in) {
    NSResult r{};r.winner=0;r.score=INF;
    for (size_t h=0;h<in.matrices.size();++h) {
        const double score=in.exact(in.matrices[h]);r.residuals+=in.n;
        if (better(score,h,r)) {r.winner=static_cast<int64_t>(h);r.score=score;}
    }
    return r;
}

std::vector<size_t> permutation(const Input& in,const uint64_t* supplied) {
    if (in.n && !supplied) throw std::invalid_argument("partial mode requires a permutation");
    std::vector<size_t> out(in.n);std::vector<unsigned char> seen(in.n,0);
    for (size_t i=0;i<in.n;++i) {
        if (supplied[i]>=in.n || seen[supplied[i]]) throw std::invalid_argument("invalid point permutation");
        out[i]=static_cast<size_t>(supplied[i]);seen[out[i]]=1;
    }
    return out;
}

// Each partial pass counts every consumed point, even if previously scored by
// a refined leaf. Successful passes reduce stored costs in original point order.
bool partial_candidate(const Input& in,size_t h,const std::vector<size_t>& order,
                       std::vector<double>& costs,NSResult& r) {
    const double guard=pad(static_cast<double>(in.n)*in.t,in.n+32);
    long double partial=0.0L;
    for (size_t first=0;first<in.n;first+=BLOCK_SIZE) {
        const size_t last=std::min(first+BLOCK_SIZE,in.n);
        double block=0.0;
        for (size_t j=first;j<last;++j) {
            const size_t p=order[j];costs[p]=in.residual(in.matrices[h],p);block+=costs[p];
        }
        r.residuals+=last-first;partial+=block;
        if (static_cast<double>(partial)>r.score+guard) {++r.pruned;return false;}
    }
    double score=0.0;for (double v:costs) score+=v;
    if (!std::isfinite(score)) throw std::overflow_error("partial score reduction overflows");
    if (better(score,h,r)) {r.winner=static_cast<int64_t>(h);r.score=score;}
    return true;
}

NSResult select_partial(const Input& in,const std::vector<size_t>& order) {
    NSResult r{};r.winner=0;r.score=in.exact(in.matrices[0]);r.residuals=in.n;
    std::vector<double> costs(in.n);
    for (size_t h=1;h<in.matrices.size();++h) partial_candidate(in,h,order,costs,r);
    return r;
}

NSResult select_moments(const Input& in,const Tree& tree,int mode,const std::vector<size_t>& order) {
    NSResult r{};std::vector<BoundStats> coarse;coarse.reserve(in.matrices.size());
    size_t incumbent=0;
    for (size_t h=0;h<in.matrices.size();++h) {
        coarse.push_back(score_bound(tree,in.matrices[h],false));r.nodes+=coarse.back().nodes;
        if (coarse[h].bound.hi<coarse[incumbent].bound.hi) incumbent=h;
    }
    r.winner=static_cast<int64_t>(incumbent);r.score=in.exact(in.matrices[incumbent]);r.residuals=in.n;
    std::vector<size_t> candidates;
    for (size_t h=0;h<in.matrices.size();++h) if (h!=incumbent) candidates.push_back(h);
    std::sort(candidates.begin(),candidates.end(),[&](size_t a,size_t b) {
        return coarse[a].bound.lo!=coarse[b].bound.lo ? coarse[a].bound.lo<coarse[b].bound.lo : a<b;
    });
    const double guard=pad(static_cast<double>(in.n)*in.t,in.n+32);
    std::vector<double> costs(mode==2 ? 0 : in.n);
    for (size_t h:candidates) {
        if (mode==3) {
            // Matched-order control: no rejection or refinement by bounds.
            partial_candidate(in,h,order,costs,r);continue;
        }
        const double safe_upper=r.score+guard;
        if (coarse[h].bound.lo>safe_upper) {++r.pruned;continue;}
        // Reuse exactly the initial per-cell bounds already paid for. Only new
        // child bounds incur node visits; direct leaf costs remain fully counted.
        const auto b=score_bound(tree,in.matrices[h],true,&coarse[h]);
        r.nodes+=b.nodes;r.residuals+=b.residuals;r.refinements+=b.refinements;
        if (b.bound.lo>safe_upper) {++r.pruned;continue;}
        if (mode==4) partial_candidate(in,h,order,costs,r);
        else {
            const double score=in.exact(in.matrices[h]);r.residuals+=in.n;
            if (better(score,h,r)) {r.winner=static_cast<int64_t>(h);r.score=score;}
        }
    }
    return r;
}

void error_text(char* buffer,uint64_t capacity,const char* message) {
    if (buffer && capacity) std::snprintf(buffer,static_cast<size_t>(capacity),"%s",message);
}

template<class Fn> int protect(char* error,uint64_t cap,Fn fn) noexcept {
    try {fn();error_text(error,cap,"");return 0;}
    catch (const std::exception& e) {error_text(error,cap,e.what());return 1;}
    catch (...) {error_text(error,cap,"unknown native exception");return 2;}
}
} // namespace

extern "C" int ns_select(const double* x1,const double* x2,uint64_t n,
    const double* matrices,uint64_t models,double tau2,const uint64_t* order,
    int mode,NSResult* out,char* error,uint64_t error_cap) noexcept {
    return protect(error,error_cap,[&] {
        if (!out) throw std::invalid_argument("output cannot be null");
        const auto started=Clock::now();
        Input input(x1,x2,n,matrices,models,tau2);
        if (mode<0 || mode>4) throw std::invalid_argument("mode must be in 0..4");
        std::vector<size_t> permutation_order;
        if (mode==1 || mode==3 || mode==4) permutation_order=permutation(input,order);
        if (mode>=2) {
            Tree tree(input);const auto ready=Clock::now();
            *out=select_moments(input,tree,mode,permutation_order);const auto done=Clock::now();
            out->setup_seconds=std::chrono::duration<double>(ready-started).count();
            out->scoring_seconds=std::chrono::duration<double>(done-ready).count();
        } else {
            const auto ready=Clock::now();
            *out=mode==0 ? select_full(input) : select_partial(input,permutation_order);
            const auto done=Clock::now();
            out->setup_seconds=std::chrono::duration<double>(ready-started).count();
            out->scoring_seconds=std::chrono::duration<double>(done-ready).count();
        }
    });
}

extern "C" int ns_scores(const double* x1,const double* x2,uint64_t n,
    const double* matrices,uint64_t models,double tau2,double* out,
    char* error,uint64_t error_cap) noexcept {
    return protect(error,error_cap,[&] {
        if (!out) throw std::invalid_argument("score output cannot be null");
        Input input(x1,x2,n,matrices,models,tau2);
        for (size_t h=0;h<input.matrices.size();++h) out[h]=input.exact(input.matrices[h]);
    });
}

extern "C" int ns_validate(const double* x1,const double* x2,uint64_t n,
    const double* matrices,uint64_t models,double tau2,NSValidation* out,
    char* error,uint64_t error_cap) noexcept {
    return protect(error,error_cap,[&] {
        if (!out) throw std::invalid_argument("validation output cannot be null");
        *out={0,0,0,-1,-1,0.0,0.0};
        Input input(x1,x2,n,matrices,models,tau2);Tree tree(input);
        auto check=[&](size_t model,int64_t cell,const Interval& b,double score) {
            if (!(b.lo<=score && score<=b.hi)) {
                ++out->violations;
                if (out->first_model<0) {out->first_model=static_cast<int64_t>(model);out->first_cell=cell;}
                if (b.lo>score) out->max_lower_violation=std::max(out->max_lower_violation,b.lo-score);
                if (b.hi<score) out->max_upper_violation=std::max(out->max_upper_violation,score-b.hi);
            }
        };
        for (size_t h=0;h<input.matrices.size();++h) {
            long double lo=0.0L,hi=0.0L;
            for (size_t k=0;k<tree.initial.size();++k) {
                const auto& c=tree.cells[tree.initial[k]];
                const auto b=cell_bound(c,input.matrices[h],tau2);
                check(h,static_cast<int64_t>(k),b,tree.exact_cell(input.matrices[h],c));
                ++out->cell_checks;lo+=b.lo;hi+=b.hi;
            }
            auto sum=enclose(static_cast<double>(lo),static_cast<double>(hi),static_cast<double>(n)*tau2,tree.initial.size()+32);
            sum.lo=std::max(0.0,sum.lo);sum.hi=std::min(cap_ceiling(n,tau2),sum.hi);
            check(h,-1,sum,input.exact(input.matrices[h]));++out->aggregate_checks;
        }
    });
}
