"""Exact binary-board permutation tails; artificial correctness prototype only.

No third-party packages, images, geometry estimation, or floating point.
Counts are unreduced integer permutation counts, not calibrated real-match p-values.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial

MAX_N = 512
SMALL_SIDE_LIMIT = 12
DEFAULT_OPERATION_LIMIT = 2_000_000


class UnsupportedBoard(RuntimeError):
    """No probability is produced for this computationally unsupported board."""

    def __init__(self, reason, operations):
        super().__init__(reason)
        self.reason = reason
        self.operations = operations


class OperationBudgetExceeded(UnsupportedBoard):
    pass


class _Budget:
    def __init__(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("operation_limit must be a positive built-in int")
        self.limit = limit
        self.used = 0

    def tick(self):
        # Charge BEFORE each defined unit of work. Refusal consumes no extra unit.
        if self.used >= self.limit:
            raise OperationBudgetExceeded("operation budget exhausted", self.used)
        self.used += 1


@dataclass(frozen=True)
class TailResult:
    n: int
    threshold: int
    numerator: int
    denominator: int
    rook_coefficients: tuple
    components: tuple
    operations: int

    @property
    def probability(self):
        return Fraction(self.numerator, self.denominator)


def _adjacency(board, budget):
    if type(board) not in (tuple, list):
        raise TypeError("board must be a built-in list or tuple of rows")
    n = len(board)
    if n > MAX_N:
        raise UnsupportedBoard("n exceeds fixed MAX_N=512", budget.used)
    rows = [[] for _ in range(n)]
    cols = [[] for _ in range(n)]
    for i, row in enumerate(board):
        budget.tick()
        if type(row) not in (tuple, list) or len(row) != n:
            raise ValueError("board must be square with built-in list/tuple rows")
        for j, value in enumerate(row):
            budget.tick()
            if type(value) not in (int, bool) or value not in (0, 1):
                raise ValueError("entries must be built-in binary int/bool")
            if value:
                rows[i].append(j)
                cols[j].append(i)
    return n, rows, cols


def _components(rows, cols, budget):
    visited_rows, visited_cols = set(), set()
    result = []
    for start in range(len(rows)):
        budget.tick()
        if start in visited_rows or not rows[start]:
            continue
        # Mark when enqueued, so each vertex is stored only once.
        stack = [(0, start)]
        visited_rows.add(start)
        rr, cc = [], []
        while stack:
            budget.tick()
            side, vertex = stack.pop()
            if side == 0:
                rr.append(vertex)
                for j in rows[vertex]:
                    budget.tick()
                    if j not in visited_cols:
                        visited_cols.add(j)
                        stack.append((1, j))
            else:
                cc.append(vertex)
                for i in cols[vertex]:
                    budget.tick()
                    if i not in visited_rows:
                        visited_rows.add(i)
                        stack.append((0, i))
        # Order affects only deterministic operation accounting, not the polynomial.
        rr.sort()
        cc.sort()
        edge_count = 0
        for i in rr:
            budget.tick()
            edge_count += len(rows[i])
        result.append((rr, cc, edge_count))
    return result


def _component_rook(rr, cc, edges, rows, cols, budget):
    a, b = len(rr), len(cc)
    small = min(a, b)
    if edges == a * b:
        poly = []
        for k in range(small + 1):
            budget.tick()
            poly.append(comb(a, k) * comb(b, k) * factorial(k))
        return poly, "complete-biclique"
    if small > SMALL_SIDE_LIMIT:
        raise UnsupportedBoard(
            "non-complete component has smaller side > 12: %d x %d" % (a, b),
            budget.used,
        )
    if a <= b:
        small_vertices, large_vertices, adjacency = rr, cc, cols
    else:
        small_vertices, large_vertices, adjacency = cc, rr, rows
    index = {v: k for k, v in enumerate(small_vertices)}
    states = {0: 1}
    for vertex in large_vertices:
        budget.tick()
        neighbors = 0
        for v in adjacency[vertex]:
            budget.tick()
            neighbors |= 1 << index[v]
        new = {}
        for mask, ways in states.items():
            budget.tick()
            new[mask] = new.get(mask, 0) + ways  # Leave this large-side vertex unused.
            available = neighbors & ~mask
            while available:
                budget.tick()
                bit = available & -available
                available -= bit
                updated = mask | bit
                new[updated] = new.get(updated, 0) + ways
        states = new
    poly = [0] * (small + 1)
    for mask, ways in states.items():
        budget.tick()
        poly[bin(mask).count("1")] += ways
    return poly, "subset-dp"


def exact_permutation_tail(board, threshold, *, operation_limit=DEFAULT_OPERATION_LIMIT):
    """Return P(sum_i A[i, pi(i)] >= threshold) for pi uniform on S_n.

    Supported boards: n<=512 and each edge-containing component either complete
    bipartite or with its smaller side <=12, within the operation budget.
    Support checking is threshold-independent, including trivial thresholds.
    Invalid input raises TypeError/ValueError. Resource refusal raises
    UnsupportedBoard (possibly OperationBudgetExceeded); no fallback p-value.
    """
    if type(threshold) is not int:
        raise TypeError("threshold must be a built-in int, excluding bool")
    budget = _Budget(operation_limit)
    n, rows, cols = _adjacency(board, budget)
    components = _components(rows, cols, budget)
    total = [1]
    metadata = []
    for rr, cc, edges in components:
        local, algorithm = _component_rook(rr, cc, edges, rows, cols, budget)
        product = [0] * (len(total) + len(local) - 1)
        for i, x in enumerate(total):
            for j, y in enumerate(local):
                budget.tick()
                product[i + j] += x * y
        total = product
        metadata.append((len(rr), len(cc), edges, algorithm))
    total.extend([0] * (n + 1 - len(total)))
    facts = [1]
    for k in range(1, n + 1):
        budget.tick()
        facts.append(facts[-1] * k)
    denominator = facts[n]
    if threshold <= 0:
        numerator = denominator
    elif threshold > n:
        numerator = 0
    else:
        numerator = 0
        for j in range(threshold, n + 1):
            budget.tick()
            term = comb(j - 1, threshold - 1) * total[j] * facts[n - j]
            numerator += term if (j - threshold) % 2 == 0 else -term
    if not 0 <= numerator <= denominator:
        raise ArithmeticError("integer tail outside its exact count range")
    return TailResult(n, threshold, numerator, denominator, tuple(total),
                      tuple(metadata), budget.used)
