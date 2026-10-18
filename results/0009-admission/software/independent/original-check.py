"""Independent bounded enumeration/rejection checks; no geometry or real data."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

path = Path('/tmp/evr-0009-tail-audit/exact_tail.py')
source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('reviewed_tail_candidate', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def independent_rooks(a):
    # Enumerate subsets of rows and injective assignments of columns directly.
    # No graph components, DP, or biclique recognition is used by this oracle.
    n = len(a)
    counts = [0]*(n+1)
    for k in range(n+1):
        for rows in itertools.combinations(range(n),k):
            for cols in itertools.permutations(range(n),k):
                if all(a[i][j] == 1 for i,j in zip(rows,cols)):
                    counts[k] += 1
    return tuple(counts)


def independent_histogram(a):
    n = len(a)
    counts = [0]*(n+1)
    for p in itertools.permutations(range(n)):
        counts[sum(a[i][p[i]] for i in range(n))] += 1
    return counts


started = time.monotonic()
cases = []
for n in range(4):
    for bits in itertools.product([0,1],repeat=n*n):
        cases.append([list(bits[n*i:n*(i+1)]) for i in range(n)])
rng = random.Random(991867)
for n in (4,5,6):
    for case in range(16):
        cutoff = [1,2,4,7][case%4]
        cases.append([[int(rng.randrange(8)<cutoff) for _ in range(n)] for _ in range(n)])

tail_comparisons = rook_vectors = 0
for a in cases:
    n=len(a)
    histogram=independent_histogram(a)
    expected_rooks=independent_rooks(a)
    for k in range(-1,n+2):
        actual=module.exact_permutation_tail(a,k)
        expected=sum(histogram[max(0,k):])
        assert actual.numerator==expected, (a,k,actual.numerator,expected)
        assert actual.denominator==math.factorial(n)
        assert actual.rook_coefficients==expected_rooks
        tail_comparisons+=1
    rook_vectors+=1

# Budget boundary checks enumerate EVERY budget below a successful run, rather
# than only checking the final-minus-one endpoint from the author's tests.
budget_boards=[[],[[0]],[[1]],[[1,0],[0,1]],
               [[1,1,0],[0,1,1],[1,0,1]],
               [[int((i<2 and j<3) or (i>=2 and j>=3)) for j in range(6)] for i in range(6)]]
budget_refusals=budget_exact_successes=zero_work_successes=0
for a in budget_boards:
    k=sum(a[i][i] for i in range(len(a)))
    full=module.exact_permutation_tail(a,k)
    assert module.exact_permutation_tail(a,k,operation_limit=max(1,full.operations))==full
    budget_exact_successes+=int(full.operations>0)
    zero_work_successes+=int(full.operations==0)
    for cap in range(1,full.operations):
        try:
            module.exact_permutation_tail(a,k,operation_limit=cap)
        except module.OperationBudgetExceeded as exc:
            assert exc.operations==cap
            assert not hasattr(exc,'probability') and not hasattr(exc,'numerator')
            budget_refusals+=1
        else:
            raise AssertionError(('budget unexpectedly accepted',a,cap,full.operations))

# Refuse a connected 13x13 non-biclique, including mathematically trivial tails.
# This is a resource-refusal fixture, not an exhaustive n>6 probability check.
large_cycle=[[int(j==i or j==(i+1)%13) for j in range(13)] for i in range(13)]
unsupported=[]
for k in (-1,0,1,13,14):
    try:
        module.exact_permutation_tail(large_cycle,k)
    except module.UnsupportedBoard as exc:
        assert 'smaller side > 12' in exc.reason
        assert not hasattr(exc,'probability')
        unsupported.append({'threshold':k,'reason':exc.reason,'operations':exc.operations})
    else:
        raise AssertionError('large complex component accepted')

assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
report={'status':'passed','source_sha256':source_hash,'python':sys.version.split()[0],
        'scope':'Artificial arithmetic n<=6 exhaustive checks, plus explicit 13x13 resource refusal only; no data, geometry performance, models, or network.',
        'matrix_cases':len(cases),'tail_comparisons':tail_comparisons,
        'independent_rook_vectors':rook_vectors,'budget_refusals':budget_refusals,
        'exact_budget_successes':budget_exact_successes,
        'zero_work_minimum_legal_budget_successes':zero_work_successes,
        'unsupported_checks':unsupported,
        'wall_seconds':time.monotonic()-started}
Path('/tmp/evr-0009-tail-independent-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
