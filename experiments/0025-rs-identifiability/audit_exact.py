#!/usr/bin/env python3
"""Audit a saved rational matrix using integer Bareiss, without its generator."""
import argparse
from fractions import Fraction
from functools import reduce
import hashlib
import json
from math import gcd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mathematics', type=Path)
    args = parser.parse_args()
    raw = args.mathematics.read_bytes()
    result = json.loads(raw)
    matrix = [[Fraction(value) for value in row] for row in result['matrix']]
    assert len(matrix) == 12 and all(len(row) == 12 for row in matrix)
    scales = [reduce(lambda a, b: a // gcd(a, b) * b,
                     [value.denominator for value in row], 1) for row in matrix]
    integers = [[int(value * scale) for value in row]
                for row, scale in zip(matrix, scales)]
    previous, sign, divisions = 1, 1, 0
    for k in range(len(integers) - 1):
        if integers[k][k] == 0:
            swap = next(i for i in range(k + 1, len(integers)) if integers[i][k])
            integers[k], integers[swap] = integers[swap], integers[k]
            sign = -sign
        pivot = integers[k][k]
        for i in range(k + 1, len(integers)):
            for j in range(k + 1, len(integers)):
                numerator = integers[i][j] * pivot - integers[i][k] * integers[k][j]
                quotient, remainder = divmod(numerator, previous)
                assert remainder == 0
                integers[i][j] = quotient
                divisions += 1
            integers[i][k] = 0
        previous = pivot
    scale_product = reduce(lambda a, b: a * b, scales, 1)
    determinant = Fraction(sign * integers[-1][-1], scale_product)
    assert determinant != 0 and determinant == Fraction(result['determinant'])
    solution = [Fraction(value) for value in result['solved_solution']]
    assert solution == [Fraction(value) for value in result['truth_solution']]
    assert all(sum(value * x for value, x in zip(row, solution)) == Fraction(rhs)
               for row, rhs in zip(matrix, result['rhs']))
    print(json.dumps({'status': 'passed', 'method': 'integer Bareiss',
                      'input_sha256': hashlib.sha256(raw).hexdigest(),
                      'exact_divisions': divisions, 'determinant': str(determinant),
                      'nonzero': True, 'exact_solution_residual_zero': True}))


if __name__ == '__main__':
    main()
