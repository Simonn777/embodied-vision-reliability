"""Independent artificial exact enumeration tests, not geometry performance."""
import itertools
import math
import random
import unittest
from fractions import Fraction

from evr.exact_assignment_tail import (MAX_N, OperationBudgetExceeded, UnsupportedBoard,
                        exact_permutation_tail)


def brute_histogram(board):
    n = len(board)
    histogram = [0] * (n + 1)
    for permutation in itertools.permutations(range(n)):
        count = sum(board[i][j] for i, j in enumerate(permutation))
        histogram[count] += 1
    return histogram


def brute_rooks(board):
    # Directly enumerate chosen row subsets and distinct ordered columns.
    n = len(board)
    result = []
    for k in range(n + 1):
        total = 0
        for rows in itertools.combinations(range(n), k):
            for columns in itertools.permutations(range(n), k):
                if all(board[i][j] for i, j in zip(rows, columns)):
                    total += 1
        result.append(total)
    return tuple(result)


class ExactTailTests(unittest.TestCase):
    boards_checked = 0
    tails_checked = 0
    permutations_checked = 0
    rook_vectors_checked = 0

    def check_board(self, board, check_rooks=False):
        histogram = brute_histogram(board)
        n = len(board)
        self.assertEqual(sum(histogram), math.factorial(n))
        type(self).boards_checked += 1
        type(self).permutations_checked += math.factorial(n)
        previous = math.factorial(n)
        for threshold in range(-1, n + 2):
            result = exact_permutation_tail(board, threshold)
            expected = sum(histogram[max(threshold, 0):]) if threshold <= n else 0
            self.assertEqual(result.numerator, expected, (board, threshold))
            self.assertEqual(result.denominator, math.factorial(n))
            self.assertEqual(result.probability, Fraction(expected, math.factorial(n)))
            self.assertLessEqual(result.numerator, previous)
            previous = result.numerator
            type(self).tails_checked += 1
        if check_rooks:
            self.assertEqual(result.rook_coefficients, brute_rooks(board))
            type(self).rook_vectors_checked += 1

    def test_all_binary_boards_n_zero_through_three(self):
        for n in range(4):
            for flat in itertools.product((0, 1), repeat=n * n):
                board = [list(flat[i*n:(i+1)*n]) for i in range(n)]
                self.check_board(board, True)

    def test_deterministic_random_boards_n_four_through_six(self):
        rng = random.Random(709127)
        for n in (4, 5, 6):
            for case in range(80):
                density = (case % 9 + 1) / 10
                board = [[int(rng.random() < density) for _ in range(n)] for _ in range(n)]
                self.check_board(board, case < 6)

    def test_degeneracies_and_permutation_board(self):
        for n in range(1, 7):
            for board in ([[0]*n for _ in range(n)], [[1]*n for _ in range(n)],
                          [[int(i == j) for j in range(n)] for i in range(n)]):
                self.check_board(board, n <= 4)
        result = exact_permutation_tail([], 0)
        self.assertEqual((result.numerator, result.denominator), (1, 1))
        self.assertEqual(exact_permutation_tail([], 1).numerator, 0)

    def test_components_convolve_rooks_not_independent_hits(self):
        # Two isolated diagonal edges: the two hit indicators coincide, rather
        # than being independent Bernoulli(1/2). Actual P(T>=2)=1/2, not 1/4.
        result = exact_permutation_tail([[1, 0], [0, 1]], 2)
        self.assertEqual(result.rook_coefficients, (1, 2, 1))
        self.assertEqual(result.probability, Fraction(1, 2))
        self.assertEqual(len(result.components), 2)
        board = [[int((i < 2 and j < 3) or (i >= 2 and j >= 3)) for j in range(6)] for i in range(6)]
        self.check_board(board, True)

    def test_complete_rectangles_and_isolated_vertices(self):
        board = [[int(i < 3 and j < 4) for j in range(6)] for i in range(6)]
        result = exact_permutation_tail(board, 3)
        self.assertEqual(result.rook_coefficients, (1, 12, 36, 24, 0, 0, 0))
        self.assertEqual(result.components, ((3, 4, 12, 'complete-biclique'),))
        self.check_board(board, True)

    def test_row_column_permutation_invariance(self):
        board = [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]
        permuted = [[board[i][j] for j in (3, 1, 0, 2)] for i in (2, 0, 3, 1)]
        for threshold in range(6):
            a = exact_permutation_tail(board, threshold)
            b = exact_permutation_tail(permuted, threshold)
            self.assertEqual(a.probability, b.probability)
            self.assertEqual(a.rook_coefficients, b.rook_coefficients)

    def test_invalid_boards_and_thresholds(self):
        for board in (None, '1', [[1, 0]], [[1], [0]], [[2]], [[-1]], [[0.0]], [[float('nan')]], [[object()]]):
            with self.subTest(board=repr(board)):
                with self.assertRaises((ValueError, TypeError)):
                    exact_permutation_tail(board, 1)
        for threshold in (True, 1.0, '1', None):
            with self.assertRaises(TypeError):
                exact_permutation_tail([[1]], threshold)
        self.assertEqual(exact_permutation_tail([[True]], 1).probability, 1)

    def test_operation_budget_is_hard_deterministic_and_explicit(self):
        board = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        full = exact_permutation_tail(board, 2)
        same = exact_permutation_tail(board, 2, operation_limit=full.operations)
        self.assertEqual(same, full)
        with self.assertRaises(OperationBudgetExceeded) as caught:
            exact_permutation_tail(board, 2, operation_limit=full.operations-1)
        self.assertEqual(caught.exception.operations, full.operations-1)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                exact_permutation_tail([], 0, operation_limit=invalid)

    def test_unsupported_component_and_dimension_have_no_probability(self):
        # A connected 13x13 cycle has smaller side 13, and is not complete.
        board = [[int(j == i or j == (i+1) % 13) for j in range(13)] for i in range(13)]
        for threshold in (-1, 2, 14):
            with self.assertRaises(UnsupportedBoard) as caught:
                exact_permutation_tail(board, threshold)
            self.assertIn('smaller side > 12', str(caught.exception))
            self.assertFalse(hasattr(caught.exception, 'probability'))
        with self.assertRaises(UnsupportedBoard):
            exact_permutation_tail([[]] * (MAX_N + 1), 0)

    def test_complete_large_component_closed_form_not_rejected(self):
        result = exact_permutation_tail([[1] * 13 for _ in range(13)], 13)
        self.assertEqual(result.probability, 1)
        self.assertEqual(result.components, ((13, 13, 169, 'complete-biclique'),))

    def test_noncomplete_smaller_side_twelve_boundary(self):
        # K_(12,13) minus one edge, embedded in 13x13 with one isolated row.
        board = [[int(i < 12 and not (i == 0 and j == 0)) for j in range(13)] for i in range(13)]
        result = exact_permutation_tail(board, 12, operation_limit=2_000_000)
        # Twelve active rows all hit unless row 0 maps to the missing column.
        self.assertEqual(result.probability, Fraction(12, 13))
        self.assertEqual(result.components[0][-1], 'subset-dp')
        expected = tuple(math.comb(12, k)*math.comb(13, k)*math.factorial(k)
                         - (math.comb(11, k-1)*math.comb(12, k-1)*math.factorial(k-1) if k else 0)
                         for k in range(13)) + (0,)
        self.assertEqual(result.rook_coefficients, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
