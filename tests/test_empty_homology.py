from itertools import product

import numpy as np
import pytest

from ssip.basic_functions import (
    CSScode,
    find_X_basis,
    find_Z_basis,
    flip_code,
    is_valid_CSScode,
    num_logical_qubits,
)


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("zero_rows", [0, 1])
def test_one_sided_repetition_code_logicals(n_qubits, flipped, zero_rows):
    checks = np.zeros((n_qubits - 1, n_qubits), dtype=np.uint8)
    for i in range(n_qubits - 1):
        checks[i, i : i + 2] = 1
    code = CSScode(checks, np.zeros((zero_rows, n_qubits), dtype=np.uint8))
    if flipped:
        code = flip_code(code)

    assert is_valid_CSScode(code)
    assert num_logical_qubits(code) == 1
    z_basis, x_basis = find_Z_basis(code), find_X_basis(code)
    assert len(z_basis) == len(x_basis) == 1
    assert not np.any(code.PX @ np.asarray(z_basis).T % 2)
    assert not np.any(code.PZ @ np.asarray(x_basis).T % 2)
    assert np.dot(z_basis[0], x_basis[0]) % 2 == 1


@pytest.mark.parametrize("n_qubits", [0, 1, 3])
@pytest.mark.parametrize("flipped", [False, True])
def test_fully_stabilized_code_has_empty_logical_bases(n_qubits, flipped):
    code = CSScode(
        np.eye(n_qubits, dtype=np.uint8), np.zeros((0, n_qubits), dtype=np.uint8)
    )
    if flipped:
        code = flip_code(code)
    assert is_valid_CSScode(code)
    assert num_logical_qubits(code) == 0
    assert find_Z_basis(code) == []
    assert find_X_basis(code) == []


@pytest.mark.parametrize("n_qubits", [1, 3])
@pytest.mark.parametrize("zero_rows", [0, 1])
def test_unstabilized_code_spans_the_full_logical_space(n_qubits, zero_rows):
    checks = np.zeros((zero_rows, n_qubits), dtype=np.uint8)
    code = CSScode(checks, checks.copy())
    assert num_logical_qubits(code) == n_qubits
    for basis in (find_Z_basis(code), find_X_basis(code)):
        assert len(basis) == n_qubits
        coefficients = list(product((0, 1), repeat=n_qubits))
        spanned = {tuple(np.asarray(c) @ np.asarray(basis) % 2) for c in coefficients}
        assert spanned == set(coefficients)
