import numpy as np
import pytest

from ssip.monic_span_checker import is_irreducible_logical


@pytest.mark.parametrize("container", [list, tuple, set, iter])
@pytest.mark.parametrize(
    ("support", "expected"), [([0, 1], True), ([0, 1, 2], False), ([0], False)]
)
def test_irreducible_logical_accepts_iterables(container, support, expected):
    checks = np.array([[1, 1, 0]], dtype=np.uint8)
    assert is_irreducible_logical(container(support), checks) is expected
