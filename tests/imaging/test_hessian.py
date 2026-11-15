"""Tests of the Hessian kernels and eigenvalues (:mod:`pybones.imaging.hessian`)."""

import numpy as np
import pytest

from pybones.imaging.hessian import (
    gaussian_derivative_kernels,
    shape_measures,
    symmetric_eigenvalues,
)


def test_eigenvalues_match_numpy() -> None:
    """The closed form matches ``numpy.linalg.eigvalsh`` on random symmetric matrices."""
    rng = np.random.default_rng(0)
    m = rng.normal(size=(500, 3, 3))
    m = m + m.transpose(0, 2, 1)
    got = symmetric_eigenvalues(
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]
    )
    expected = np.linalg.eigvalsh(m)[:, ::-1].T  # decreasing
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_eigenvalues_of_isotropic_matrices() -> None:
    """A multiple of the identity (the degenerate case) has three equal eigenvalues."""
    z = np.zeros(3)
    got = symmetric_eigenvalues(np.full(3, 2.0), np.full(3, 2.0), np.full(3, 2.0), z, z, z)
    np.testing.assert_allclose(got, 2.0)


def test_kernels_differentiate_polynomials() -> None:
    """The kernels preserve constants and differentiate ``x`` and ``x²`` (as correlations)."""
    half = 6
    g0, g1, g2 = gaussian_derivative_kernels(1.5, half)
    x = np.arange(-half, half + 1, dtype=np.float64)
    assert g0.sum() == pytest.approx(1.0, abs=1e-6)
    assert float(g1 @ x) == pytest.approx(-1.0, rel=0.02)  # correlation = -d/dx
    assert float(g2 @ (x**2)) == pytest.approx(2.0, rel=0.02)
    assert float(g2.sum()) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("eigenvalues", "tube_ratio"),
    [
        ((0.0, -10.0, -10.0), 1.0),  # tube
        ((0.0, 0.0, -10.0), 0.0),  # plate
        ((-10.0, -10.0, -10.0), 1.0),  # blob: as round as a tube in cross-section
        ((-10.0, 0.0, 0.0), 0.0),  # order does not matter (sorted by magnitude)
    ],
)
def test_tube_ratio_separates_tubes_from_plates(
    eigenvalues: tuple[float, float, float], tube_ratio: float
) -> None:
    """``|l2| / |l3|`` is 1 for a tube and 0 for a plate, whatever the eigenvalue order."""
    shape = shape_measures(np.array(eigenvalues).reshape(3, 1))
    assert float(shape.tube_ratio[0]) == pytest.approx(tube_ratio)
    assert float(shape.structure2[0]) == pytest.approx(sum(v * v for v in eigenvalues))
