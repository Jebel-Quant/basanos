"""Tests for taipan.math.signal utilities (osc, returns_adjust, shrink2id)."""

import numpy as np

from basanos.math._signal import shrink2id


def test_shrink2id_behavior():
    """shrink2id mixes a matrix with identity across lambda values."""
    mat = np.array([[2.0, 1.0], [1.0, 3.0]])

    # lamb=1.0 -> original
    np.testing.assert_allclose(shrink2id(mat, lamb=1.0), mat)

    # lamb=0.0 -> identity
    np.testing.assert_allclose(shrink2id(mat, lamb=0.0), np.eye(2))

    # lamb=0.5 -> halfway between M and I
    np.testing.assert_allclose(shrink2id(mat, lamb=0.5), 0.5 * mat + 0.5 * np.eye(2))
