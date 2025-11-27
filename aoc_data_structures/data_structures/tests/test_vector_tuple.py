"""
Test the vector tuple class.
"""

import unittest
import numpy as np
from ..vector_tuple import VectorTuple


class TestVectorTuple(unittest.TestCase):
    """
    Test the vector tuple class.
    """

    def test_add(self):
        """
        Test element-wise addition.
        """
        self.assertEqual(
            VectorTuple(1, 1) + VectorTuple(1, 2),
            VectorTuple(2, 3),
        )

    def test_manhattan_0(self):
        """
        Test manhattan() method with magnitude 0.
        """
        self.assertEqual(
            VectorTuple(0, 0).manhattan(),
            0,
        )

    def test_manhattan_negative(self):
        """
        Test manhattan() method with negative values.
        """
        self.assertEqual(
            VectorTuple(-2, -2).manhattan(),
            4,
        )

    def test_manhattan_positive(self):
        """
        Test manhattan() method with positive values.
        """
        self.assertEqual(
            VectorTuple(2, 2).manhattan(),
            4,
        )

    def test_manhattan_mixed_sign(self):
        """
        Test manhattan() method with positive and negative values.
        """
        self.assertEqual(
            VectorTuple(2, -2).manhattan(),
            4,
        )

    def test_radius_0(self):
        """
        Ensure radius of size 0 yields an empty set of coordinates.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(10, 10).radius(grid, 0))
        self.assertEqual(
            actual,
            set(),
        )

    def test_radius_2(self):
        """
        Ensure radius of size 2 yields appropriate coordinates.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(10, 10).radius(grid, 2))
        expected = {
            VectorTuple(10, 8),
            VectorTuple(10, 9),
            VectorTuple(10, 11),
            VectorTuple(10, 12),
            VectorTuple(8, 10),
            VectorTuple(9, 10),
            VectorTuple(11, 10),
            VectorTuple(12, 10),
            VectorTuple(9, 9),
            VectorTuple(11, 11),
            VectorTuple(11, 9),
            VectorTuple(9, 11),
        }
        self.assertEqual(actual, expected)

    def test_radius_nw_corner(self):
        """
        Ensure radius at north-west corner of grid is truncated.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(0, 0).radius(grid, 2))
        expected = {
            VectorTuple(0, 1),
            VectorTuple(0, 2),
            VectorTuple(1, 0),
            VectorTuple(2, 0),
            VectorTuple(1, 1),
        }
        self.assertEqual(actual, expected)

    def test_radius_ne_corner(self):
        """
        Ensure radius at north-east corner of grid is truncated.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(0, 19).radius(grid, 2))
        expected = {
            VectorTuple(0, 17),
            VectorTuple(0, 18),
            VectorTuple(1, 19),
            VectorTuple(2, 19),
            VectorTuple(1, 18),
        }
        self.assertEqual(actual, expected)

    def test_radius_sw_corner(self):
        """
        Ensure radius at south-west corner of grid is truncated.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(19, 0).radius(grid, 2))
        expected = {
            VectorTuple(19, 1),
            VectorTuple(19, 2),
            VectorTuple(17, 0),
            VectorTuple(18, 0),
            VectorTuple(18, 1),
        }
        self.assertEqual(actual, expected)

    def test_radius_se_corner(self):
        """
        Ensure radius at south-east corner of grid is truncated.
        """
        grid = np.full((20, 20), 0)
        actual = set(VectorTuple(19, 19).radius(grid, 2))
        expected = {
            VectorTuple(19, 17),
            VectorTuple(19, 18),
            VectorTuple(17, 19),
            VectorTuple(18, 19),
            VectorTuple(18, 18),
        }
        self.assertEqual(actual, expected)
