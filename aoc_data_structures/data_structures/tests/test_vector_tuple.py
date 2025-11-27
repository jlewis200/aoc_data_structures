"""
Test the vector tuple class.
"""

import unittest
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