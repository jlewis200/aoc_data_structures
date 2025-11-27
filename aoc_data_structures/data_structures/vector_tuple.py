"""
Datastructures collection.
"""

from itertools import product, chain
import numpy as np


class VectorTuple(tuple):
    """
    This class replicates vectorized operations of numpy arrays, with the
    advantage that it's hashable.
    """

    def __new__(cls, *args):
        if len(args) == 1 and not isinstance(args[0], tuple):
            args = args[0]
        return tuple.__new__(VectorTuple, args)

    def __add__(self, other):
        return VectorTuple(
            self_element + other_element
            for self_element, other_element in zip(self, other)
        )

    def __sub__(self, other):
        return VectorTuple(
            self_element - other_element
            for self_element, other_element in zip(self, other)
        )

    def __mul__(self, other):
        return VectorTuple(
            self_element * other_element
            for self_element, other_element in zip(self, other)
        )

    def __truediv__(self, other):
        return VectorTuple(
            self_element / other_element
            for self_element, other_element in zip(self, other)
        )

    def __mod__(self, other):
        return VectorTuple(
            self_element % other_element
            for self_element, other_element in zip(self, other)
        )

    def __abs__(self):
        return VectorTuple((abs(element) for element in self))

    def within_range(self, *ranges):
        """
        Return true if all elements of the VectorTuple are within the provided ranges.
        """
        return all(element in range_ for element, range_ in zip(self, ranges))

    @staticmethod
    def _get_ranges(bounds):
        if bounds is None:
            return None, None
        if isinstance(bounds, np.ndarray):
            return range(bounds.shape[0]), range(bounds.shape[1])
        if isinstance(bounds, int):
            return range(bounds), range(bounds)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            if all(map(lambda x: isinstance(x, int), bounds)):
                return range(bounds[0]), range(bounds[1])
        raise AssertionError(
            "'bounds' must be one of:  None, np.ndarray, int, tuple(int, int)"
        )

    def orthogonals(self, bounds):
        """
        Generate E, N, W, S adjacencies.
        """
        vertical_range, horizontal_range = self._get_ranges(bounds)
        for delta in (
            VectorTuple(0, 1),
            VectorTuple(-1, 0),
            VectorTuple(0, -1),
            VectorTuple(1, 0),
        ):
            next_pos = self + delta
            if bounds is None:
                yield next_pos
            elif next_pos.within_range(vertical_range, horizontal_range):
                yield next_pos

    def diagonals(self, bounds):
        """
        Generate NE, NW, SW, SE adjacencies.
        """
        vertical_range, horizontal_range = self._get_ranges(bounds)
        for delta in (
            VectorTuple(-1, 1),
            VectorTuple(-1, -1),
            VectorTuple(1, -1),
            VectorTuple(1, 1),
        ):
            next_pos = self + delta
            if bounds is None:
                yield next_pos
            elif next_pos.within_range(vertical_range, horizontal_range):
                yield next_pos

    def adjacencies(self, bounds):
        """
        Generate orthogonal and diagonal adjacencies.
        """
        yield from chain(self.orthogonals(bounds), self.diagonals(bounds))

    def radius(self, grid, size):
        """
        Generate coordinates within a manhattan radius.
        """
        for dx, dy in product(range(-size, size + 1), repeat=2):
            delta = VectorTuple(dy, dx)
            adjacency = self + delta

            if (
                delta == VectorTuple(0, 0)
                or delta.manhattan() > size
                or not adjacency.within_range(
                    range(grid.shape[0]),
                    range(grid.shape[1]),
                )
            ):
                continue

            yield adjacency

    def manhattan(self):
        """
        Get manhattan magnitude of self.
        """
        return sum(abs(self))
