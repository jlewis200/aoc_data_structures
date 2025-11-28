"""
Datastructures collection.
"""

from itertools import product
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

    def add_all(self, others):
        """
        Add multiple other values to self.
        """
        result = self
        for other in others:
            result += other
        return result

    def within_range(self, *ranges):
        """
        Return true if all elements of the VectorTuple are within the provided ranges.
        """
        return all(element in range_ for element, range_ in zip(self, ranges))

    def valid_coord(self, bounds):
        """
        Validate if the coord is within provided bounds.
        """
        ranges = self._get_ranges(bounds)
        return self.within_range(ranges)

    def _get_ranges(self, bounds):
        if bounds is None:
            return None
        if isinstance(bounds, np.ndarray):
            return self._get_ranges_from_ndarray(bounds)
        if isinstance(bounds, int):
            return self._get_ranges_from_int(bounds)
        if isinstance(bounds, tuple):
            if all(map(lambda x: isinstance(x, int), bounds)):
                return self._get_ranges_from_max_tuple(bounds)
            if all(map(lambda x: isinstance(x, tuple), bounds)):
                return self._get_ranges_from_min_max_tuple(bounds)
        raise AssertionError(
            "'bounds' must be one of:  None, np.ndarray, int, tuple(int, int)"
        )

    def _get_ranges_from_ndarray(self, array):
        assert array.ndim == len(self)
        ranges = []
        for idx in range(array.ndim):
            ranges.append(range(array.shape[idx]))
        return ranges

    def _get_ranges_from_int(self, bound):
        return [range(bound) for _ in self]

    def _get_ranges_from_max_tuple(self, bounds):
        assert len(bounds) == len(self)
        return [range(bound) for bound in bounds]

    def _get_ranges_from_min_max_tuple(self, bounds):
        assert len(bounds) == len(self)
        return [range(bound[0], bound[1]) for bound in bounds]

    def _adjacency_deltas(self, include_zero_delta=False):
        offsets = (-1, 0, 1) if include_zero_delta else (-1, 1)
        deltas = []
        for dimension, _ in enumerate(self):
            dimensional_deltas = []
            for offset in offsets:
                delta = [0 for _ in self]
                delta[dimension] += offset
                dimensional_deltas.append(VectorTuple(delta))
            deltas.append(dimensional_deltas)
        return deltas

    def orthogonals(self, bounds):
        """
        Generate orthogonal adjacencies by incrementing/decrementing each
        dimension.

        >>> list(VectorTuple(0, 0).orthogonals(None))
        [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        >>> list(VectorTuple(0, 0, 0).orthogonals(None))
        [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        """
        ranges = self._get_ranges(bounds)
        for dimension in self._adjacency_deltas():
            for delta in dimension:
                next_pos = self + delta
                if ranges is None:
                    yield next_pos
                elif next_pos.within_range(*ranges):
                    yield next_pos

    def diagonals(self, bounds):
        """
        Generate diagonal adjacencies by constructing increment deltas for each
        dimension, and then add each unique combination to self.

        >>> list(VectorTuple(0, 0).diagonals(None))
        [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        >>> list(VectorTuple(0, 0, 0).diagonals(None))
        [
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ]
        """
        ranges = self._get_ranges(bounds)
        for delta_combination in product(*self._adjacency_deltas()):
            next_pos = self.add_all(delta_combination)
            if ranges is None:
                yield next_pos
            elif next_pos.within_range(*ranges):
                yield next_pos

    def adjacencies(self, bounds):
        """
        Generate square/cube/hyper-cube adjacencies.  If the current/self
        VectorTuple represents the central core of a rubik's cube, this
        yields the 26 other elements of the cube:  faces/edges/corners.
        """
        ranges = self._get_ranges(bounds)
        for delta_combination in product(
            *self._adjacency_deltas(include_zero_delta=True)
        ):
            next_pos = self.add_all(delta_combination)
            if next_pos == self:
                continue
            if ranges is None:
                yield next_pos
            elif next_pos.within_range(*ranges):
                yield next_pos

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
