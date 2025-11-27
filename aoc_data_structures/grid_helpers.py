"""
A collection of helpers for common grid-based AOC challenges.
"""

import numpy as np


def expand_grid(grid, expansion_size=2):
    """
    Expand a 2-d numpy array by duplicating elements.

    Example expand by 2:

        original:
            1 0
            0 1

        expanded:
            1 1 0 0
            1 1 0 0
            0 0 1 1
            0 0 1 1
    """
    grid = np.repeat(grid, expansion_size, axis=0)
    return np.repeat(grid, expansion_size, axis=1)


def parse(lines):
    """
    Parse a list of strings representing a grid.  Return as a 2-D numpy array.

    ex input:
    [
        "####",
        "# ##",
        "## #",
        "####",
    ]

    returns:
    [
        ["#", "#", "#", "#"],
        ["#", " ", "#", "#"],
        ["#", "#", " ", "#"],
        ["#", "#", "#", "#"],
    ]
    """
    grid = []

    for line in lines:
        grid.append(list(line.strip()))

    return np.array(grid)


def grid_str(grid):
    """
    Return the string representation of a numpy array where each element can be
    represented as a single character.
    """
    return "\n".join("".join(row) for row in grid)


def hash_array(array):
    """
    Produce a hash based on current contents of a grid/array.
    """
    return hash("".join(array.flatten()))
