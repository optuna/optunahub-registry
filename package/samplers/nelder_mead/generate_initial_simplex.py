# This code is taken from aiaccel (https://github.com/aistairc/aiaccel) distributed under the MIT license.
#
# MIT License
#
# Copyright (c) 2022 National Institute of Advanced Industrial Science and Technology (AIST), Japan, All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import numpy as np


def generate_initial_simplex(
    dim: int,
    edge: float = 0.5,
    centroid: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Generate an initial simplex with a regular shape.
    """

    assert 0.0 <= centroid <= 1.0, "The centroid must be exists in the unit hypercube. "

    assert (
        0.0 < edge <= max(centroid, 1 - centroid)
    ), f"Maximum edge length is {max(centroid, 1 - centroid)}"

    # Our implementation normalizes the search space to unit hypercube [0, 1]^n.
    bdrys = np.array([[0, 1] for _ in range(dim)])

    # Generate regular simplex.
    initial_simplex = np.zeros((dim + 1, dim))
    b = 0.0
    for i in range(dim):
        c = np.sqrt(1 - b)
        initial_simplex[i][i] = c
        r = ((-1 / dim) - b) / c
        for j in range(i + 1, dim + 1):
            initial_simplex[j][i] = r
        b = b + r**2

    # Rotate the generated initial simplex.
    if rng is not None:
        V = rng.random((dim, dim))
    else:
        V = np.random.random((dim, dim))

    for i in range(dim):
        for j in range(0, i):
            V[i] = V[i] - np.dot(V[i], V[j]) * V[j]
        V[i] = V[i] / (np.sum(V[i] ** 2) ** 0.5)
    for i in range(dim + 1):
        initial_simplex[i] = np.dot(initial_simplex[i], V)

    #  Scale up or down and move the generated initial simplex.
    for i in range(dim + 1):
        initial_simplex[i] = edge * initial_simplex[i]
    Matrix_centroid = initial_simplex.mean(axis=0)
    initial_simplex = initial_simplex + (centroid - Matrix_centroid)

    # Check the condition of the generated initial simplex.
    if check_initial_simplex(initial_simplex, bdrys):
        generate_initial_simplex(dim, edge, centroid)
    y = np.array(initial_simplex)

    return y


def check_initial_simplex(initial_simplex: np.ndarray, bdrys: np.ndarray) -> bool:
    """
    Check whether there is at least one vertex of the generated simplex in the search space.
    """
    dim = len(initial_simplex) - 1
    if dim + 1 > sum([out_of_boundary(vertex, bdrys) for vertex in initial_simplex]):
        return False
    return True


def out_of_boundary(y: np.ndarray, bdrys: np.ndarray) -> bool:
    """
    Check whether the input vertex is in the search space.
    """
    for yi, b in zip(y, bdrys):
        if float(b[0]) <= float(yi) <= float(b[1]):
            pass
        else:
            return True
    return False
