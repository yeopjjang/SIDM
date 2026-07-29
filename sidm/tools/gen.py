"""Cycle-safe gen-particle ancestry helpers."""

import numpy
import numba
from coffea.nanoevents import transforms


DISTINCT_PARENT_MAX_STEPS = 1000


@numba.njit
def _capped_distinctParent_kernel(allpart_parent, allpart_pdg):
    """coffea's distinctParent kernel with a cycle guard."""
    n = len(allpart_pdg)
    out = numpy.empty(n, dtype=numpy.int64)
    for i in range(n):
        parent = allpart_parent[i]
        if parent < 0:
            out[i] = -1
            continue

        thispdg = allpart_pdg[i]
        steps = 0
        while parent >= 0 and parent < n and allpart_pdg[parent] == thispdg:
            parent = allpart_parent[parent]
            steps += 1
            if steps >= DISTINCT_PARENT_MAX_STEPS:
                parent = -1
                break

        out[i] = parent if parent < n else -1
    return out


def install():
    """Install the cycle-safe distinctParent kernel process-wide."""
    if not hasattr(transforms, "_distinctParent_kernel"):
        raise RuntimeError(
            "coffea.nanoevents.transforms._distinctParent_kernel not found; "
            "sidm.tools.gen needs updating for this coffea version."
        )
    transforms._distinctParent_kernel = _capped_distinctParent_kernel


install()


def distinct_ancestor(particles, generations=1):
    """Walk .distinctParent a fixed number of generations."""
    cur = particles
    for _ in range(generations):
        cur = cur.distinctParent
    return cur
