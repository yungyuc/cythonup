cimport numpy as np

# Initialize NumPy.
np.import_array()

def march_cython(double cfl,
                 np.ndarray[double, ndim=1, mode="c"] sol,
                 np.ndarray[double, ndim=1, mode="c"] soln):
    cdef int itmax = sol.shape[0]-1
    cdef int it = 1
    while it < itmax:
        soln[it] = (1 - cfl*cfl) * sol[it]
        soln[it] += cfl * (cfl+1) / 2 * sol[it-1]
        soln[it] += cfl * (cfl-1) / 2 * sol[it+1]
        it += 1
    sol[:] = soln[:]

# vim: set ff=unix fenc=utf8 ft=pyrex nobomb ai et sw=4 ts=4 tw=79:
