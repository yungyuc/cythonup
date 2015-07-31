cimport numpy as np
cimport cython

# Initialize NumPy.
np.import_array()

cdef extern:
    void _march_c(double cfl, int nsol, double *sol, double *soln)

@cython.boundscheck(False)
def march_c(double cfl,
            np.ndarray[double, ndim=1, mode="c"] sol,
            np.ndarray[double, ndim=1, mode="c"] soln):
    _march_c(cfl, sol.shape[0], &sol[0], &soln[0])

# vim: set ff=unix fenc=utf8 ft=pyrex nobomb ai et sw=4 ts=4 tw=79:
