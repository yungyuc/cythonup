#cython: boundscheck=False, infer_types=True, nonecheck=False
import cython

@cython.locals(v="int", K="int")
cdef int estimate(list, int K0)

@cython.locals(i="int", v="int")
cdef int search(list items, int K, int best_v=*, int current_v=*)

