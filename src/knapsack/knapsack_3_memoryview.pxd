#cython: boundscheck=False, infer_types=True, nonecheck=False
import cython
import numpy as np


cdef packed struct _Item:
  int index, value, weight

@cython.locals(v="int", K="int", i="int", item="_Item")
cdef int estimate(_Item[:] items, int K0)

@cython.locals(item="_Item", i="int", v="int")
cdef int _search(_Item[:] items, int K, int best_v=*, int current_v=*)

