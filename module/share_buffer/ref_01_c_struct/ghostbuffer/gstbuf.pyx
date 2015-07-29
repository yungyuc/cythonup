from __future__ import absolute_import, division, print_function

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

# Initialize NumPy.
np.import_array()


cdef extern:
    void gstbuf_print_int32(gstbuf_t gbuf)


cdef public:
    ctypedef struct gstbuf_t:
        char *elem
        np.npy_intp *shape
        int ndim
        int elemsize


cdef class GhostArray:
    cdef gstbuf_t *_data
    # Overwrite this memory holder may result into segfault with _data.elem.
    cdef readonly object nda

    def __cinit__(self, *args, **kw):
        self._data = <gstbuf_t*>malloc(sizeof(gstbuf_t))
        self._data.elem = <char*>(NULL)
        self._data.shape = <np.npy_intp*>(NULL)
        self._data.ndim = 0
        self._data.elemsize = 0

    def __dealloc__(self):
        if NULL != self._data:
            if NULL != self._data.shape:
                free(self._data.shape)
            free(self._data)

    def __init__(self, *args, **kw):
        # Pop all custom keyword arguments.
        gshape = kw.pop("gshape", None)
        creator_name = kw.pop("creation", "empty")
        # Create the ndarray and thus control the life cycle.
        create = getattr(np, creator_name)
        self.nda = create(*args, **kw)
        if not self.nda.flags.c_contiguous:
            raise ValueError("not C contiguous")
        ndim = len(self.nda.shape)
        if ndim == 0:
            raise ValueError("zero dimension is not allowed")
        assert ndim > 0
        # Initialize internal data.
        ## elem.
        cdef np.ndarray cnda = self.nda
        self._data.elem = <char*>cnda.data
        ## shape (just a duplication of PyArrayObject.dimensions).
        if NULL != self._data.shape:
            free(self._data.shape)
        self._data.shape = <np.npy_intp*>malloc(sizeof(np.npy_intp)*ndim)
        for it in range(ndim):
            self._data.shape[it] = self.nda.shape[it]
        ## ndim (just a duplication of PyArrayObject.nd).
        self._data.ndim = ndim
        ## elemsize (just a duplication of PyArray_Descr.elsize).
        self._data.elemsize = self.nda.itemsize

    def __getattr__(self, name):
        return getattr(self.nda, name)

    @property
    def ndim(self):
        return self._data.ndim

    def print_int32(self):
        assert "int32" == self.nda.dtype
        gstbuf_print_int32((self._data)[0])

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb ai et sw=4 ts=4 tw=79:
