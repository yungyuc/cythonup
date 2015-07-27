from __future__ import absolute_import, division, print_function

DEF GSTBUF_DEBUG_PRINT = 0

import numpy as np
cimport numpy as np

# Initialize NumPy.
np.import_array()


cdef extern from "stdlib.h":
    void* malloc(size_t size)
    void free(void* ptr)


cdef extern:
    void gstbuf_ranged_fill(gstbuf_t gbuf)


ctypedef struct gstbuf_t:
    char *elem
    # Dimension range: each dimension has a (nghost,nbody) pair.
    np.npy_intp *drange
    int ndim
    int elemsize


cdef class GhostArray:
    cdef gstbuf_t *_data
    # Overwrite this memory holder may result into segfault with _data.elem.
    cdef readonly object nda

    def __cinit__(self, *args, **kw):
        self._data = <gstbuf_t*>malloc(sizeof(gstbuf_t))
        self._data.elem = <char*>(NULL)
        self._data.drange = <np.npy_intp*>(NULL)
        self._data.ndim = 0
        self._data.elemsize = 0

    def __dealloc__(self):
        if NULL != self._data:
            if NULL != self._data.drange:
                free(self._data.drange)
            free(self._data)

    def __init__(self, *args, **kw):
        # Pop all custom keyword arguments.
        gshape = kw.pop("gshape", None)
        creator_name = kw.pop("creation", "empty")
        # Create the ndarray and thus control the life cycle.
        create = getattr(np, creator_name)
        self.nda = create(*args, **kw)
        ndim = len(self.nda.shape)
        if ndim == 0:
            raise ValueError("zero dimension is not allowed")
        assert ndim > 0
        # Initialize internal data.
        drange = self._calc_drange(self.nda.shape, gshape)
        offset = self._calc_offset(drange)
        ## elem.
        cdef np.ndarray cnda = self.nda
        cdef char *ndhead = <char*>cnda.data
        ndhead += self.nda.itemsize * offset
        self._data.elem = ndhead
        ## drange.
        self._data.drange = <np.npy_intp*>malloc(sizeof(np.npy_intp)*ndim*2)
        for it, (nghost, nbody) in enumerate(drange):
            self._data.drange[it*2  ] = nghost
            self._data.drange[it*2+1] = nbody
        ## ndim (just a duplication of PyArrayObject.nd).
        self._data.ndim = ndim
        ## elemsize (just a duplication of PyArray_Descr.elsize).
        self._data.elemsize = self.nda.itemsize
        IF GSTBUF_DEBUG_PRINT:
            print("GSTBUF: Initialized")

    @staticmethod
    def _calc_drange(shape, gshape):
        ndim = len(shape)
        # Make sure gshape is a list.
        if isinstance(gshape, int):
            gshape = [gshape]
        elif None is gshape:
            gshape = [0] * ndim
        gshape = list(gshape)
        # Get gshape correct length.
        if len(gshape) < ndim:
            gshape += [0] * (ndim - len(gshape))
        elif len(gshape) > ndim:
            gshape = gshape[:ndim]
        # Make sure gshape is all positive.
        gshape = [max(0, ng) for ng in gshape]
        # Make sure elements in gshape doesn't exceed those in shape.
        gshape = [min(ng, na) for (ng, na) in zip(gshape, shape)]
        # Build drange.
        drange = [(ng, na-ng) for ng, na in zip(gshape, shape)]
        # Return.
        IF GSTBUF_DEBUG_PRINT:
            print("GSTBUF: drange =", drange)
        return drange

    @staticmethod
    def _calc_offset(drange):
        shape = [ng+nb for (ng, nb) in drange]
        ndim = len(drange)
        offset = 0
        for it, (nghost, nbody) in enumerate(drange):
            if ndim-1 == it:
                trail = 1
            else:
                trail = np.multiply.reduce(shape[it+1:])
            offset += nghost * trail
        IF GSTBUF_DEBUG_PRINT:
            print("GSTBUF: offset =", offset)
        return offset

    def __getattr__(self, name):
        return getattr(self.nda, name)

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def drange(self):
        return tuple((self._data.drange[it*2], self._data.drange[it*2+1])
            for it in range(self._data.ndim))

    @property
    def gshape(self):
        return tuple(self._data.drange[it*2] for it in range(self._data.ndim))

    @property
    def bshape(self):
        return tuple(self._data.drange[it*2+1] for it in range(self._data.ndim))

    @property
    def offset(self):
        return self._calc_offset(self.drange)

    @property
    def _ghostaddr(self):
        cdef np.ndarray cnda = self.nda
        return <unsigned long>(cnda.data)

    @property
    def _bodyaddr(self):
        return <unsigned long>(self._data.elem)

    @property
    def is_separable(self):
        cdef int ndim = self._data.ndim
        if ndim == 1:
            return True
        cdef int it = 1
        while it < ndim:
            if self._data.drange[it*2] != 0:
                return False
            it += 1
        return True

    @property
    def ghostpart(self):
        if not self.is_separable:
            raise ValueError("malformed ghost shape")
        return self.nda[:self.gshape[0],...]

    @property
    def bodypart(self):
        if not self.is_separable:
            raise ValueError("malformed ghost shape")
        return self.nda[self.gshape[0]:,...]

    def ranged_fill(self):
        assert self.is_separable
        gstbuf_ranged_fill((self._data)[0])


# vim: set fenc=utf8 ft=pyrex ff=unix nobomb ai et sw=4 ts=4 tw=79:
