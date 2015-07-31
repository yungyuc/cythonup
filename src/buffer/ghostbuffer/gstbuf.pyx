# Copyright (c) 2015, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the software nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import, division, print_function

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

# Initialize NumPy.
np.import_array()

DEF GSTBUF_DEBUG_PRINT = 0


cdef extern:
    void gstbuf_print_int32(gstbuf_t gbuf)
    void gstbuf_print_int32_md(gstbuf_t gbuf)
    void gstbuf_ranged_fill(gstbuf_t gbuf)


cdef public:
    ctypedef struct gstbuf_t:
        char *elem
        np.npy_intp *shape
        # Dimension range: each dimension has a (nghost,nbody) pair.
        np.npy_intp *drange
        np.npy_intp nelem
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
        self._data.drange = <np.npy_intp*>(NULL)
        self._data.ndim = 0
        self._data.elemsize = 0

    def __dealloc__(self):
        if NULL != self._data:
            if NULL != self._data.shape:
                free(self._data.shape)
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
        if not self.nda.flags.c_contiguous:
            raise ValueError("not C contiguous")
        ndim = len(self.nda.shape)
        if ndim == 0:
            raise ValueError("zero dimension is not allowed")
        assert ndim > 0
        # Initialize internal data.
        drange = self._calc_drange(self.nda.shape, gshape)
        ## elem.
        cdef np.ndarray cnda = self.nda
        cdef char *ndhead = <char*>cnda.data
        offset = self._calc_offset(drange)
        ndhead += self.nda.itemsize * offset
        self._data.elem = ndhead
        ## shape (just a duplication of PyArrayObject.dimensions).
        if NULL != self._data.shape:
            free(self._data.shape)
        self._data.shape = <np.npy_intp*>malloc(sizeof(np.npy_intp)*ndim)
        for it in range(ndim):
            self._data.shape[it] = self.nda.shape[it]
        ## drange.
        if NULL != self._data.drange:
            free(self._data.drange)
        self._data.drange = <np.npy_intp*>malloc(sizeof(np.npy_intp)*ndim*2)
        for it, (nghost, nbody) in enumerate(drange):
            self._data.drange[it*2  ] = nghost
            self._data.drange[it*2+1] = nbody
        ## nelem.
        self._data.nelem = self.nda.size
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
    def nelem(self):
        return self._data.nelem

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
        return self.nda[self.gshape[0]-1::-1,...]

    @property
    def bodypart(self):
        if not self.is_separable:
            raise ValueError("malformed ghost shape")
        return self.nda[self.gshape[0]:,...]

    def print_int32(self):
        assert self._ghostaddr == self._bodyaddr
        assert "int32" == self.nda.dtype
        gstbuf_print_int32((self._data)[0])

    def print_int32_md(self):
        assert self._ghostaddr == self._bodyaddr
        assert "int32" == self.nda.dtype
        gstbuf_print_int32_md((self._data)[0])

    def ranged_fill(self):
        assert self.is_separable
        assert "int32" == self.nda.dtype
        gstbuf_ranged_fill((self._data)[0])

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb ai et sw=4 ts=4 tw=79:
