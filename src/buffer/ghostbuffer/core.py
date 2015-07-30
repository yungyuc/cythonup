from __future__ import absolute_import, division, print_function

from . import gstbuf


def print_int32(num):
    grr = gstbuf.GhostArray(num, creation="arange", dtype="int32")
    grr.print_int32()


def print_int32_md(shape):
    grr = gstbuf.GhostArray(shape, creation="empty", dtype="int32")
    grr.nda[...] = np.arange(grr.nda.size, dtype="int32")[::-1].reshape(shape)
    grr.print_int32()
    print(grr.nda)

# vim: set fenc=utf8 ff=unix nobomb ai et sw=4 ts=4 tw=79:
