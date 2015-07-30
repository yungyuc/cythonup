from __future__ import absolute_import, division, print_function

import numpy as np
from . import gstbuf


def print_int32(num):
    grr = gstbuf.GhostArray(num, creation="arange", dtype="int32")
    grr.print_int32()


def print_int32_md(shape):
    grr = gstbuf.GhostArray(shape, creation="empty", dtype="int32")
    grr.nda[...] = np.arange(grr.nda.size, dtype="int32")[::-1].reshape(shape)
    grr.print_int32()
    print(grr.nda)


def print_parts(shape=(5,4), gshape=2):
    grr = gstbuf.GhostArray(shape, gshape=gshape, dtype="int32")
    grr.ranged_fill()
    print("Do I hold a correct pointer?",
        "Yes" if grr._bodyaddr == grr._ghostaddr + grr.itemsize * grr.offset
        else "No")
    print("The whole array:\n", grr.nda)
    print("The ghost part (first dimension reversed):\n", grr.ghostpart)
    print("The body part:\n", grr.bodypart)

# vim: set fenc=utf8 ff=unix nobomb ai et sw=4 ts=4 tw=79:
