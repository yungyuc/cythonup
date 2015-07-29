from __future__ import absolute_import, division, print_function

from . import gstbuf


def print_int32(num):
    grr = gstbuf.GhostArray(num, creation="arange", dtype="int32")
    grr.print_int32()

# vim: set fenc=utf8 ff=unix nobomb ai et sw=4 ts=4 tw=79:
