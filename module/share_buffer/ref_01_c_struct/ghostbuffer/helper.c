#include <Python.h>
#include <stdio.h>

#include <numpy/arrayobject.h>

#include "gstbuf.h"


void gstbuf_print_int32(gstbuf_t gbuf) {
    if (4 != gbuf.elemsize) {
        return;
    }
    int *elem = (int *)gbuf.elem;
    for (npy_intp jt=0; jt<gbuf.nelem; jt++) {
        printf("%lu: %d\n", jt, elem[jt]);
    }
}

// vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
