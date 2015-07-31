#include <Python.h>
#include <stdio.h>

#include <numpy/arrayobject.h>

#include "gstbuf.h"


void gstbuf_print_int32(gstbuf_t gbuf) {
    if (4 != gbuf.elemsize) {
        return;
    }
    int *elem = (int *)gbuf.elem;
    npy_intp jt;
    for (jt=0; jt<gbuf.nelem; jt++) {
        printf("%lu: %d\n", jt, elem[jt]);
    }
}

void gstbuf_print_int32_md(gstbuf_t gbuf) {
    if (4 != gbuf.elemsize) {
        return;
    }
    npy_intp its[gbuf.ndim];
    int it;
    for (it=0; it<gbuf.ndim; ++it) {
        its[it] = 0;
    }
    int *elem = (int *)gbuf.elem;
    npy_intp jt;
    for (jt=0; jt<gbuf.nelem; jt++) {
        for (it=0; it<gbuf.ndim; ++it) {
            printf("%lu ", its[it]);
        }
        printf(": %d\n", elem[jt]);
        ++its[gbuf.ndim-1];
        for (it=gbuf.ndim-1; it>=0; --it) {
            if (its[it] == gbuf.shape[it]) {
                its[it] = 0;
                if (it != 0) { // Carry.
                    ++its[it-1];
                }
            }
        }
    }
}

// vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
