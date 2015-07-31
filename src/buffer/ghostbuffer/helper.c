#include <Python.h>
#include <stdio.h>

#include <numpy/arrayobject.h>

#include "gstbuf.h"


/**
 * Assume gbuf has no ghost.
 */
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

/**
 * Assume gbuf has no ghost.
 */
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

/**
 * Assume gbuf is valid and separable.
 */
void gstbuf_ranged_fill(gstbuf_t gbuf) {
    size_t nelem;
    int *elem = (int *)gbuf.elem;
    if (4 != gbuf.elemsize)
        return; // Do nothing.
    // Ghost part.
    nelem = gbuf.drange[0];
    int it;
    for (it=1; it<gbuf.ndim; it++)
        nelem *= gbuf.drange[it*2] + gbuf.drange[it*2+1];
    long jt;
    for (jt=-1; jt>=-nelem; jt--)
        elem[jt] = jt;
    // Body part.
    nelem = gbuf.drange[1];
    for (it=1; it<gbuf.ndim; it++)
        nelem *= gbuf.drange[it*2] + gbuf.drange[it*2+1];
    for (jt=0; jt<nelem; jt++)
        elem[jt] = jt;
}

// vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
