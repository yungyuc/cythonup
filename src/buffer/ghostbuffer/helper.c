#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct {
    char *elem;
    npy_intp *drange;
    int ndim;
    int elemsize;
} gstbuf_t;

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
    for (int it=1; it<gbuf.ndim; it++)
        nelem *= gbuf.drange[it*2] + gbuf.drange[it*2+1];
    for (long it=-1; it>=-nelem; it--)
        elem[it] = it;
    // Body part.
    nelem = gbuf.drange[1];
    for (int it=1; it<gbuf.ndim; it++)
        nelem *= gbuf.drange[it*2] + gbuf.drange[it*2+1];
    for (long it=0; it<nelem; it++)
        elem[it] = it;
}

// vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
