#cython: overflowcheck=False
from libc.stdio cimport putchar
from vector_list cimport vector
cdef:
    vector[char] cells
    vector[char] P
    vector[int] positions
    int i, j, p, cnt
    char c
    char PUTC, GETC, FWD, BWD, INC, DEC, START, END
