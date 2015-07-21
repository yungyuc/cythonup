#cython: overflowcheck=False
from vector_list cimport vector
cdef:
    vector[char] cells
    vector[char] P
    int i, p, cnt
    char c
    int PUTC, GETC, FWD, BWD, INC, DEC, START, END
