cdef extern from "fibc.h":
    int fibc(int n)

def fib(n):
    return fibc(n)
