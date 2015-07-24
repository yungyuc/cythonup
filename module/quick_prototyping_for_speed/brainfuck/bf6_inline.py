from __future__ import print_function
import sys

OPTIMIZATION = True
CHECK_BOUNDS = False

cython_start = """
#cython: overflowcheck=False, boundscheck=False, wraparound=False, nonecheck=False
cdef extern from "stdio.h" nogil:
    int putchar (int c)
cdef:
    char cells[1000]
    int p, cnt
import sys
cells = [0]*1000
p = 0
"""
prog_lines = [cython_start]
P = [ord(x) for x in open(sys.argv[1], "r").read()]
PUTC, GETC, FWD, BWD, INC, DEC, START, END = tuple(map(ord, ".,><+-[]"))
indent = ""

def add(s):
    prog_lines.append(indent+s)
    
for c in P:
    if c == PUTC:
        add("putchar(cells[p])")
    elif c == GETC:
        add("cells[p] = ord(sys.stdin.read(1))")
    elif c == FWD:
        if OPTIMIZATION and prog_lines[-1][:len(indent)+5]==indent+"p += ":
            n = int(prog_lines[-1][len(indent)+5:]) + 1
            prog_lines[-1] = indent+"p += %d"%n
        else:
            add("p += 1")
        if CHECK_BOUNDS:
            add("while p > cells.__len__(): cells.append(0)")
    elif c == BWD:
        if OPTIMIZATION and prog_lines[-1][:len(indent)+5]==indent+"p -= ":
            n = int(prog_lines[-1][len(indent)+5:]) + 1
            prog_lines[-1] = indent+"p -= %d"%n
        else:
            add("p -= 1")
        if CHECK_BOUNDS:
            add("if p < 0: sys.exit(1)")
    elif c == INC:
        if OPTIMIZATION and prog_lines[-1][:len(indent)+12]==indent+"cells[p] += ":
            n = int(prog_lines[-1][len(indent)+12:]) + 1
            prog_lines[-1] = indent+"cells[p] += %d"%n
        else:            
            add("cells[p] += 1")
    elif c == DEC:
        if OPTIMIZATION and prog_lines[-1][:len(indent)+12]==indent+"cells[p] -= ":
            n = int(prog_lines[-1][len(indent)+12:]) + 1
            prog_lines[-1] = indent+"cells[p] -= %d"%n
        else:            
            add("cells[p] -= 1")
    elif c == START:
        add("while cells[p]:")
        indent += "  "
    elif c == END:        
        indent = indent[:-2]

import cython
cython.inline("\n".join(prog_lines))
