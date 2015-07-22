#cython: overflowcheck=False
from __future__ import print_function
from libcpp.map cimport map
from libcpp.vector cimport vector
import sys
cdef:
    map[int, char] cells
    vector[char] P = [ord(x) for x in open(sys.argv[1], "r").read()]
    int i, p, cnt
    char c
    char PUTC=ord('.'), GETC=ord(','), FWD = ord('>'), BWD=ord('<')
    char INC=ord('+'), DEC=ord('-'), START=ord('['), END=ord(']')

i=p=0
while i < P.size():
    c = P[i]
    p += (c == FWD) - (c == BWD)
    cells[p] += (c == INC) - (c == DEC)
    if c==PUTC:
        print(chr(cells[p]), end="")
    elif c==GETC:
        cells[p] = ord(sys.stdin.read(1))
    elif c==START and not cells[p]: 
        cnt = -1
        while cnt:
            i+=1
            cnt += (P[i]==END) - (P[i]==START)          
    elif c==END and cells[p]:
        cnt = -1
        while  cnt:
            i-=1
            cnt += (P[i]==START) - (P[i]==END)
    i+=1