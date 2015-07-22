from __future__ import print_function
import sys

cells = [0]*1000
P = [ord(x) for x in open(sys.argv[1], "r").read()]
PUTC, GETC, FWD, BWD, INC, DEC, START, END = tuple(map(ord, ".,><+-[]"))

i = p = 0
while i < P.__len__() and p>=0:
    c = P[i]
    p += (c == FWD) - (c == BWD)
    if p > cells.__len__():
        cells.append(0)
    cells[p] = (cells[p] + (c == INC) - (c == DEC))
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