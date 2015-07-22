from __future__ import print_function
import sys

# putchar is declared in .pxd, so this line only python interpreter, not cython.
globals()['putchar'] = lambda x: print(chr(x), end="")

cells = [0]*1000
P = [ord(x) for x in open(sys.argv[1], "r").read()]

PUTC, GETC, FWD, BWD, INC, DEC, START, END = tuple(map(ord, ".,><+-[]"))
positions = [0]*P.__len__()
starts = []
cnt = 0
for i in range(P.__len__()):
    if P[i]== START:
        starts.append(i)
    elif P[i] == END:
        j = starts.pop()
        positions[i] = j
        positions[j] = i        

i = p = 0
while i < P.__len__():
    c = P[i]
    if c == FWD:        
        p += 1      
    elif c == BWD:
        p -= 1        
    elif c == INC:
        cells[p] += 1        
    elif c == DEC:
        cells[p] -= 1        
    elif c==PUTC:
        putchar(cells[p])
    elif c==GETC:
        cells[p] = ord(sys.stdin.read(1))
    elif c==START and not cells[p]:
        i = positions[i]
    elif c==END and cells[p]:
        i = positions[i]
    i+=1
