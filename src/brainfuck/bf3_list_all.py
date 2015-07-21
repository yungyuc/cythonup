from __future__ import print_function
import sys
cells = [0]*1000
P = open(sys.argv[1], "r").read()+"\x00"
i = p = 0
while i < len(P) and p>=0:
    c = P[i]
    p += (c == '>') - (c == '<')
    if p>=len(cells):
        cells.append(0)
    cells[p] = (cells[p] + (c == '+') - (c == '-'))&0xff
    if c=='.':
        print(chr(cells[p]), end="")
    elif c==',':
        cells[p] = ord(sys.stdin.read(1))
    elif c=='[' and not cells[p]: 
        cnt = -1
        while cnt:
            i+=1
            cnt += (P[i]==']') - (P[i]=='[')	  
    elif c==']' and cells[p]:
        cnt = -1
        while  cnt:
            i-=1
            cnt += (P[i]=='[') - (P[i]==']')	
    i+=1