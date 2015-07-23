import cython
cython.declare(a='long', i='long', b='double', A='object', x='long')
a=100
b=30.5

for i in range(10):
    x = i + 1

@cython.locals(d='str', g='double', f='object', i='long', z='long')
def f(x):
    y, z = x, 3
    g = 5.0
    f = [0]*3
    d = "aaaaaa"
    for i in range(5):
        h = "aaaaa"+g
        z = z  + i * x
        y = y + x 
        print(i+z)

@cython.locals(a='str', b='long')
def func(a, b):
    print(a)
    a = 1
    b += a
    a = 'abc'
    return a, str(b)

class A:
    @cython.locals(i='long')
    def f(i):
        i = 3