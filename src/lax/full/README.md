# Full example for Lax-Wendroff scheme.

Build it with:

```bash
$ python setup.py build_ext --inplace
```

Show animation with:

```bash
$ python -m lax -s 100 -a 50
```

Use four different runners to get the same animation:

```bash
$ python -m lax -s 100 -a 50 -m python
$ python -m lax -s 100 -a 50 -m numpy
$ python -m lax -s 100 -a 50 -m cython
$ python -m lax -s 100 -a 50 -m c
```

Skip the animation for the end plot:

```bash
$ python -m lax -s 100 -m python
$ python -m lax -s 100 -m numpy
$ python -m lax -s 100 -m cython
$ python -m lax -s 100 -m c
```

Benchmark the four runners:

```bash
$ python -m timeit -s "from lax import core" "core.run(core.march_python)"
10 loops, best of 3: 66.9 msec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_numpy)"
1000 loops, best of 3: 1.98 msec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_cython)"
1000 loops, best of 3: 794 usec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_c)"
1000 loops, best of 3: 515 usec per loop
```
