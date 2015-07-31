# Interaction with C

Open the interactive notebook:

```
$ ipython notebook buffer.ipynb --ip "*" --no-browser
```

Serve the notebook in slide mode:

```
$ ipython nbconvert buffer.ipynb --to slides --post serve
```

In `01_build_cython_and_c/` directory:

```
$ python setup.py build_ext --inplace
$ python -c "import march_c; assert hasattr(march_c, 'march_c')"
```

In `02_make_a_package/` directory:

```
$ python setup.py build_ext --inplace
$ python -m lax.core -h
```

In `03_compare_all/` directory:

```
$ python setup.py build_ext --inplace
$ python -m timeit -s "from lax import core" "core.run(core.march_python)"
10 loops, best of 3: 64.6 msec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_numpy)"
100 loops, best of 3: 2.1 msec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_cython)"
1000 loops, best of 3: 817 usec per loop
$ python -m timeit -s "from lax import core" "core.run(core.march_c)"
1000 loops, best of 3: 495 usec per loop
```
