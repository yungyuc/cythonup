# Python manages, C works

Open the interactive notebook:

```
$ ipython notebook buffer.ipynb --ip "*" --no-browser
```

Serve the notebook in slide mode:

```
$ ipython nbconvert buffer.ipynb --to slides --post serve
```

In `01_c_cstruct/` directory:

```
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_int32(10)"
```

In `02_multiple_dimension` directory:

```
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_int32_md((2,3))"
```

In `03_ghost_index` directory:

```
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_parts((5,4),2)"
$ python -c "from ghostbuffer import core; core.print_parts((5,2,4),2)"
```
