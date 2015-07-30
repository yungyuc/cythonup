# Python manages, C works

In `01_c_cstruct/` directory:

```bash
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_int32(10)"
```

In `02_multiple_dimension` directory:

```bash
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_int32_md((2,3))"
```

In `03_ghost_index` directory:

```bash
$ python setup.py build_ext --inplace
$ python -c "from ghostbuffer import core; core.print_parts(shape=(5,4))"
$ python -c "from ghostbuffer import core; core.print_parts(shape=(5,2,4))"
```
