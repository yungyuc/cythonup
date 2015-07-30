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
