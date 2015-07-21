# Brainfuck interpreter example

author: tjwei


This directory contains a few Brainfuck interpreter implementations in python/cython.

## interpreters
Most of .py files works for compatible to both python 3 and python 2.7.


* bf_0_original.py: This is a naive and straightforward implementation. 
The only unusual thing about this interpreter is that the memory modeled as a defauldict. 
Typical brainfuck interpreters assume the memory is fixed size. 
* bf1_map.pyx: cython port of bf_0, using stl map to replace python defaultdict
* bf2_vector.pyx: program is stored in stl vector instead of python list
* bf3_list_all.py: both program and memory are stored in python list
* bf3_vector_all.pyx: cython version of bf_3, both program and memory are stored in stl vector
* vector_list.pxd: interface to stl vector provides some method alias compatible to python list
* bf4_merge.py: by using vector_list.pxd, we can merge two version of bf3 can share the same source code.
* bf5_improved.py: adds more optimization to bf_4. 
* bf6_inline.py: brainfuck interpreter powerd by cython inline compiler

## Compile py/pyx 
To compile py/pyx by cython, in terminal,
```bash
cython --embed --cplus bf5_improved.py
g++ -O2 -o bf5 bf5_improved.cpp `python2-config --cflags --ldflags`
```

## Supplement
Along with these python implementations, a few examples in bf/ and a very fast c implementation in bff4 is also provided for testing and comparing. 
Examples are from http://esoteric.sange.fi/brainfuck/, the credit goes to respective authors. 
The mandelbrot.bf example is written by Erik Bosman and is frequently used for benchmarking Brainfuck interpreters and compilers.
The fast interpreter bff4.c is written by Oleg Mazonka (http://mazonka.com/brainf/), is said to be the fastest brainfuck interpreter (http://esolangs.org/wiki/Brainfuck).
