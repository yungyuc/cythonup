# A simple solver of knapsack problem

This directory contains 4 python implementations of knapsack problem solver. 
All 4 implementations share the same algorithm, but with different implementation details.

* knapsack_0_original: original implementation, very straightforward. 
* knapsack_1_improved: removing some temporary lists and list copy to improve speed.
* knapsack_2_typed: add type annotation
* knapsack_3_memoryview: use cython typed memoryview

## Usage

All four implementations, knapsack_0_original, knapsack_1_improved, knapsack_2_typed, knapsack_3_memoryview have the same usage. 

Take knapsack_1_improved for example, in terminal, 
```bash
time python kapsack_1_improved.py
```
to test the performance of the implementation.
It uses ks_10000_0 as default data. This can be override by specifying the file name in the argument
```bash
time python kapsack_1_improved.py ks_400_0
```
Data files in this directory are from Professor Pascal Van Hentenryck's discrete optimization course https://coursera.org/course/optimization

These implementations are compatible with both python 2.7 and python 3.

To compile and run them with cython, in terminal, 
```bash
cython --embed knapsack_1_improved.py
gcc -O3 -o knapsack_1_improved knapsack_1_improved.c `python2-config --cflags --ldflags`
time ./knapsack_1_improved
```
you can replace python2-config with python3-config if you wish to test them with python 3 runtime.

