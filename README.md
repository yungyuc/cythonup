# cythonup: a one-day Cython workshop

Agenda:

1. Introduction and Installation by Cheng-Lung Sung
2. Quick Prototyping for Speed by Tzer-jen Wei
3. Interaction with C by Yung-Yu Chen

Contents:

- `src/`: Source code of the examples.
- `module/`: Course notes used in the classroom.

## Introduction and Installation

Instructor: Cheng-Lung Sung

 1. Introduction to Cython
 2. Build and Running Cython Code
 3. Extenstion Types
 4. Wrapping C library with Cython
 5. Using C++ in Cython
 6. Example Code: chatroom (tentative)

## Quick Prototyping for Speed

Instructor: Tzer-jen Wei

 1. When and when not to use cython for prototyping and speed up
 2. Fused type, type inference and Jedi typer automatic type injection
 3. Multithreading and GIL
 4. Typed memoryviews
 5. Basic hands on examples
 6. More hands on examples (if time permits)
    - Discrete optimization algorithm
    - Brainfuck interpreter
    - ~~CPU emulator~~ (Edit: too similar to interpreter, add Go AI example instead)
    - Go AI

## Interaction with C

Instructor: [Yung-Yu Chen](mailto:yyc@solvcon.net)

1. Array-based computing: a complete journey from Python to C
   1. Step 0: Set up the problem framework code
   2. Step 1: The initial Python version
   3. Step 2: Let Numpy help (from element-wise to vector-like code)
   4. Step 3: Cython-enhanced code (back from vector-like to element-wise code)
   5. Step 4: Go to C if it's not enough
   6. Example 1: Build Cython and C
   7. Example 2: Make a package
   8. Example 3: Compare all
   9. Conclusion
2. Data sharing between Cython/Python and C
   1. Tweak pointers
   2. Share declaration
