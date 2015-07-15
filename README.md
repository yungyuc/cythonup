# cythonup: a one-day Cython workshop

Tentative syllabus:

- Introduction and Installation
- Quick Prototyping for Speed
   1. When and when not to use cython for prototyping and speed up
   2. Fused type, type inference and Jedi typer automatic type injection
   3. Multithreading and GIL
   4. Typed memoryviews
   5. Basic hands on examples
   6. More hands on examples (if time permits)
      - Discrete optimization algorithm
      - Brain fuck interpreter
      - CPU emulator
- Interaction with C
   1. Array-based computing: a complete journey from Python to C
      1. 1D Lax-Wendroff scheme in vanilla Python
      2. Bring in NumPy
      3. Make it Cython
         - Come back to element-wise code from vector-like
      4. Make it C
         - Let's worry about building the code
   2. Tweak pointers between Cython and C
   3. Share declaration between Cython and C

# Introduction and Installation

Instructor: Cheng-Lung Sung

# Quick Prototyping for Speed

Instructor: Tzer-jen Wei

# Interaction with C

Instructor: Yung-Yu Chen