# Contributing to this project

## Issues and bugs
If you have problems using this code, please contact the developers and/or open an issue. Provide as much information as possible:

* Wider context:
    - Which problem are you solving? 
    - How do you call the library from your code?
* Details on the environment such as Python version and installed packages 
* Exact error messages and logs
* What have you already tried to fix the problem?
* Can you provide a minimal working example (MWE) which demonstrates the issue?

## General guidelines
To contribute new functionality, please fork the repository and create a pull request. Contact the developers to discuss how to merge your changes.

Any contributions should follow the common development guidelines to ensure that the code remains maintainable and useable:

* Add tests which can be integrated into the CI pipeline
* Keep the code modular
* Adhere to existing interfaces
* Thoroughly document both your code and the mathematics behind it

## Suggested areas of development

### Extend and optimise CUDA code

Note that the JAX code is already very fast, so only do this if it is likely lead to an improvement

* Implement Anderson acceleration for CUDA solvers
* Multi-GPU parallelisation
* Detailed profiling and optimisation 

### Additional problems

* Extend the code to 2d
* Add Lippmann Schwinger solvers for other equations, in particular advection reaction diffusion $-\nabla K \cdot (\nabla u) + a u + b\cdot \nabla u = f$