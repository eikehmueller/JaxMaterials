# Objective

The JAX implementation of the Lippmann-Schwinger solver supports both isotropic and anisotropic materials. The code can be found in the folder src/jaxmaterials/solver. The isotropic solver gets passed the two Lame parameter lambda and mu, which each are represented as tensors of shape (Nx,Ny,Nz), i.e. one value per voxel. The anisotropic solver gets passed a single input of shape (21,Nx,Ny,Nz) which represents the 21 independent entries of the linear elasticity tensor in each voxel.

In contrast, the CUDA implementation only supports isotropic materials.

The goal of this code change is to implement the JAX functionality also in the CUDA code, which should support both isotropic and anisotropic materials. Tests should be developed alongside the code and the changes should be introduced in stages to allow debugging.

# General guidance

Only proceed to the next stage once a stage has been completed and approved by the human developer. If there is any ambiguity in the description, double check with the human developer. Split steps into substeps if you feel that they introduce too many changes in one go. This document might change, so double check regularly.

Verify compile and run focused tests at each stage.

For any code you write, please clearly indicate your authorship by stating your version.

Use consistent naming conventions, for example:

* compute_stress_isotropic / compute_stress_anisotropic
* acoustic_tensor_anisotropic / fourier_solve_anisotropic

# Step-by-step instructions

## Step 1
Do all work in the local branch cuda_anisotropic.

## Step 2

### Step 2a
Remove the class method compute_stress() and associated kernel from the LippmannSchwinger class in the CUDA code and place them in methods outside the class. Rewrite the files lippmann_schwinger.hh and lippmann_schwinger.cu accordingly and make sure that the code still compiles and runs correctly.

### Step 2b
Create a new pair of files hooke.hh and hooke.cu which should implement the computation of stress from strain for both the isotropic (which was previously in compute_stress()) and anisotropic case. These files should implement the functionality which is currently implemented in hooke.py. 

Write a CUDA test in the file cuda/src/tests/test_hooke.hh which replicates the Python test in test_hooke.py.

## Step 3
In fourier.hh / fourier.cu implement a method which computes the acoustic tensor for the anisotropic case, i.e. translate the method get_anisotropic_acoustic_tensor() in fourier.py into CUDA code.

Write a corresponding test in test_fourier.hh. This test should do the same as test_acoustic_tensor() in test_fourier.py.

## Step 4
In fourier.hh / fourier.cu implement a method which computes the *inverse* of the acoustic tensor for the anisotropic case, i.e. translate the method get_inverse_anisotropic_acoustic_tensor() in fourier.py into CUDA code.

Write a corresponding test in test_fourier.hh. This test should do the same as test_inverse_acoustic_tensor() in test_fourier.py, i.e. it should verify that for an isotropic material the inverse of the acoustic tensor computed with the anisotropic method matches the analytical expression.

## Step 5
In fourier.hh / fourier.cu implement a method which solves the homogeneous reference problem in Fourier space for the anisotropic case. This method should do the same as  fourier_solve_anisotropic() in fourier.py, i.e. it should solve residual equation for homogeneous anisotropic reference material in Fourier space.

Write a corresponding test which verifies that Fourier solve with the new routine for the anisotropic case gives same results as the computation with fourier_solve_device() when applied to an isotropic reference material.

## Step 6
Extend the Lippmann Schwinger solver in lippmann_schwinger.hh / lippmann_schwinger.cu such that it can handle both isotropic and anisotropic materials. For this, proceed as follows:

### Step 6a
Take the current LippmannSchwinger class and split it into a base class, which will contain common data structures and variables which are required for both the isotropic and anisotropic case, and a derived class which implements the isotropic Lippmann Schwinger solver. This should be a null-change, i.e. the code should still provide exactly the same functionality.

### Step 6b
Derive a new class for the anisotropic Lippmann Schwinger solver from the same based class. This derived class should now implement an apply method which is specific to the anisotropic case.

### Step 6c

Implement CUDA tests which verify that both classes give the same result when applied to an isotropic material.

### Step 7
The goal of this step is to:

* provide a new anisotropic entrypoint
* support updated input shape handling
* backward compatibility for existing isotropic calls

Adapt the Python interface for the CUDA code such that it can handle both the current isotropic solver and the new anisotropic CUDA solver implemented in the previous steps.

Write tests which verify that the CUDA and JAX solver give the same results when applied to an anisotropic material, i.e. replicate the test test_jax_matches_cuda() in test_lippmann_schwinger.py for the anisotropic case.