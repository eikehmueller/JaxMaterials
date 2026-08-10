---
title: 'JaxMaterials: A JAX package for efficient differentiable material modelling'
tags:
  - Python
  - JAX
  - differentiable programming
  - adjoint state method
  - material modelling
  - composite materials
  - Lippmann Schwinger equations
authors:
  - name: Yang Chen
    orcid: 0000-0003-1026-0482
    equal-contrib: true
    affiliation: "1"
  - name: Eike Hermann Mueller
    orcid: 0000-0003-3006-3347
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: University of Bath, BA2 7EX Claverton Down, United Kingdom
   index: 1
date: 6 August 2026
bibliography: paper.bib

---

# Summary

# Statement of need
Many materials of interest in engineering, such as carbon-fibre composites **REFERENCE**, can be modelled by solving a system of PDEs for spatially varying stress $\sigma(x)$ and strain $\varepsilon(x)$. In general, these two quantities are related by a constituitive law of the form $\sigma = \Sigma(\varepsilon|\theta)$ which depends on problem-specific parameters $\theta$. For example, for linear elasticity problems $\theta$ represents the spatially varying elasticity tensor $C(x)$ and $\sigma(x)=C(x)\varepsilon(x)$. The PDE solver might be embedded into an outer iteration, for example when including dynamic fracture formation @Chen:2019 where $\Sigma$ depends non-linearly on $\varepsilon$. The Lippmann Schwinger iteration with a FFT-based based homogenous solver @Moulinec:1998, @Schneider:2021 is a widely used and highly efficient method if the PDE system is discretised on a structured grid. 

However, in many cases, not just the value of some objective function $J=J(\varepsilon,\sigma)$ needs to be computed, but the sensitivity $\delta J/\delta \theta$ to the input parameters is also required. This includes applications in uncertainty quantification **REFERENCE** and hybrid machine learning approaches such as Physics Enhanced Surrogates (PEDS) @Pestourie:2023, which embed the PDE solver into a machine-learning workflow.

To resolve fine structure in multiscale simulations, implementations need to be fast, differentiable and easily adaptable to arbitrary constituitive laws specified by domain specialists. Our code addresses this challenge since it allows the differentiable solution of the fundamental PDEs for an arbitrary, user defined constitutitive law.

Usually the dimension of the objective function $J$ is much smaller than the dimension of the input parameters $\theta$ and *forward mode differentiation* with Jabobian-vector products (jax.jvp's) is very inefficient (see discussion in Section 2 of @Pundir:2025). On the other hand, using *reverse mode differentiation* (backpropagation) is not trivial due to the iterative nature of the Lippmann Schwinger solver:

* In the forward pass, the states for all iterations need to be stored to allow back-propgation of gradients, which leads to significant memory overhead and might make the simulation of large problems intractable.
* If - as in all real applications - a dynamic stopping criterion is used to terminate the iteration, JAX, cannot compute the reverse mode gradient since the trip-count of while-loops is unknown at compile time.

To address these issues, we employ the adjoint state method (see e.g. @Hinze:2008, @Johnson:2012). This leads to an adjoint Lippmann Schwinger equation of a very simular structure which is solved iteratively.

# State of the field
Since the semial work in @Moulinec:1998, a well established approach has been to solve the PDEs for stress and strain with the iterative Lippmann-Schwinger algorithm. This is used in sophisticated software packages for material modelling such as AMITEX @Gelebart:2020, which is implemented in Fortran. Recently there has been significant interest in differentiable implementations. This has be spurned by the advent of easy-to-use libraries such as JAX and PyTorch, which allow the automatic forward and backward propagation of gradients in sophisticated neural network architectures. JAX employs just-in-time (JIT) compilation to generate efficient code which runs on CPUs and GPUs. The authors of @Pundir:2025 describe a JAX implementation for material modelling: the user only needs to encode the functional relationships and all gradients are derived symbolically in JAX. In this work, we focus on the efficient implementation of differentiable Lippmann Schwinger iterations based on the adjoint-state method, which allows reserve mode differentiation. This method is widely used for PDE solvers based on finite elements, consider for example pyadjoint @Mitusch:2019, which has been integrated into the Firedrake framework @Farrell:2013, @Rathgeber:2016. The approach has recently been used for material modelling @Farsi:2025, but here we extend it to Lippmann Schwinger solvers which are expected to give superior performance on structured grids.

# Software design

## JAX implementation

Since the code is based on JAX, all functions are pure and parameters are passed as state variables. The central functionality is exposed through the function `lippmann_schwinger()` whichs gets passed as a user-defined constituitive law $\Sigma(\varepsilon|\theta)$ of the form:

```Python
def compute_sigma(epsilon, params):
    # Compute stress sigma from strain epsilon, given params
    return sigma
```

Internally, this calls a backend function which is equipped with custom reverse mode gradients through JAX's `defvjp` functionality. It should be stressed that `compute_sigma()` can be any function, as long as it is reverse mode differentiable, and by design our library allows the implementation of non-trivial models such as the one in @Chen:2019.

For convenience, special cases for isotropic and anisotropic elastic materials have been implemented as well; in both cases the constituitive law is implemented in `hooke.py` and the user only needs to pass the relevant entries of the elasticity tensor.

## Low-level CUDA-C implementation

In addition to the Python code, a low level CUDA-C implementation for elastic materials is also provided and it can be accessed through the same interface functions. Note that this needs to be compiled separately (with CMake). Depending on the hardware, this can be faster than the JAX code, in particular for the isotropic case. At the moment, the CUDA implementation is not differentiable, but in principle this limitation could be overcome by implementing the adjoint Lippmann Schwinger solve.

## Example usage

Consider an isotropic elastic material for which $\sigma_{ij}(x) = \lambda(x) \operatorname{tr}(\varepsilon(x))\delta_{ij} + 2\mu(x)\varepsilon_{ij}(x)$. In this case $\theta = \{\mu(x),\lambda(x)\}$ and the constituitive law $\Sigma(\varepsilon|\theta)$ is implemented as the following function:

```Python
def compute_sigma(epsilon, params):
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * params["mu"] * epsilon + params["lambda"] * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
    return sigma
```

The parameters are passed as a dictionary which constitutes a JAX pytree:

```Python
params = {"lambda": lmbda, "mu": mu}
```

In the code, we first import the necessary libraries

```Python
import numpy as np
import jax

from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
```

and construct a structured grid of the domain $\Omega = [0,1]\times[0,1]\times[0,\frac{1}{2}]$ with $32\times32\times16$ voxels:

```Python 
nx, ny, nz = 32, 32, 16
grid_spec = GridSpec(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=0.5)
```

In this example, the Lame parameters $\mu(x)$, $\lambda(x)$ are set to random values and only the first component of the mean strain $\overline{\varepsilon}$ is nonzero:

```Python
rng = np.random.default_rng(seed=47273)
mu = rng.uniform(low=0.8, high=1.1, size=(nx, ny, nz)).astype(np.float32)
lmbda = rng.uniform(low=0.6, high=0.7, size=(nx, ny, nz)).astype(np.float32)
epsilon_bar = np.array([1,0,0,0,0,0]),dtype=np.float32)
params = {"lambda": lmbda, "mu": mu}
```

As in @Moulinec:1998 the reference Lame parameters are obtained by averaging the largest and smallest values across the domain $\mu^0=[\mu]$, $\lambda^0=[\lambda]$ with $[f] := \frac{1}{2}\left(\min_{x\in\Omega}\{f(x)\} + \max_{x\in\Omega}\{f(x)\}\right)$:

```Python
ref_params = {
    key: 1 / 2 * (np.min(value) + np.max(value)) for (key, value) in params.items()
}
```

Assume that we are interested in the objective function $J = \varepsilon^2+\sigma^2$. This can be implemented as a function of $\theta$, $\overline{\varepsilon}$ by calling the differentiable Lippmann Schwinger solver:

```Python
def loss_fn(params, epsilon_bar):
    epsilon, sigma = lippmann_schwinger(
        compute_sigma, params, epsilon_bar, ref_params, grid_spec
    )
    return jnp.sum(epsilon**2 + sigma**2)
```

Finally, gradients of $J$ with respect to $\theta$, $\overline{\varepsilon}$ can be computed by using JAX's autodiff capabilities, for example

```Python
grad_fn = jax.grad(loss_fn, argnums=(0, 1))
g_params, g_epsilon_bar = grad_fn(params, epsilon_bar)
```

### Special case: isotropic elastic material

In this particular case we can also use the bespoke function `lippmann_schwinger_isotropic()` which assumes an isotropic elastic constituitive law. This function only needs to be passed the Lame parameters `params = {"lambda": lmbda, "mu": mu}` and automatically computes $\mu^0$, $\lambda^0$:

```Python
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger_isotropic


def loss_fn(params, epsilon_bar):
    epsilon, sigma = lippmann_schwinger_isotropic(
        params, epsilon_bar, grid_spec=grid_spec
    )
    return jnp.sum(epsilon**2 + sigma**2)
```

If we are only interested in the forward solve, we can also use the slightly more efficient CUDA implementation

```Python
epsilon, sigma = lippmann_schwinger_isotropic(
    params, epsilon_bar, grid_spec, use_cuda=True
)
```

# Research impact statement


## Benchmarking against existing FFT solver
We present some numerical comparison of JaxMaterials agains an established open-source FFT solver AMITEX [REF]. ..........

### 


## Topology optimisation
We design periodic porous metamaterial by topology optimisation with the objective of maximising the effective bulk modulus $K$, subject to prescribed solid volume fractions $\phi=0.1$, $0.2$, $0.3$. 
A cubic representative volume element (RVE) of dimensions $0.5\times 0.5 \times 0.5$ $mm^3$ was considered. The solid phase was assigned a Young's modulus of $E_1=1$ GPa and a Poisson's ratio of 0.3, while the void phase was approximated bys a much softer material with Young's modulus of $E_0=10^{-6}$ GPa. The combination of a very high stiffness contrast and high porosity (low solid volume fracion) poses significant convergence challenge to the basic scheme of [Moulinec-Suquet], but this has been overcome using Anderson acceleration, as discussed in [the previous section].

The optimisation was performed using optimality criteria method [REF], requiring the computation of the gradient of the objective function (negative effective bulk modulus, $-K$) with respect to the density map $\rho$. The effective bulk modulus $K$ was computed from one homogenisation simulation subject to a macroscopic strain load $\overline{\boldsymbol{\varepsilon}}= (1,1,1,0,0,0)^T$ (energy-based method [REF]):

$$
\begin{aligned}
K &= \frac{1}{9} \sum_{i,j=1}^{3} C_{iijj}
   = \frac{1}{9} \overline{\boldsymbol{\varepsilon}}^T : \mathbb{C} : \overline{\boldsymbol{\varepsilon}}
\qquad\text{with}\qquad
\overline{\boldsymbol{\varepsilon}} = (1,1,1,0,0,0)^T
\end{aligned}
$$

The sensitivity $\frac{\partial K}{\partial \rho}$ was evaluated through the chain rule $\frac{\partial K}{\partial \rho} = \frac{\partial K}{\partial \mu} \frac{\partial \mu}{\partial \rho}$, where the gradient $\frac{\partial K}{\partial \mu}$ was computed by JaxMaterials. $\mu$ is the second Lam\'e coefficient (shear modulus), which is related to the Young's modulus $E$ and Poisson's ratio $\nu$ through $\mu=\frac{E}{2(1+\nu)}$. The Young's modulus $E$ was a function of density $\rho$:

$$
E = E_0 + (E_1 - E_0) \rho^p
$$
where $p$ is a SIMP penalty parameter, set to 5 in all examples presented herein. With this explicit expreission of $\mu(\rho)$, the derivative of $\frac{\mu}{\rho}$ can be obtained.

To mitigate numerical instabilities (checker-boarding and mesh-dependency), a sensitivity filtering [REF] was applied:
$$
\widetilde{\frac{\partial K}{\partial \rho}} = \frac{(\frac{\partial K}{\partial \rho}\rho) \odot \omega}{\overline{\omega} \rho}
$$
where $\odot$ denotes periodic convolution (the RVE is periodic). $\omega$ is the convolution kernel that was set to a $2\times 2\times 2$ array with all elements equal to 1, and $\overline{\omega}$ is the sum of all elements in the kernel.

The density map $\rho$ was initialised with the target solid volume fraction $\phi$ in a spherical structure as

$$
\rho =
\begin{cases}
    \phi / 2, & x \in \mathcal{D}, \\
    \phi, & \text{otherwise}.
\end{cases}
\qquad\text{with}\qquad
\mathcal{D}\text{ is the sphere region}
$$
In all cases, the sphere diameter was set to 2/3 of the domain size. 

The evolution of the optimised topology for the three prescribed volume fractions is shown in Figure 1. For visualisation purposes, only a half-cut view of each structure is displayed, revealing the internal morphology of the evolving porous architectures.

<figure>

<div style="display: flex; justify-content: center; gap: 10px;">

<figure style="margin: 0; text-align: center;">
  <img src="figures/to_vf0.1_seq.gif" alt="Figure 1.a" width="400">
  <figcaption>(a) 10% </figcaption>
</figure>

<figure style="margin: 0; text-align: center;">
  <img src="figures/to_vf0.2_seq.gif" alt="Figure 1.b" width="400">
  <figcaption>(b) 20% </figcaption>
</figure>

<figure style="margin: 0; text-align: center;">
  <img src="figures/to_vf0.3_seq.gif" alt="Figure 1.c" width="400">
  <figcaption>(c) 30% </figcaption>
</figure>

</div>

<figcaption style="text-align: center;">
Figure 1: Evolution of the optimised structure with constraint on different volume fractions (a-c) of the solid phase. Only half cut-off view is shown to visualise the inside structure.
</figcaption>

</figure>

The convergence histories of the effective bulk modulus are presented in Figure 2 for the three volume-fraction constraints. In all cases, the optimisation process yields a monotonic increase in bulk stiffness before reaching a stable plateau, indicating successful convergence.

<figure>
<div style="display: flex; justify-content: center; gap: 10px;">
<figure style="margin: 0; text-align: center;">
  <img src="figures/to_convergence.png" alt="Figure 2" width="400">
</figure>
</div>
<figcaption style="text-align: center;">
Figure 2: Evolution of the effective bulk modulus of the metamaterial at various solid volume fractions (vf).
</figcaption>
</figure>









# Mathematics

## PDE Problem
We consider the following system of equations for spatially varying strain $\varepsilon$ and stress $\sigma$:

$$
\begin{aligned}
    \partial_j \sigma_{ij} &= 0 & \text{(Cauchy momentum equation)}\\
    \sigma_{ij} &= \Sigma_{ij}(\varepsilon|\theta)\qquad\text{with $\varepsilon_{k\ell} = \varepsilon^*_{k\ell} + \overline{\varepsilon}_{k\ell}$} & \text{(Constituitive law)}\\
    \varepsilon^*_{k\ell} &= \frac{1}{2}\left(\partial_k u_\ell + \partial_\ell u_k\right) & \text{(Strain-displacement relation)}
\end{aligned}\label{eqn:continuum}
$$

The problem is solved in a rectangular domain with periodic boundary conditions for the displacement field $u(x)$ and for given average strain $\overline{\varepsilon}$.

The problem-dependent constituitive law $\sigma = \Sigma(\varepsilon|\theta)$ depends on the parameters $\theta$. Special cases are:

* Linear anisotropic materials with $\sigma = C \varepsilon$ where $C=C(x)=:\theta$ is the spatially varying elasticity tensor
* Linear isotropic materials for which $C(x) = \lambda(x) \delta_{ij}\delta_{k\ell} + \mu(x) (\delta_{ik}\delta_{j\ell} + \delta_{i\ell}\delta_{jk})$. In this case $\theta$ encapsulates the two Lame parameters $\{\mu(x),\lambda(x)\}=:\theta$.

## Lippmann Schwinger iteration

The problem in (\autoref{eqn:continuum}) can be written in the form 

$$
\varepsilon + \Gamma^0 * (\Sigma(\varepsilon|\theta) - C^0 \varepsilon)= \overline{\varepsilon}\label{eqn:lippmann_schwinger}
$$

Here, $C^0$ is the homogenous isotropic elasticity tensor described by the two reference Lame parameters $\mu^0,\lambda^0\in\mathbb{R}$. The operator $\Gamma^0$ is constructed from the tensor-valued Green's function of the corresponding PDE (see appendix of @Moulinec:1998). 

As discussed in @Moulinec:1998, the self-consistent equation in (\autoref{eqn:lippmann_schwinger}) is solved by exploiting the fact that $\Gamma^0$ is diagonal in Fourier space and by applying the Lippmann Schwinger iteration

$$
\varepsilon^{(s+1)} = \overline{\varepsilon} - \mathcal{F}^{-1} \circ\widehat{\Gamma}^0 \circ\mathcal{F}\tau^{(s)} \qquad\text{with $\tau^{(s)} = (\Sigma(\varepsilon^{(s)}|\theta) - C^0\varepsilon^{(s)})$ and $\varepsilon^{(0)} = \overline{\varepsilon}$}
$$

Here $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the forward and inverse Fourier transform respectively. In the code we also implemented Anderson acceleration @Wicht:2021 which - for linear stress-strain relationships - is equivalent to a preconditioned GMRES iteration @Walker:2011.

## Reverse-mode differentiation with the adjoint method

Let $J=J(\varepsilon,\sigma)$ be the objective function. We want to compute the sensitivities

$$
\frac{\delta J}{\delta \theta},\quad \frac{\delta J}{\delta \overline{\varepsilon}}
$$

subject to the condition that strain $\varepsilon=\varepsilon(\theta,\overline{\varepsilon})$ and stress $\sigma=\sigma(\theta,\overline{\varepsilon})$ satisfy the equations in (\autoref{eqn:continuum}) for given $\overline{\varepsilon}$ and material parameters $\theta$.

Since every step of the Lippmann Schwinger iteration consists of elementary, differential operations, in principle the sensitivites can be obtained with JAX automatic differentiation capabilities provided the constitutitive law $\sigma=\Sigma(\varepsilon|\theta)$ if differentiable. As reverse mode-differentiation is not available for while-loops, we employ the adjoint state method @Hinze:2008, @Johnson:2012. This leads to an adjoint Lippmann Schwinger equation of a very simular structure as (\autoref{eqn:lippmann_schwinger}) which is solved iteratively, possibly with Anderson acceleration.

$$
\Lambda + (\Gamma^0*\Lambda)\frac{\delta \Sigma}{\delta \varepsilon} - \lambda^0 \operatorname{tr}(\Gamma^0*\Lambda)\mathbb{I} - 2\mu^0 (\Gamma^0*\Lambda) = -\left(\frac{\delta J}{\delta \varepsilon}+\frac{\delta J}{\delta \sigma}\frac{\delta \Sigma}{\delta \varepsilon}\right)
$$

From $\Lambda$ the derivatives of the objective function with respect to the parameters $\theta$ and the average strain $\overline{\varepsilon}$ can be computed as

$$
\frac{\delta J}{\delta\theta} = \left(\frac{\delta J}{\delta\sigma} + \Gamma^0*\Lambda\right): \frac{\delta\Sigma}{\delta \theta},\qquad
\frac{\delta J}{\delta\overline{\varepsilon}} = -\int_\Omega \Lambda(z)\;dz.
$$

# AI usage disclosure

 Generative AI (mainly GitHub copilot) was used to generate some of the code, to identify and locate bugs and to generate informed feedback on the code and documentation. Code generation was closely supervised by limiting it to routine transformations and by breaking it into small, transparent tasks the correctness implementation of which could be easily verified. All generated code and all AI suggestions were carefully reviewed by the authors to ensure correctness. 

# Acknowledgements

# References