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

# State of the field

# Software design

# Research impact statement

# Mathematics

## Problem
We consider the steady state elasiticity equation which imposes the following conditions on the spatially verying strain $\varepsilon$ and stress $\sigma$:

$$
\begin{aligned}
    \partial_j \sigma_{ij} &= 0 \\
    \sigma_{ij} &= \Sigma_{ij}(\varepsilon|\theta)\qquad\text{with $\varepsilon_{k\ell} = \varepsilon^*_{k\ell} + \overline{\varepsilon}_{k\ell}$}\\
    \varepsilon^*_{k\ell} &= \frac{1}{2}\left(\partial_k u_\ell + \partial_\ell u_k\right)
\end{aligned}\label{eqn:continuum}
$$

in the rectangular domain with periodic boundary conditions and given average strain $\overline{\varepsilon}$.

The problem-dependent stress-strain relationship $\sigma = \Sigma(\varepsilon|\theta)$ depends on the parameters $\theta$. Special cases are:

* Linear anisotropic materials with $\sigma = C \varepsilon$ where $C=C(x)=:\theta$ is the spatially varying elasticity tensor
* Linear isotropic materials for which $C(x) = \lambda(x) \delta_{ij}\delta_{k\ell} + \mu(x) (\delta_{ik}\delta_{j\ell} + \delta_{i\ell}\delta_{jk})$. In this case $\theta$ encapsulates the two Lame parameters $\{\mu(x),\lambda(x)\}=:\theta$.

## Lippmann Schwinger iteration

The problem in \autoref{eqn:continuum} can be written in the form 

$$
\varepsilon = \overline{\varepsilon} - \Gamma^0 * (\Sigma(\varepsilon|\theta) - C^0 \varepsilon)\label{eqn:lippmann_schwinger}
$$

Here, $C^0$ is the homogeneous isotropic elasticity tensor described by the two reference Lame parameters $\mu^0,\lambda^0\in\mathbb{R}$, as discussed in `@Moulinec:1998`.

The self-consistent equation in \autoref{eqn:lippmann_schwinger} is solved by exploiting the fact that $\Gamma^0$ is diagonal in Fourier space and by applying the Lippmann Schwinger iteration

$$
\varepsilon^{(s+1)} = \overline{\varepsilon} - \mathcal{F}^{-1} \circ\widehat{\Gamma}^0 \circ\mathcal{F}\tau^{(s)} \qquad\text{with $\tau^{(s)} = (\Sigma(\varepsilon^{(s)}|\theta) - C^0\varepsilon^{(s)})$ and $\varepsilon^{(0)} = \overline{\varepsilon}$}
$$

to obtain an approximate solution for the strain $\varepsilon$ and stress $\sigma$. $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the forward and inverse Fourier transform respectively. In the code we also implemented Anderson acceleration `@Wicht:2021` which - for linear stress-strain relationships - is equivalent to a preconditioned GMRES iteration `@Walker:2011`.

## Sensitivity

Let $J=J(\varepsilon,\sigma)$ be and objective function. We want to compute the sensitivities

$$
\frac{\delta J}{\delta \theta},\quad \frac{\delta J}{\delta \overline{\varepsilon}}
$$

subject to the condition that strain $\varepsilon=\varepsilon(\theta,\overline{\varepsilon})$ and stress $\sigma=\sigma(\theta,\overline{\varepsilon})$ satisfy the equations in \autoref{eqn:continuum} for given $\overline{\varepsilon}$ and material parameters $\theta$.

This is required in classical sensitivity studies but also if the model forms part of a Physics Enhanced Deep Surrogate (PEDS) `@Pestourie:2023`.

Since every step of the Lippmann Schwinger iteration consists of elementary, differential operations, in principle the sensitivites can be obtained with JAX automatic differentiation capabilities. Since usually the dimension of the objective function $J$ is much smaller than the dimension of the input parameters $\theta$, forward mode differentiation is very inefficient. Instead, reverse mode differtiation should be used. This can be obtained automatically in the form of as JAX vector-Jacobian products (jax.vjp's), provided the function $\Sigma(\varepsilon|\theta)$ is differentiable. However, there are two problems:

* The states for all Lippmann Schwinger iterations need to be stored to implement the reverse mode differentiation
* If - as in all real applications - a dynamic stopping criterion is used instead of a fixed number of iterations, JAX cannot compute the reverse mode derivative

To address these issues, the adjoint state method (see e.g. `@Hinze:2008`, `@Johnson:2012`) is employed instead. This leads to an adjoint Lippmann Schwinger equation of a very simular structure which is solved iteratively, possibly with Anderson acceleration.

# AI usage disclosure

 Generative AI (mainly GitHub copilot) was used to generate some of the code, to identify and locate bugs and to generate informed feedback on the code and documentation. Code generation was closely supervised by limiting it to routine transformations and by breaking it into small, transparent tasks that could be easily verified. All generated code and all AI suggestions were carefully reviewed by the authors to ensure correctness. 
# Acknowledgements

# References