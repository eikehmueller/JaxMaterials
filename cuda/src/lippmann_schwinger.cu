/** @brief Implementation of lippmann_schwinger.hh */

#include "lippmann_schwinger.hh"

/* **** CUDA kernels **** */

/* Kernel for setting the values of epsilon to the constant bar(epsilon) */
__global__ void set_epsilon_initial_kernel(float *__restrict__ dev_epsilon,
                                           float *__restrict__ dev_epsilon_bar,
                                           float *__restrict__ dev_delta_epsilon_initial,
                                           const size_t nvoxels)
{
  int ell = blockDim.x * blockIdx.x + threadIdx.x;
  if (ell < nvoxels)
    for (int alpha = 0; alpha < 6; ++alpha)
      dev_epsilon[alpha * nvoxels + ell] = dev_epsilon_bar[alpha] + dev_delta_epsilon_initial[alpha * nvoxels + ell];
}

/* Kernel for incrementing solution epsilon -> epsilon + alpha*r */
__global__ void increment_solution_kernel(float *__restrict__ dev_epsilon,
                                          float *__restrict__ dev_r,
                                          const float alpha,
                                          const size_t ndof)
{
  int ell = blockDim.x * blockIdx.x + threadIdx.x;
  if (ell < ndof)
  {
    dev_epsilon[ell] += alpha * dev_r[ell];
  }
}

/* **** class methods **** */

/* Set the values of epsilon to bar(epsilon) on the device */
void LippmannSchwingerSolverBase::set_epsilon_initial(float *__restrict__ dev_epsilon,
                                                      float *__restrict__ epsilon_bar,
                                                      float *__restrict__ delta_epsilon_initial)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  const size_t nblocks = (nvoxels + BLOCKSIZE - 1) / BLOCKSIZE;
  set_epsilon_initial_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, epsilon_bar, delta_epsilon_initial, nvoxels);
}

/* Increment solution epsilon -> epsilon + 1/nvoxels * r */
void LippmannSchwingerSolverBase::increment_solution(float *__restrict__ dev_epsilon,
                                                     float *__restrict__ dev_r)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  size_t ndof = 6 * nvoxels;
  const size_t nblocks = (ndof + BLOCKSIZE - 1) / BLOCKSIZE;
  float alpha = 1.0 / float(nvoxels);
  increment_solution_kernel<<<nblocks, BLOCKSIZE>>>(dev_epsilon, dev_r, alpha, ndof);
}

/* Compute normalised divergence for stopping criterion in Fourier space */
float LippmannSchwingerSolverBase::relative_divergence_norm(cufftComplex *__restrict__ dev_sigma_hat)
{
  // Compute divergence in Fourier space
  divergence_fourier(dev_sigma_hat, dev_div_sigma_hat, dev_xi, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t nmodes = grid_spec.number_of_modes();
  // STEP 1: Compute nrm_div_sigma =  ||div(sigma)||
  float nrm_div_sigma = reduce_fourier(dev_div_sigma_hat, dev_sum, sum, 3, grid_spec);
  // STEP 2: compute ||hat(sigma)||
  // Extract zero mode, which is identical to the sum of sigma over the domain, i.e. nvoxels * <sigma>
  CUDA_CHECK(cudaMemcpy2D(sigma_0, sizeof(cufftComplex), dev_sigma_hat, nmodes * sizeof(cufftComplex), sizeof(cufftComplex), 6, cudaMemcpyDeviceToHost));
  // Compute norm of zero mode
  float nrm_sigma = tensor_norm(sigma_0, 1);
  return nrm_div_sigma / nrm_sigma;
}

/* Constructor */
LippmannSchwingerSolverBase::LippmannSchwingerSolverBase(const GridSpec grid_spec, const int verbose)
    : grid_spec(grid_spec), verbose(verbose)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  size_t nmodes = grid_spec.number_of_modes();
  // Set up cuFFT plans
  int n[3] = {(int)grid_spec.nx, (int)grid_spec.ny, (int)grid_spec.nz};
  int n_fourier[3] = {(int)grid_spec.nx, (int)grid_spec.ny, (int)grid_spec.nz / 2 + 1};
  CUFFT_CHECK(cufftPlanMany(&plan_forward, 3, n, n, 1, nvoxels, n_fourier, 1, nmodes, CUFFT_R2C, 6));
  CUFFT_CHECK(cufftPlanMany(&plan_inverse, 3, n, n_fourier, 1, nmodes, n, 1, nvoxels, CUFFT_C2R, 6));
  CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_xi, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_div_sigma, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_div_sigma_hat, 3 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_sigma_hat, 6 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_residual, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_residual_hat, 6 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&sigma_0, 6 * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_epsilon_bar, 6 * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_delta_epsilon_initial, 6 * nvoxels * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&dev_sum, sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sum, sizeof(float)));
  // initialize Fourier vectors
  initialize_xi_device(dev_xi, grid_spec);
  initialize_xizero_device(dev_xi_zero, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
}

/* Destructor */
LippmannSchwingerSolverBase::~LippmannSchwingerSolverBase()
{
  // free memory
  CUDA_CHECK(cudaFree(dev_xi));
  CUDA_CHECK(cudaFree(dev_xi_zero));
  CUDA_CHECK(cudaFree(dev_epsilon));
  CUDA_CHECK(cudaFree(dev_sigma));
  CUDA_CHECK(cudaFree(dev_div_sigma));
  CUDA_CHECK(cudaFree(dev_div_sigma_hat));
  CUDA_CHECK(cudaFree(dev_residual));
  CUDA_CHECK(cudaFree(dev_residual_hat));
  CUDA_CHECK(cudaFree(dev_sigma_hat));
  CUDA_CHECK(cudaFree(dev_epsilon_bar));
  CUDA_CHECK(cudaFree(dev_delta_epsilon_initial));
  CUDA_CHECK(cudaFreeHost(sigma_0));
  CUDA_CHECK(cudaFree(dev_sum));
  CUDA_CHECK(cudaFreeHost(sum));
  CUFFT_CHECK(cufftDestroy(plan_forward));
  CUFFT_CHECK(cufftDestroy(plan_inverse));
}

/* Constructor */
LippmannSchwingerSolver::LippmannSchwingerSolver(const GridSpec grid_spec, const int verbose)
    : LippmannSchwingerSolverBase(grid_spec, verbose)
{
  size_t nvoxels = this->grid_spec.number_of_voxels();
  CUDA_CHECK(cudaMalloc(&dev_lambda, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_mu, nvoxels * sizeof(float)));
}

/* Destructor */
LippmannSchwingerSolver::~LippmannSchwingerSolver()
{
  CUDA_CHECK(cudaFree(dev_lambda));
  CUDA_CHECK(cudaFree(dev_mu));
}

/* Constructor */
LippmannSchwingerAnisotropicSolver::LippmannSchwingerAnisotropicSolver(const GridSpec grid_spec, const int verbose)
    : LippmannSchwingerSolverBase(grid_spec, verbose)
{
  size_t nvoxels = this->grid_spec.number_of_voxels();
  size_t nmodes = this->grid_spec.number_of_modes();
  CUDA_CHECK(cudaMalloc(&dev_stiffness, 21 * nvoxels * sizeof(float)));
}

/* Destructor */
LippmannSchwingerAnisotropicSolver::~LippmannSchwingerAnisotropicSolver()
{
  CUDA_CHECK(cudaFree(dev_stiffness));
}

/* apply solver */
int LippmannSchwingerSolver::apply(float *lambda, float *mu, float *epsilon_bar,
                                   float *delta_epsilon_initial,
                                   float *epsilon, float *sigma,
                                   float rtol, float atol, int maxits)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  // Average values of lambda and mu
  float lambda_ref = 0.5 * (*std::max_element(lambda, lambda + nvoxels) + *std::min_element(lambda, lambda + nvoxels));
  float mu_ref = 0.5 * (*std::max_element(mu, mu + nvoxels) + *std::min_element(mu, mu + nvoxels));

  // copy Lame parameters to device
  CUDA_CHECK(cudaMemcpy(dev_lambda, lambda, nvoxels * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_mu, mu, nvoxels * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_epsilon_bar, epsilon_bar, 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_delta_epsilon_initial, delta_epsilon_initial, 6 * nvoxels * sizeof(float), cudaMemcpyHostToDevice));

  // set average value of epsilon
  set_epsilon_initial(dev_epsilon, dev_epsilon_bar, dev_delta_epsilon_initial);
  CUDA_CHECK(cudaDeviceSynchronize());
  // main Lippmann-Schwinger loop
  float rel_div_norm = 0;
  float rel_div_norm0;
  int its;
  if (verbose > 1)
  {
    printf("==== CUDA forward solve ====\n");
    printf("  iteration  E = ||div(sigma)||/||sigma||  E/E_0\n");
  }
  for (its = 0; its < maxits; ++its)
  {
    /* ==== STEP 1 ==== Compute stress: sigma_{ij} = C_{ijkl} epsilon_{kl} */
    compute_stress_isotropic(dev_epsilon, dev_sigma, dev_lambda, dev_mu, grid_spec);
    /* ==== STEP 2 ==== Fourier transform:  hat(sigma) = FFT(sigma)*/
    CUFFT_CHECK(cufftExecR2C(plan_forward, dev_sigma, dev_sigma_hat));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 3 ==== Check convergence */
    rel_div_norm = relative_divergence_norm(dev_sigma_hat);
    if (its == 0)
      rel_div_norm0 = rel_div_norm;
    if (verbose > 1)
      printf("     %4d          %8.4e  %8.4e\n", its, rel_div_norm, rel_div_norm / rel_div_norm0);
    if (rel_div_norm < max(rtol * rel_div_norm0, atol))
      break;
    /* ==== STEP 4 ==== Solve in Fourier space: hat(r)_{kl} = -Gamma^{0}_{klij} hat(sigma)_{ij} */
    fourier_solve_device(dev_sigma_hat, dev_residual_hat, dev_xi_zero, lambda_ref, mu_ref, grid_spec);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 5 ==== Inverse Fourier transform: r = FFT^{-1}(hat(r)) */
    CUFFT_CHECK(cufftExecC2R(plan_inverse, dev_residual_hat, dev_residual));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 6 ==== Update solution: epsilon -> epsilon + r*/
    increment_solution(dev_epsilon, dev_residual);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  if (verbose > 0)
  {
    printf("CUDA forward solver ");
    if (its < maxits)
      printf("converged after %6d of %6d iterations\n", its, maxits);
    else
      printf("failed to converge after %6d iterations\n", maxits);
    printf("E = ||div(sigma)||/||sigma|| = %8.2e E/E_0 = %6.2e\n", rel_div_norm, rel_div_norm / rel_div_norm0);
  }
  // Copy solution back to host
  CUDA_CHECK(cudaMemcpy(epsilon, dev_epsilon, 6 * nvoxels * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sigma, dev_sigma, 6 * nvoxels * sizeof(float), cudaMemcpyDeviceToHost));

  return its;
}

/* apply solver (anisotropic) */
int LippmannSchwingerAnisotropicSolver::apply(float *stiffness,
                                              float *epsilon_bar,
                                              float *delta_epsilon_initial,
                                              const float lambda_ref,
                                              const float mu_ref,
                                              float *epsilon,
                                              float *sigma,
                                              float rtol, float atol, int maxits)
{
  size_t nvoxels = grid_spec.number_of_voxels();

  // Copy material data to device and construct anisotropic Fourier operator.
  CUDA_CHECK(cudaMemcpy(dev_stiffness, stiffness, 21 * nvoxels * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_epsilon_bar, epsilon_bar, 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_delta_epsilon_initial, delta_epsilon_initial, 6 * nvoxels * sizeof(float), cudaMemcpyHostToDevice));

  // Set average value of epsilon.
  set_epsilon_initial(dev_epsilon, dev_epsilon_bar, dev_delta_epsilon_initial);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Main Lippmann-Schwinger loop.
  float rel_div_norm = 0;
  float rel_div_norm0;
  int its;
  if (verbose > 1)
  {
    printf("==== CUDA forward solve ====\n");
    printf("  iteration  E = ||div(sigma)||/||sigma||  E/E_0\n");
  }
  for (its = 0; its < maxits; ++its)
  {
    /* ==== STEP 1 ==== Compute stress: sigma_{ij} = C_{ijkl} epsilon_{kl} */
    compute_stress_anisotropic(dev_epsilon, dev_sigma, dev_stiffness, grid_spec);
    /* ==== STEP 2 ==== Fourier transform:  hat(sigma) = FFT(sigma) */
    CUFFT_CHECK(cufftExecR2C(plan_forward, dev_sigma, dev_sigma_hat));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 3 ==== Check convergence */
    rel_div_norm = relative_divergence_norm(dev_sigma_hat);
    if (its == 0)
      rel_div_norm0 = rel_div_norm;
    if (verbose > 1)
      printf("     %4d          %8.4e  %8.4e\n", its, rel_div_norm, rel_div_norm / rel_div_norm0);
    if (rel_div_norm < max(rtol * rel_div_norm0, atol))
      break;
    /* ==== STEP 4 ==== Solve in Fourier space: hat(r) = -Gamma^{0} hat(sigma) */
    fourier_solve_device(dev_sigma_hat, dev_residual_hat, dev_xi_zero, lambda_ref, mu_ref, grid_spec);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 5 ==== Inverse Fourier transform: r = FFT^{-1}(hat(r)) */
    CUFFT_CHECK(cufftExecC2R(plan_inverse, dev_residual_hat, dev_residual));
    CUDA_CHECK(cudaDeviceSynchronize());
    /* ==== STEP 6 ==== Update solution: epsilon -> epsilon + r */
    increment_solution(dev_epsilon, dev_residual);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  if (verbose > 0)
  {
    printf("CUDA forward solver ");
    if (its < maxits)
      printf("converged after %6d of %6d iterations\n", its, maxits);
    else
      printf("failed to converge after %6d iterations\n", maxits);
    printf("E = ||div(sigma)||/||sigma|| = %8.2e E/E_0 = %6.2e\n", rel_div_norm, rel_div_norm / rel_div_norm0);
  }

  // Copy solution back to host.
  CUDA_CHECK(cudaMemcpy(epsilon, dev_epsilon, 6 * nvoxels * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sigma, dev_sigma, 6 * nvoxels * sizeof(float), cudaMemcpyDeviceToHost));

  return its;
}
/* Lippmann Schwinger iteration */
extern "C"
{
  int lippmann_schwinger_solve_isotropic(float *lambda, float *mu, float *epsilon_bar,
                                         float *delta_epsilon_initial,
                                         float *epsilon, float *sigma,
                                         int *voxels,
                                         float *extents,
                                         float rtol, float atol, int maxits,
                                         int verbose)
  {
    GridSpec grid_spec;
    grid_spec.nx = voxels[0];
    grid_spec.ny = voxels[1];
    grid_spec.nz = voxels[2];
    grid_spec.Lx = extents[0];
    grid_spec.Ly = extents[1];
    grid_spec.Lz = extents[2];
    LippmannSchwingerSolver solver(grid_spec, verbose);
    int its = solver.apply(lambda, mu, epsilon_bar,
                           delta_epsilon_initial,
                           epsilon, sigma,
                           rtol, atol, maxits);
    return its;
  }

  int lippmann_schwinger_solve_anisotropic(float *stiffness,
                                           float *epsilon_bar,
                                           float *delta_epsilon_initial,
                                           const float lambda_ref,
                                           const float mu_ref,
                                           float *epsilon,
                                           float *sigma,
                                           int *voxels,
                                           float *extents,
                                           float rtol, float atol,
                                           int maxits,
                                           int verbose)
  {
    GridSpec grid_spec;
    grid_spec.nx = voxels[0];
    grid_spec.ny = voxels[1];
    grid_spec.nz = voxels[2];
    grid_spec.Lx = extents[0];
    grid_spec.Ly = extents[1];
    grid_spec.Lz = extents[2];
    LippmannSchwingerAnisotropicSolver solver(grid_spec, verbose);
    int its = solver.apply(stiffness, epsilon_bar,
                           delta_epsilon_initial,
                           lambda_ref, mu_ref,
                           epsilon, sigma,
                           rtol, atol, maxits);
    return its;
  }
}