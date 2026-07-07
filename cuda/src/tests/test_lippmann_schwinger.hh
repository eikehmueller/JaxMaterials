#ifndef TEST_LIPPMANN_SCHWINGER_HH
#define TEST_LIPPMANN_SCHWINGER_HH TEST_LIPPMANN_SCHWINGER_HH
#include <random>
#include <algorithm>
#include <string>
#include <gtest/gtest.h>
#include "cufft.h"
#include "common.hh"
#include "lippmann_schwinger.hh"

class LippmannSchwingerTest : public ::testing::TestWithParam<std::string>
{
public:
  /** @Create a new instance */
  LippmannSchwingerTest() {}

protected:
  /** @brief initialise tests */
  void SetUp() override
  {
    if (GetParam() == "even")
    {
      grid_spec.nx = 48;
      grid_spec.ny = 64;
      grid_spec.nz = 32;
    }
    else
    {
      grid_spec.nx = 47;
      grid_spec.ny = 61;
      grid_spec.nz = 37;
    }
    grid_spec.Lx = 1.1;
    grid_spec.Ly = 0.9;
    grid_spec.Lz = 0.7;
  }
  /* Grid specification */
  GridSpec grid_spec;
};

INSTANTIATE_TEST_SUITE_P(LS, LippmannSchwingerTest, testing::Values("even", "odd"));

/** @brief Check whether relative divergence is computed consistently
 */
TEST_P(LippmannSchwingerTest, TestRelativeDivergence)
{
  /* Random number generator */
  std::default_random_engine rng;
  /* normal distribution */
  std::normal_distribution<float> distribution;

  size_t nvoxels = grid_spec.number_of_voxels();
  size_t nmodes = grid_spec.number_of_modes();
  cufftComplex *dev_sigma_hat = nullptr;
  float *dev_sigma = nullptr;
  float *sigma = nullptr;
  float *div_sigma = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_sigma_hat, 6 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * nvoxels * sizeof(float)));

  // Initialize sigma with random numbers
  std::generate(sigma, sigma + 6 * nvoxels, [&]()
                { return distribution(rng); });

  backward_divergence_host(sigma, div_sigma, grid_spec);
  float sigma_avg[6];
  for (int alpha = 0; alpha < 6; ++alpha)
  {
    sigma_avg[alpha] = 0;
    for (int j = 0; j < nvoxels; ++j)
      sigma_avg[alpha] += sigma[alpha * nvoxels + j];
    sigma_avg[alpha] /= nvoxels;
  }

  float reldiv_real = vector_norm(div_sigma, nvoxels) / (sqrt(nvoxels) * tensor_norm(sigma_avg, 1));
  LippmannSchwingerSolver solver(grid_spec);

  /* cuFFT plan */
  cufftHandle plan;
  int n[3] = {(int)grid_spec.nx, (int)grid_spec.ny, (int)grid_spec.nz};
  int n_fourier[3] = {(int)grid_spec.nx, (int)grid_spec.ny, (int)grid_spec.nz / 2 + 1};
  CUFFT_CHECK(cufftPlanMany(&plan, 3, n, n, 1, nvoxels, n_fourier, 1, nmodes, CUFFT_R2C, 6));
  CUDA_CHECK(cudaMemcpy(dev_sigma, sigma, 6 * nvoxels * sizeof(float), cudaMemcpyHostToDevice));
  // Fourier transform sigma
  CUFFT_CHECK(cufftExecR2C(plan, dev_sigma, dev_sigma_hat));
  CUDA_CHECK(cudaDeviceSynchronize());

  float reldiv_fourier = solver.relative_divergence_norm(dev_sigma_hat);
  float rel_diff = (reldiv_fourier - reldiv_real) / reldiv_real;
  // free memory
  CUDA_CHECK(cudaFree(dev_sigma));
  CUDA_CHECK(cudaFree(dev_sigma_hat));
  CUDA_CHECK(cudaFreeHost(sigma));
  CUDA_CHECK(cudaFreeHost(div_sigma));
  CUFFT_CHECK(cufftDestroy(plan));
  float tolerance = 1.E-4;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/** @brief Check whether solver converges in zero iterations for homogeneous materials
 */
TEST_P(LippmannSchwingerTest, TestHomogeneousMaterial)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  float *mu = nullptr;
  float *lambda = nullptr;
  float *epsilon = nullptr;
  float *sigma = nullptr;
  float epsilon_bar[6] = {1.0, 0.4, 0.3, 0.1, -0.4, 0.7};
  CUDA_CHECK(cudaMallocHost(&mu, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&lambda, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
  std::fill(mu, mu + nvoxels, 1.2);
  std::fill(lambda, lambda + nvoxels, 1.2);

  LippmannSchwingerSolver solver(grid_spec);
  float rtol = 1.E-20;
  float atol = 1.E-4;
  int iter = solver.apply(lambda, mu, epsilon_bar, epsilon, sigma, rtol, atol);
  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
  EXPECT_EQ(iter, 0);
}

/** @brief Check whether solver converges
 */
TEST_P(LippmannSchwingerTest, TestConvergence)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  float *mu = nullptr;
  float *lambda = nullptr;
  float *epsilon = nullptr;
  float *sigma = nullptr;
  float *div_sigma = nullptr;
  float epsilon_bar[6] = {1.0, 0.4, 0.3, 0.1, -0.4, 0.7};
  CUDA_CHECK(cudaMallocHost(&mu, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&lambda, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * nvoxels * sizeof(float)));

  std::default_random_engine rng;
  std::uniform_real_distribution<float> distribution(0.8, 0.9);
  std::generate(mu, mu + nvoxels, [&]()
                { return distribution(rng); });
  std::generate(lambda, lambda + nvoxels, [&]()
                { return distribution(rng); });

  LippmannSchwingerSolver solver(grid_spec);
  float rtol = 1.E-20;
  float atol = 1.E-4;
  int iter = solver.apply(lambda, mu, epsilon_bar, epsilon, sigma, rtol, atol);

  // normalised divergence
  backward_divergence_host(sigma, div_sigma, grid_spec);
  float div_sigma_nrm = vector_norm(div_sigma, nvoxels);
  float sigma_avg[6];
  for (int alpha = 0; alpha < 6; ++alpha)
  {
    sigma_avg[alpha] = 0;
    for (int j = 0; j < nvoxels; ++j)
      sigma_avg[alpha] += sigma[alpha * nvoxels + j];
    sigma_avg[alpha] /= nvoxels;
  }
  float sigma_avg_nrm = tensor_norm(sigma_avg, 1);

  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
  CUDA_CHECK(cudaFreeHost(div_sigma));
  // Check that number of iterations is small
  EXPECT_LT(iter, 10);
  EXPECT_LT(div_sigma_nrm / sigma_avg_nrm, 1.E-2);
}

/** @brief Check anisotropic and isotropic solvers agree for isotropic materials
 *
 * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
 */
TEST_P(LippmannSchwingerTest, TestAnisotropicMatchesIsotropic)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  float *mu = nullptr;
  float *lambda = nullptr;
  float *stiffness = nullptr;
  float *epsilon_isotropic = nullptr;
  float *sigma_isotropic = nullptr;
  float *epsilon_anisotropic = nullptr;
  float *sigma_anisotropic = nullptr;
  float epsilon_bar[6] = {1.0, 0.4, 0.3, 0.1, -0.4, 0.7};
  CUDA_CHECK(cudaMallocHost(&mu, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&lambda, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&stiffness, 21 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon_isotropic, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma_isotropic, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&epsilon_anisotropic, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma_anisotropic, 6 * nvoxels * sizeof(float)));

  std::default_random_engine rng;
  std::uniform_real_distribution<float> distribution(0.8, 0.9);
  std::generate(mu, mu + nvoxels, [&]()
                { return distribution(rng); });
  std::generate(lambda, lambda + nvoxels, [&]()
                { return distribution(rng); });

  for (size_t ell = 0; ell < nvoxels; ++ell)
  {
    float lambda_ell = lambda[ell];
    float mu_ell = mu[ell];
    stiffness[0 * nvoxels + ell] = 2 * mu_ell + lambda_ell;
    stiffness[1 * nvoxels + ell] = 2 * mu_ell + lambda_ell;
    stiffness[2 * nvoxels + ell] = 2 * mu_ell + lambda_ell;
    stiffness[3 * nvoxels + ell] = mu_ell;
    stiffness[4 * nvoxels + ell] = mu_ell;
    stiffness[5 * nvoxels + ell] = mu_ell;
    stiffness[6 * nvoxels + ell] = lambda_ell;
    stiffness[7 * nvoxels + ell] = lambda_ell;
    stiffness[8 * nvoxels + ell] = lambda_ell;
    for (int alpha = 9; alpha < 21; ++alpha)
      stiffness[alpha * nvoxels + ell] = 0.0f;
  }

  LippmannSchwingerSolver isotropic_solver(grid_spec);
  LippmannSchwingerAnisotropicSolver anisotropic_solver(grid_spec);
  float rtol = 1.E-20;
  float atol = 1.E-4;
  int iter_isotropic = isotropic_solver.apply(lambda, mu, epsilon_bar,
                                              epsilon_isotropic, sigma_isotropic,
                                              rtol, atol);
  int iter_anisotropic = anisotropic_solver.apply(stiffness, epsilon_bar,
                                                  epsilon_anisotropic, sigma_anisotropic,
                                                  rtol, atol);

  float rel_diff_epsilon = relative_difference(epsilon_anisotropic,
                                               epsilon_isotropic,
                                               6 * nvoxels);
  float rel_diff_sigma = relative_difference(sigma_anisotropic,
                                             sigma_isotropic,
                                             6 * nvoxels);

  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(stiffness));
  CUDA_CHECK(cudaFreeHost(epsilon_isotropic));
  CUDA_CHECK(cudaFreeHost(sigma_isotropic));
  CUDA_CHECK(cudaFreeHost(epsilon_anisotropic));
  CUDA_CHECK(cudaFreeHost(sigma_anisotropic));

  EXPECT_LE(abs(iter_anisotropic - iter_isotropic), 1);
  EXPECT_NEAR(rel_diff_epsilon, 0.0, 2.E-5);
  EXPECT_NEAR(rel_diff_sigma, 0.0, 2.E-5);
}

#endif // TEST_LIPPMANN_SCHWINGER_HH