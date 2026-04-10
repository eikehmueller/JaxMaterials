/** @brief Test Fourier methods */
#ifndef TEST_FOURIER_HH
#define TEST_FOURIER_HH TEST_FOURIER_HH
#include "common.hh"
#include "cufft.h"
#include "derivatives.hh"
#include "fourier.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <string>

class FourierTest : public ::testing::TestWithParam<std::string>
{
public:
  /** @brief Constructor
   *
   * Create a new instance */
  FourierTest() {}

protected:
  /** @brief Initialise tests */
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
    size_t nvoxels = grid_spec.number_of_voxels();
    size_t nmodes = grid_spec.number_of_modes();

    // allocate memory
    CUDA_CHECK(cudaMallocHost(&xi_zero, 3 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&tau, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&tau_hat, 6 * nmodes * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(&epsilon_hat, 6 * nmodes * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(&epsilon, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&div_sigma, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&acoustic_tensor, 9 * nmodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_acoustic_tensor, 9 * nmodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_xi_zero, 3 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_tau, 6 * nvoxels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_tau_hat, 6 * nmodes * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_epsilon_hat, 6 * nmodes * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&dev_epsilon, 6 * nvoxels * sizeof(float)));

    // Initialise Fourier vectors
    initialize_xizero_host(xi_zero, grid_spec);
    initialize_xizero_device(dev_xi_zero, grid_spec);
    int n[3] = {(int)grid_spec.nx, (int)grid_spec.ny, (int)grid_spec.nz};
    int n_fourier[3] = {(int)grid_spec.nx, (int)grid_spec.ny,
                        (int)grid_spec.nz / 2 + 1};
    CUFFT_CHECK(cufftPlanMany(&plan_forward, 3, n, n, 1, nvoxels, n_fourier, 1,
                              nmodes, CUFFT_R2C, 6));
    CUFFT_CHECK(cufftPlanMany(&plan_inverse, 3, n, n_fourier, 1, nmodes, n, 1,
                              nvoxels, CUFFT_C2R, 6));
  }
  void TearDown() override
  {
    // free memory
    CUDA_CHECK(cudaFree(dev_xi_zero));
    CUDA_CHECK(cudaFree(dev_tau_hat));
    CUDA_CHECK(cudaFree(dev_tau));
    CUDA_CHECK(cudaFree(dev_epsilon_hat));
    CUDA_CHECK(cudaFree(dev_epsilon));
    CUDA_CHECK(cudaFreeHost(xi_zero));
    CUDA_CHECK(cudaFreeHost(tau));
    CUDA_CHECK(cudaFreeHost(tau_hat));
    CUDA_CHECK(cudaFreeHost(epsilon_hat));
    CUDA_CHECK(cudaFreeHost(epsilon));
    CUDA_CHECK(cudaFreeHost(sigma));
    CUDA_CHECK(cudaFreeHost(div_sigma));
    CUDA_CHECK(cudaFreeHost(acoustic_tensor));
    CUDA_CHECK(cudaFree(dev_acoustic_tensor));
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUFFT_CHECK(cufftDestroy(plan_inverse));
  }
  /* Grid specification */
  GridSpec grid_spec;
  /* constant reference Lame parameters */
  const float lambda_0 = 0.8;
  const float mu_0 = 1.3;
  /* Fourier vector on host */
  float *xi_zero;
  /* Fourier vector on device */
  float *dev_xi_zero;
  /* right hand side tau on host */
  float *tau;
  /* right hand side tau on host */
  float *dev_tau;
  /* right hand side tau on device */
  cufftComplex *dev_tau_hat;
  /* Fourier-transform of tau */
  cufftComplex *tau_hat;
  /* Fourier transform of strain tensor on device */
  cufftComplex *dev_epsilon_hat;
  /* train tensor on device */
  float *dev_epsilon;
  /* Fourier transform of strain tensor on host */
  cufftComplex *epsilon_hat;
  /* Strain tensor on host */
  float *epsilon;
  /* Stress tensor on host */
  float *sigma;
  /* Divergence of stress tensor on host */
  float *div_sigma;
  /* Acoustic tensor on host */
  float *acoustic_tensor;
  /* Acoustic tensor on device */
  float *dev_acoustic_tensor;
  /* Random number generator */
  std::default_random_engine rng;
  /* normal distribution */
  std::normal_distribution<float> distribution;
  /* cuFFT plans */
  cufftHandle plan_forward;
  cufftHandle plan_inverse;
};

INSTANTIATE_TEST_SUITE_P(Fourier, FourierTest, testing::Values("even", "odd"));

/** @brief Check whether xi-zero is constructed consistently on device and host
 */
TEST_P(FourierTest, TestXiZero)
{
  float tolerance = 1.E-6;
  // halo size
  size_t nvoxels = grid_spec.number_of_voxels();
  // allocate host memory
  float *xi_zero_ref = nullptr;
  CUDA_CHECK(cudaMallocHost(&xi_zero_ref, 3 * nvoxels * sizeof(float)));
  initialize_xizero_host(xi_zero_ref, grid_spec);

  CUDA_CHECK(cudaMemcpy(xi_zero, dev_xi_zero, 3 * nvoxels * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float rel_diff = relative_difference(xi_zero, xi_zero_ref, 3 * nvoxels);

  // Free memory
  CUDA_CHECK(cudaFreeHost(xi_zero_ref));
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether norm of complex-Hermitian Fourier vector is computed correctly
 */
TEST_P(FourierTest, TestFourierNorm)
{
  // allocate memory
  float *sum;
  float *dev_sum;
  // allocate memory
  cudaMallocHost(&sum, sizeof(float));
  cudaMalloc(&dev_sum, sizeof(float));
  size_t nmodes = grid_spec.number_of_modes();
  size_t batchsize = 6;
  std::generate(epsilon_hat, epsilon_hat + 6 * nmodes, [&]()
                {
    cufftComplex z;
    z.x = distribution(rng);
    z.y = distribution(rng);
    return z; });

  // copy data to device
  CUDA_CHECK(cudaMemcpy(dev_epsilon_hat, epsilon_hat,
                        batchsize * nmodes * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice));
  float sum_f =
      reduce_fourier(dev_epsilon_hat, dev_sum, sum, batchsize, grid_spec);
  float s = 0;
  for (int i = 0; i < batchsize * nmodes; ++i)
  {
    float nrm2 = epsilon_hat[i].x * epsilon_hat[i].x +
                 epsilon_hat[i].y * epsilon_hat[i].y;
    int r = i % (grid_spec.nz / 2 + 1);
    if ((r == 0) or (grid_spec.nz % 2 == 0) and (r == grid_spec.nz / 2))
      s += nrm2;
    else
      s += 2 * nrm2;
  }
  float rel_diff = abs((sqrt(s) - sum_f) / sqrt(s));
  // free memory
  CUDA_CHECK(cudaFreeHost(sum));
  CUDA_CHECK(cudaFree(dev_sum));
  float tolerance = 1E-4;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether divergence computation is consistent in Fourier- and real space
 */
TEST_P(FourierTest, TestFourierDivergence)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  size_t nmodes = grid_spec.number_of_modes();
  float *dev_xi = nullptr;
  float *div_epsilon = nullptr;
  cufftComplex *dev_div_epsilon_hat = nullptr;
  cufftComplex *div_epsilon_hat = nullptr;
  cufftComplex *div_epsilon_hat_full = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_xi, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&dev_div_epsilon_hat, 3 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&div_epsilon, 3 * nvoxels * sizeof(float)));
  CUDA_CHECK(
      cudaMallocHost(&div_epsilon_hat, 3 * nmodes * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMallocHost(&div_epsilon_hat_full,
                            3 * nvoxels * sizeof(cufftComplex)));

  // Initialize epsilon with random numbers
  std::generate(epsilon, epsilon + 6 * nvoxels,
                [&]()
                { return distribution(rng); });

  // compute divergence of epsilon in real space
  backward_divergence_host(epsilon, div_epsilon, grid_spec);

  CUDA_CHECK(cudaMemcpy(dev_epsilon, epsilon, 6 * nvoxels * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Fourier transform epsilon
  CUFFT_CHECK(cufftExecR2C(plan_forward, dev_epsilon, dev_epsilon_hat));
  CUDA_CHECK(cudaDeviceSynchronize());

  // compute divergence of hat(epsilon) in Fourier space on device
  initialize_xi_device(dev_xi, grid_spec);
  divergence_fourier(dev_epsilon_hat, dev_div_epsilon_hat, dev_xi, grid_spec);
  cudaDeviceSynchronize();
  // Copy back to host
  CUDA_CHECK(cudaMemcpy(div_epsilon_hat, dev_div_epsilon_hat,
                        3 * nmodes * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost));

  // Copy 'half' Fourier array to 'full' Fourier array
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz = grid_spec.nz;
  for (int alpha = 0; alpha < 3; ++alpha)
  {
    for (int i = 0; i < nx; ++i)
    {
      for (int j = 0; j < ny; ++j)
      {
        div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, 0)].x =
            div_epsilon_hat[FIDX(nx, ny, nz / 2 + 1, alpha, i, j, 0)].x;
        div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, 0)].y =
            div_epsilon_hat[FIDX(nx, ny, nz / 2 + 1, alpha, i, j, 0)].y;
        for (int k = 1; k < nz / 2 + 1; ++k)
        {
          cufftComplex z{
              div_epsilon_hat[FIDX(nx, ny, nz / 2 + 1, alpha, i, j, k)].x,
              div_epsilon_hat[FIDX(nx, ny, nz / 2 + 1, alpha, i, j, k)].y};
          div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, k)].x = z.x;
          div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, k)].y = z.y;
          div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, nz - k)].x = z.x;
          div_epsilon_hat_full[FIDX(nx, ny, nz, alpha, i, j, nz - k)].y = -z.y;
        }
      }
    }
  }

  // Compute norms ||D(epsilon)|| and ||xi.hat(epsilon)||
  float norm_real = vector_norm(div_epsilon, nvoxels);
  float norm_fourier =
      vector_norm(div_epsilon_hat_full, nvoxels) / sqrt(nvoxels);
  float rel_diff = abs(norm_fourier - norm_real) / norm_real;
  // Free memory
  CUDA_CHECK(cudaFreeHost(div_epsilon));
  CUDA_CHECK(cudaFree(dev_xi));
  CUDA_CHECK(cudaFree(dev_div_epsilon_hat));
  CUDA_CHECK(cudaFreeHost(div_epsilon_hat_full));
  CUDA_CHECK(cudaFreeHost(div_epsilon_hat));
  float tolerance = 1.E-4;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether div(sigma^0) = 0 for homogeneous material in real space
 *
 * Here sigma^0_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij} is obtained by solving
 * the equations of linear elasticity in a homogeneous isotropic material with
 *
 *     C^0_{ijkl} = lambda^0 (delta_{ij}delta_{kl}
 *                + mu^0 (delta_{ik}delta_{jl} + delta_{il}delta_{jk}))
 *
 * This test checks that div(sigma^0) = 0 in real space.
 *
 */
TEST_P(FourierTest, TestDivSigma)
{
  size_t nvoxels = grid_spec.number_of_voxels();
  // Initialize tau with random numbers
  std::generate(tau, tau + 6 * nvoxels, [&]()
                { return distribution(rng); });

  CUDA_CHECK(cudaMemcpy(dev_tau, tau, 6 * nvoxels * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Fourier transform the real-valued tau
  CUFFT_CHECK(cufftExecR2C(plan_forward, dev_tau, dev_tau_hat));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve on device and Fourier transform back
  fourier_solve_device(dev_tau_hat, dev_epsilon_hat, dev_xi_zero, lambda_0,
                       mu_0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUFFT_CHECK(cufftExecC2R(plan_inverse, dev_epsilon_hat, dev_epsilon));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back to host
  CUDA_CHECK(cudaMemcpy(epsilon, dev_epsilon, 6 * nvoxels * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Scale by 1/nvoxels to account for the fact that the inverse Fourier
  // transform is not normalised in cuFFT
  for (int ell = 0; ell < 6 * nvoxels; ++ell)
    epsilon[ell] *= 1.0 / nvoxels;
  // Compute stress sigma_{ij} = C^0_{ijkl} epsilon_{kl} + tau_{ij}
  for (int ell = 0; ell < nvoxels; ++ell)
  {
    float tr_epsilon = epsilon[0 * nvoxels + ell] + epsilon[1 * nvoxels + ell] +
                       epsilon[2 * nvoxels + ell];
    for (int alpha = 0; alpha < 6; ++alpha)
    {
      int idx = alpha * nvoxels + ell;
      sigma[idx] = tau[idx] + 2 * mu_0 * epsilon[idx];
      if (alpha < 3)
      {
        sigma[idx] += lambda_0 * tr_epsilon;
      }
    }
  }
  // Compute divergence
  backward_divergence_host(sigma, div_sigma, grid_spec);

  float sigma_nrm = tensor_norm(sigma, nvoxels);
  float div_nrm = vector_norm(div_sigma, nvoxels);
  float rel_diff = abs(div_nrm / sigma_nrm);
  float tolerance = 5.E-5;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

/* Check whether anisotropic acoustic tensor computation agrees with analytical
 * expression for isotropic material
 *
 * @note Implemented by GitHub Copilot (Raptor mini Preview) reviewed by Eike Mueller
 */
TEST_P(FourierTest, TestAcousticTensor)
{
  size_t nmodes = grid_spec.number_of_modes();
  // Create isotropic stiffness tensor
  float stiffness_tensor0[21];
  for (int i = 0; i < 3; ++i)
    stiffness_tensor0[i] = 2 * mu_0 + lambda_0;
  for (int i = 3; i < 6; ++i)
    stiffness_tensor0[i] = mu_0;
  for (int i = 6; i < 9; ++i)
    stiffness_tensor0[i] = lambda_0;
  for (int i = 9; i < 21; ++i)
    stiffness_tensor0[i] = 0.0f;
  // Copy to device
  float *dev_stiffness_tensor0;
  CUDA_CHECK(cudaMalloc(&dev_stiffness_tensor0, 21 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_stiffness_tensor0, stiffness_tensor0, 21 * sizeof(float), cudaMemcpyHostToDevice));
  // Compute acoustic tensor
  get_anisotropic_acoustic_tensor_device(dev_acoustic_tensor, dev_xi_zero, dev_stiffness_tensor0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Copy back
  CUDA_CHECK(cudaMemcpy(acoustic_tensor, dev_acoustic_tensor, 9 * nmodes * sizeof(float), cudaMemcpyDeviceToHost));
  // Compute reference on host
  float *acoustic_tensor_ref;
  CUDA_CHECK(cudaMallocHost(&acoustic_tensor_ref, 9 * nmodes * sizeof(float)));
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz_half = grid_spec.nz / 2 + 1;
  for (size_t i = 0; i < nx; ++i)
    for (size_t j = 0; j < ny; ++j)
      for (size_t k = 0; k < nz_half; ++k)
      {
        float xi[3];
        for (int alpha = 0; alpha < 3; ++alpha)
          xi[alpha] = xi_zero[FIDX(nx, ny, nz_half, alpha, i, j, k)];
        size_t mode_idx = FIDX(nx, ny, nz_half, 0, i, j, k);
        acoustic_tensor_ref[9 * mode_idx + 0] = mu_0 * (xi[1] * xi[1] + xi[2] * xi[2]) + (lambda_0 + 2 * mu_0) * xi[0] * xi[0]; // K^0_{00}
        acoustic_tensor_ref[9 * mode_idx + 1] = (lambda_0 + mu_0) * xi[0] * xi[1];                                              // K^0_{01}
        acoustic_tensor_ref[9 * mode_idx + 2] = (lambda_0 + mu_0) * xi[0] * xi[2];                                              // K^0_{02}
        acoustic_tensor_ref[9 * mode_idx + 3] = acoustic_tensor_ref[9 * mode_idx + 1];                                          // K^0_{10} = K^0_{01}
        acoustic_tensor_ref[9 * mode_idx + 4] = mu_0 * (xi[0] * xi[0] + xi[2] * xi[2]) + (lambda_0 + 2 * mu_0) * xi[1] * xi[1]; // K^0_{11}
        acoustic_tensor_ref[9 * mode_idx + 5] = (lambda_0 + mu_0) * xi[1] * xi[2];                                              // K^0_{12}
        acoustic_tensor_ref[9 * mode_idx + 6] = acoustic_tensor_ref[9 * mode_idx + 2];                                          // K^0_{20} = K^0_{02}
        acoustic_tensor_ref[9 * mode_idx + 7] = acoustic_tensor_ref[9 * mode_idx + 5];                                          // K^0_{21} = K^0_{12}
        acoustic_tensor_ref[9 * mode_idx + 8] = mu_0 * (xi[0] * xi[0] + xi[1] * xi[1]) + (lambda_0 + 2 * mu_0) * xi[2] * xi[2]; // K^_{22}
      }
  // Compare
  float rel_diff = relative_difference(acoustic_tensor, acoustic_tensor_ref, 9 * nmodes);
  // Free memory
  CUDA_CHECK(cudaFree(dev_stiffness_tensor0));
  CUDA_CHECK(cudaFreeHost(acoustic_tensor_ref));
  float tolerance = 1.E-6;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

TEST_P(FourierTest, TestInverseAcousticTensor)
{
  // Implemented by GitHub copilot; reviewed by Eike Mueller
  size_t nmodes = grid_spec.number_of_modes();
  // Create isotropic stiffness tensor
  float stiffness_tensor0[21];
  for (int i = 0; i < 3; ++i)
    stiffness_tensor0[i] = 2 * mu_0 + lambda_0;
  for (int i = 3; i < 6; ++i)
    stiffness_tensor0[i] = mu_0;
  for (int i = 6; i < 9; ++i)
    stiffness_tensor0[i] = lambda_0;
  for (int i = 9; i < 21; ++i)
    stiffness_tensor0[i] = 0.0f;

  // Copy to device
  float *dev_stiffness_tensor0;
  float *dev_acoustic_tensor;
  float *dev_inverse_acoustic_tensor;
  CUDA_CHECK(cudaMalloc(&dev_stiffness_tensor0, 21 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_acoustic_tensor, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_inverse_acoustic_tensor, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_stiffness_tensor0, stiffness_tensor0, 21 * sizeof(float), cudaMemcpyHostToDevice));

  // Compute acoustic tensor on device
  get_anisotropic_acoustic_tensor_device(dev_acoustic_tensor, dev_xi_zero, dev_stiffness_tensor0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute inverse acoustic tensor on device
  get_inverse_anisotropic_acoustic_tensor_device(dev_inverse_acoustic_tensor, dev_acoustic_tensor, dev_xi_zero, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back
  float *inverse_acoustic_tensor;
  CUDA_CHECK(cudaMallocHost(&inverse_acoustic_tensor, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(inverse_acoustic_tensor, dev_inverse_acoustic_tensor, 9 * nmodes * sizeof(float), cudaMemcpyDeviceToHost));

  // Compute reference on host
  float *inverse_acoustic_tensor_ref;
  CUDA_CHECK(cudaMallocHost(&inverse_acoustic_tensor_ref, 9 * nmodes * sizeof(float)));
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz_half = grid_spec.nz / 2 + 1;
  float rho = (lambda_0 + mu_0) / (lambda_0 + 2 * mu_0);
  for (size_t i = 0; i < nx; ++i)
    for (size_t j = 0; j < ny; ++j)
      for (size_t k = 0; k < nz_half; ++k)
      {
        float xi[3];
        for (int alpha = 0; alpha < 3; ++alpha)
          xi[alpha] = xi_zero[FIDX(nx, ny, nz_half, alpha, i, j, k)];

        // Check if xi_norm > 1.0e-8
        float xi_nrm_sq = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2];
        float mask = (xi_nrm_sq > 1.0e-8) ? 1.0f : 0.0f;

        size_t mode_idx = FIDX(nx, ny, nz_half, 0, i, j, k);
        float factor = mask / mu_0;

        // Analytical inverse for isotropic material
        inverse_acoustic_tensor_ref[9 * mode_idx + 0] = factor * (1.0f - rho * xi[0] * xi[0]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 1] = factor * (-rho * xi[0] * xi[1]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 2] = factor * (-rho * xi[0] * xi[2]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 3] = factor * (-rho * xi[1] * xi[0]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 4] = factor * (1.0f - rho * xi[1] * xi[1]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 5] = factor * (-rho * xi[1] * xi[2]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 6] = factor * (-rho * xi[2] * xi[0]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 7] = factor * (-rho * xi[2] * xi[1]);
        inverse_acoustic_tensor_ref[9 * mode_idx + 8] = factor * (1.0f - rho * xi[2] * xi[2]);
      }

  // Compare
  float rel_diff = relative_difference(inverse_acoustic_tensor, inverse_acoustic_tensor_ref, 9 * nmodes);

  // Free memory
  CUDA_CHECK(cudaFree(dev_stiffness_tensor0));
  CUDA_CHECK(cudaFree(dev_acoustic_tensor));
  CUDA_CHECK(cudaFree(dev_inverse_acoustic_tensor));
  CUDA_CHECK(cudaFreeHost(inverse_acoustic_tensor));
  CUDA_CHECK(cudaFreeHost(inverse_acoustic_tensor_ref));

  float tolerance = 1.E-6;
  EXPECT_NEAR(rel_diff, 0.0, tolerance);
}

TEST_P(FourierTest, TestInverseAcousticTensorAnisotropic)
{
  // Verify that K * K^{-1} = I for a general anisotropic material
  // Implementation by GitHub Copilot; reviewed by Eike Mueller
  size_t nmodes = grid_spec.number_of_modes();

  // Create random anisotropic stiffness tensor
  float stiffness_tensor0[21];
  float delta = 0.2f;
  std::uniform_real_distribution<float> real_dist(-delta, delta);
  for (int i = 0; i < 3; ++i)
    stiffness_tensor0[i] = (2 * mu_0 + lambda_0) * (1.0f + real_dist(rng));
  for (int i = 3; i < 6; ++i)
    stiffness_tensor0[i] = mu_0 * (1.0f + real_dist(rng));
  for (int i = 6; i < 9; ++i)
    stiffness_tensor0[i] = lambda_0 * (1.0f + real_dist(rng));
  for (int i = 9; i < 21; ++i)
    stiffness_tensor0[i] = real_dist(rng);

  // Copy to device
  float *dev_stiffness_tensor0;
  float *dev_acoustic_tensor;
  float *dev_inverse_acoustic_tensor;
  CUDA_CHECK(cudaMalloc(&dev_stiffness_tensor0, 21 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_acoustic_tensor, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_inverse_acoustic_tensor, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_stiffness_tensor0, stiffness_tensor0, 21 * sizeof(float), cudaMemcpyHostToDevice));

  // Compute acoustic tensor on device
  get_anisotropic_acoustic_tensor_device(dev_acoustic_tensor, dev_xi_zero, dev_stiffness_tensor0, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute inverse acoustic tensor on device
  get_inverse_anisotropic_acoustic_tensor_device(dev_inverse_acoustic_tensor, dev_acoustic_tensor, dev_xi_zero, grid_spec);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back
  float *acoustic_tensor_h;
  float *inverse_acoustic_tensor_h;
  CUDA_CHECK(cudaMallocHost(&acoustic_tensor_h, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&inverse_acoustic_tensor_h, 9 * nmodes * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(acoustic_tensor_h, dev_acoustic_tensor, 9 * nmodes * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(inverse_acoustic_tensor_h, dev_inverse_acoustic_tensor, 9 * nmodes * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify K * K^{-1} = I for each Fourier mode
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.ny;
  size_t nz_half = grid_spec.nz / 2 + 1;
  float max_error = 0.0f;

  for (size_t i = 0; i < nx; ++i)
    for (size_t j = 0; j < ny; ++j)
      for (size_t k = 0; k < nz_half; ++k)
      {
        float xi_0[3];
        for (int alpha = 0; alpha < 3; ++alpha)
          xi_0[alpha] = xi_zero[FIDX(nx, ny, nz_half, alpha, i, j, k)];
        float xi_nrm_sq = xi_0[0] * xi_0[0] + xi_0[1] * xi_0[1] + xi_0[2] * xi_0[2];

        size_t mode_idx = FIDX(nx, ny, nz_half, 0, i, j, k);
        // verify that inverse is zero for xi=0 modes
        if (xi_nrm_sq < 1.0e-8)
        {
          for (int a = 0; a < 9; ++a)
          {

            float error = abs(inverse_acoustic_tensor_h[9 * mode_idx + a]);
            max_error = fmax(max_error, error);
          }
        }
        else
        {

          // Extract 3x3 matrices
          float K[3][3], K_inv[3][3];
          for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
            {
              K[row][col] = acoustic_tensor_h[9 * mode_idx + 3 * row + col];
              K_inv[row][col] = inverse_acoustic_tensor_h[9 * mode_idx + 3 * row + col];
            }

          // Compute K * K_inv
          float product[3][3];
          for (int row = 0; row < 3; ++row)
          {
            for (int col = 0; col < 3; ++col)
            {
              product[row][col] = 0.0f;
              for (int m = 0; m < 3; ++m)
              {
                product[row][col] += K[row][m] * K_inv[m][col];
              }
            }
          }

          // Check against identity
          for (int row = 0; row < 3; ++row)
          {
            for (int col = 0; col < 3; ++col)
            {
              float expected = (row == col) ? 1.0f : 0.0f;
              float error = abs(product[row][col] - expected);
              max_error = fmax(max_error, error);
            }
          }
        }
      }

  // Free memory
  CUDA_CHECK(cudaFree(dev_stiffness_tensor0));
  CUDA_CHECK(cudaFree(dev_acoustic_tensor));
  CUDA_CHECK(cudaFree(dev_inverse_acoustic_tensor));
  CUDA_CHECK(cudaFreeHost(acoustic_tensor_h));
  CUDA_CHECK(cudaFreeHost(inverse_acoustic_tensor_h));

  // Tolerance: single precision matrix inversion can introduce larger errors
  // depending on condition number of the matrix
  float tolerance = 1.E-6;
  EXPECT_NEAR(max_error, 0.0, tolerance);
}

#endif // TEST_FOURIER_HH