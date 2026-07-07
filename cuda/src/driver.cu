#include "common.hh"
#include "lippmann_schwinger.hh"

/** @brief Initialise Lame parameters for testing
 *
 * @param[out] lambda Lame parameter lambda
 * @param[out] mu Lame parameter mu
 * @param[in] grid_spec specification of computational grid
 */
void initialize_lame_parameters(float *lambda, float *mu, const GridSpec grid_spec)
{
  size_t nx = grid_spec.nx;
  size_t ny = grid_spec.nz;
  size_t nz = grid_spec.nz;
  float x0 = 0.2;
  float y0 = 0.3;
  float r = 0.1;

  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        float x = grid_spec.Lx * (i + 0.5) / nx;
        float y = grid_spec.Ly * (j + 0.5) / ny;
        float z = grid_spec.Lz * (k + 0.5) / nz;
        if (((x - x0) * (x - x0) + (y - y0) * (y - y0)) < r * r)
        {
          lambda[IDX(nx, ny, nz, i, j, k)] = 0.2;
          mu[IDX(nx, ny, nz, i, j, k)] = 0.3;
        }
        else
        {
          lambda[IDX(nx, ny, nz, i, j, k)] = 0.8;
          mu[IDX(nx, ny, nz, i, j, k)] = 1.2;
        }
      }
}

/* M A I N */
int main(int argc, char *argv[])
{
  // domain size
  int voxels[3] = {64, 64, 64};
  float extents[3] = {1.0, 1.0, 1.0};
  GridSpec grid_spec;
  grid_spec.nx = voxels[0];
  grid_spec.ny = voxels[1];
  grid_spec.nz = voxels[2];
  grid_spec.Lx = extents[0];
  grid_spec.Ly = extents[1];
  grid_spec.Lz = extents[2];
  size_t nvoxels = grid_spec.number_of_voxels();
  float *lambda;
  float *mu;
  float epsilon_bar[6] = {1.1, 0.4, 0.2, 1.5, 0.8, 0.7};
  float *epsilon;
  float *sigma;
  CUDA_CHECK(cudaMallocHost(&lambda, nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&mu, nvoxels * sizeof(float)));
  initialize_lame_parameters(lambda, mu, grid_spec);
  CUDA_CHECK(cudaMallocHost(&epsilon, 6 * nvoxels * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
  const float rtol = 1.E-6;
  const float atol = 1.E-4;
  const int maxiter = 32;
  const int verbose = 2;
  int niter = lippmann_schwinger_solve_isotropic(lambda, mu, epsilon_bar, epsilon, sigma, voxels, extents, rtol, atol, maxiter, verbose);

  CUDA_CHECK(cudaFreeHost(lambda));
  CUDA_CHECK(cudaFreeHost(mu));
  CUDA_CHECK(cudaFreeHost(epsilon));
  CUDA_CHECK(cudaFreeHost(sigma));
}