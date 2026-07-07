/** @brief Implementation of fourier.hh */
#include "fourier.hh"

/* kernel to initialize Fourier vectors */
__global__ void initialize_xi_kernel(float *__restrict__ dev_xi,
                                     const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    size_t nz_half = nz / 2 + 1;
    float two_hx_inv = 2 * grid_spec.nx / grid_spec.Lx;
    float two_hy_inv = 2 * grid_spec.ny / grid_spec.Ly;
    float two_hz_inv = 2 * grid_spec.nz / grid_spec.Lz;
    int k_a = blockDim.z * blockIdx.z + threadIdx.z;
    int k_b = blockDim.y * blockIdx.y + threadIdx.y;
    int k_c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((k_a < nx) && (k_b < ny) && (k_c < nz_half))
    {
        float xi_0_half = M_PI * k_a / nx;
        float xi_1_half = M_PI * k_b / ny;
        float xi_2_half = M_PI * k_c / nz;
        dev_xi[FIDX(nx, ny, nz_half, 0, k_a, k_b, k_c)] = two_hx_inv *
                                                          sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
        dev_xi[FIDX(nx, ny, nz_half, 1, k_a, k_b, k_c)] = two_hy_inv *
                                                          cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
        dev_xi[FIDX(nx, ny, nz_half, 2, k_a, k_b, k_c)] = two_hz_inv *
                                                          cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
    }
}

/* Initialize Fourier vectors*/
void initialize_xi_device(float *__restrict__ dev_xi,
                          const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    dim3 grid((nz / 2 + 1 + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    initialize_xi_kernel<<<grid, block>>>(dev_xi, grid_spec);
}

/* kernel to initialize Fourier vectors */
__global__ void initialize_xizero_kernel(float *__restrict__ dev_xi_zero,
                                         const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    size_t nz_half = nz / 2 + 1;
    float two_hx_inv = 2 * grid_spec.nx / grid_spec.Lx;
    float two_hy_inv = 2 * grid_spec.ny / grid_spec.Ly;
    float two_hz_inv = 2 * grid_spec.nz / grid_spec.Lz;
    int k_a = blockDim.z * blockIdx.z + threadIdx.z;
    int k_b = blockDim.y * blockIdx.y + threadIdx.y;
    int k_c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((k_a < nx) && (k_b < ny) && (k_c < nz_half))
    {
        float xi_0_half = M_PI * k_a / nx;
        float xi_1_half = M_PI * k_b / ny;
        float xi_2_half = M_PI * k_c / nz;
        float tilde_xi_0 = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
        float tilde_xi_1 = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
        float tilde_xi_2 = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
        float tilde_xi_nrm = sqrt(tilde_xi_0 * tilde_xi_0 + tilde_xi_1 * tilde_xi_1 + tilde_xi_2 * tilde_xi_2);
        // Avoid division by zero
        const float tolerance = 1.E-6;
        if (tilde_xi_nrm < tolerance)
            tilde_xi_nrm = 1.0;
        dev_xi_zero[FIDX(nx, ny, nz_half, 0, k_a, k_b, k_c)] = tilde_xi_0 / tilde_xi_nrm;
        dev_xi_zero[FIDX(nx, ny, nz_half, 1, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
        dev_xi_zero[FIDX(nx, ny, nz_half, 2, k_a, k_b, k_c)] = tilde_xi_2 / tilde_xi_nrm;
    }
}

/* Initialize Fourier vectors*/
void initialize_xizero_device(float *__restrict__ dev_xi_zero,
                              const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    dim3 grid((nz / 2 + 1 + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X);
    dim3 block(BLOCKSIZE_Z, BLOCKSIZE_Y, BLOCKSIZE_X);
    initialize_xizero_kernel<<<grid, block>>>(dev_xi_zero, grid_spec);
}

/* kernel to initialize Fourier vectors on host */
void initialize_xizero_host(float *__restrict__ xi_zero,
                            const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    size_t nz_half = nz / 2 + 1;
    float two_hx_inv = 2 * grid_spec.nx / grid_spec.Lx;
    float two_hy_inv = 2 * grid_spec.ny / grid_spec.Ly;
    float two_hz_inv = 2 * grid_spec.nz / grid_spec.Lz;
    for (int k_a = 0; k_a < nx; ++k_a)
        for (int k_b = 0; k_b < ny; ++k_b)
            for (int k_c = 0; k_c < nz_half; ++k_c)
            {
                float xi_0_half = M_PI * k_a / nx;
                float xi_1_half = M_PI * k_b / ny;
                float xi_2_half = M_PI * k_c / nz;
                float tilde_xi_0 = two_hx_inv * sin(xi_0_half) * cos(xi_1_half) * cos(xi_2_half);
                float tilde_xi_1 = two_hy_inv * cos(xi_0_half) * sin(xi_1_half) * cos(xi_2_half);
                float tilde_xi_2 = two_hz_inv * cos(xi_0_half) * cos(xi_1_half) * sin(xi_2_half);
                float tilde_xi_nrm = sqrt(tilde_xi_0 * tilde_xi_0 + tilde_xi_1 * tilde_xi_1 + tilde_xi_2 * tilde_xi_2);
                // Avoid division by zero
                const float tolerance = 1.E-6;
                if (tilde_xi_nrm < tolerance)
                    tilde_xi_nrm = 1.0;
                xi_zero[FIDX(nx, ny, nz_half, 0, k_a, k_b, k_c)] = tilde_xi_0 / tilde_xi_nrm;
                xi_zero[FIDX(nx, ny, nz_half, 1, k_a, k_b, k_c)] = tilde_xi_1 / tilde_xi_nrm;
                xi_zero[FIDX(nx, ny, nz_half, 2, k_a, k_b, k_c)] = tilde_xi_2 / tilde_xi_nrm;
            }
}

/** @brief Compute sum of squared absolute values of complex-Hermitian Fourier array
 *
 * The array dev_u is assumed to represent a four-dimensional complex-Hermitian Fourier field of shape
 * (B,nx,ny,nz/2+1), i.e. n = B*nx*ny*(nz/2+1) entries in total. The storage format is row-major, with
 * the final index running fastest.
 *
 * This kernel computes the following sum:
 *
 *   sum_{b,i,j} ( |u_{b,i,j,0}|^2 + 2 sum_{k>0} |u_{b,i,j,k}|^2 )
 *
 * @param[in] dev_u complex-valued device array of size n
 * @param[out] dev_sum device array (of size 1) holding the final sum
 * @param[in] n size of input array dev_u
 * @param[in] nz number of modes in the z-direction
 */
__global__ void reduce_fourier_kernel(cufftComplex *dev_u, float *dev_sum, const int n, const int nz)
{
    // size of temporary memory = blocksize / warpsize
    extern __shared__ float local_sum[];
    // global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // block-local index
    int tid = threadIdx.x;
    // Set value of |u_i|^2 or 2|u_i|^2 for each thread
    float nrm2;
    if (idx < n)
    {
        float u_x = dev_u[idx].x;
        float u_y = dev_u[idx].y;
        float u_nrm2 = u_x * u_x + u_y * u_y;
        int r = idx % (nz / 2 + 1);
        if ((r == 0) or (nz % 2 == 0) and (r == nz / 2))
            nrm2 = u_nrm2;
        else
            nrm2 = 2 * u_nrm2;
    }
    else
        nrm2 = 0;
    // reduce within warp using shuffles
    for (int delta = 1; delta < warpSize; delta *= 2)
        nrm2 += __shfl_xor_sync((unsigned)-1, nrm2, delta);
    local_sum[tid / warpSize] = nrm2;
    // reduce in shared memory
    size_t nlocal = blockDim.x / warpSize;
    for (int delta = 1; delta < nlocal; delta *= 2)
    {
        __syncthreads();
        if (tid + delta < nlocal)
            local_sum[tid] += local_sum[tid + delta];
    }
    // Atomic add into global sum
    if (tid == 0)
        atomicAdd(dev_sum, local_sum[0]);
}

/** @brief Compute norm of complex-Hermitian Fourier field
 *
 *
 * The array dev_u is assumed to represent a four-dimensional complex-Hermitian Fourier field of shape
 * (B,nx,ny,nz/2+1), i.e. n = B*nx*ny*(nz/2+1) entries in total. The storage format is row-major, with
 * the final index running fastest.
 *
 * @param[in] dev_u the device array to be summed, size n
 * @param[inout] dev_sum temporary scratch space for sum on device
 * @param[inout] sum temporary scratch space for sum on host
 * @param[in] batchsize number of fields B
 * @param[in]  grid_spec Specification of computational grid
 */
float reduce_fourier(cufftComplex *dev_u, float *dev_sum, float *sum, const size_t batchsize, const GridSpec grid_spec)
{
    const size_t nmodes = grid_spec.number_of_modes();
    size_t nblocks = (batchsize * nmodes + BLOCKSIZE - 1) / BLOCKSIZE;
    CUDA_CHECK(cudaMemset(dev_sum, 0, sizeof(float)));
    reduce_fourier_kernel<<<nblocks, BLOCKSIZE, BLOCKSIZE / WARPSIZE * sizeof(float)>>>(dev_u, dev_sum, batchsize * nmodes, grid_spec.nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(sum, dev_sum, sizeof(float), cudaMemcpyHostToDevice));
    float nrm = sqrt(sum[0]);
    return nrm;
}

/* Kernel for computing stress divergence in Fourier space */
__global__ void divergence_fourier_kernel(cufftComplex *__restrict__ dev_sigma_hat,
                                          float *__restrict__ dev_xi,
                                          cufftComplex *__restrict__ dev_div_sigma_hat,
                                          const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    size_t nz_half = nz / 2 + 1;
    int k_a = blockDim.z * blockIdx.z + threadIdx.z;
    int k_b = blockDim.y * blockIdx.y + threadIdx.y;
    int k_c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((k_a < nx) && (k_b < ny) && (k_c < nz_half))
    {
        float xi[3];
        float sigma_hat_x[6];
        float sigma_hat_y[6];
        for (int alpha = 0; alpha < 3; ++alpha)
            xi[alpha] = dev_xi[FIDX(nx, ny, nz_half, alpha, k_a, k_b, k_c)];
        for (int alpha = 0; alpha < 6; ++alpha)
        {
            sigma_hat_x[alpha] = dev_sigma_hat[FIDX(nx, ny, nz_half, alpha, k_a, k_b, k_c)].x;
            sigma_hat_y[alpha] = dev_sigma_hat[FIDX(nx, ny, nz_half, alpha, k_a, k_b, k_c)].y;
        }
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 0, k_a, k_b, k_c)].x = xi[0] * sigma_hat_x[0] +
                                                                       xi[1] * sigma_hat_x[3] +
                                                                       xi[2] * sigma_hat_x[4];
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 0, k_a, k_b, k_c)].y = xi[0] * sigma_hat_y[0] +
                                                                       xi[1] * sigma_hat_y[3] +
                                                                       xi[2] * sigma_hat_y[4];
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 1, k_a, k_b, k_c)].x = xi[0] * sigma_hat_x[3] +
                                                                       xi[1] * sigma_hat_x[1] +
                                                                       xi[2] * sigma_hat_x[5];
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 1, k_a, k_b, k_c)].y = xi[0] * sigma_hat_y[3] +
                                                                       xi[1] * sigma_hat_y[1] +
                                                                       xi[2] * sigma_hat_y[5];
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 2, k_a, k_b, k_c)].x = xi[0] * sigma_hat_x[4] +
                                                                       xi[1] * sigma_hat_x[5] +
                                                                       xi[2] * sigma_hat_x[2];
        dev_div_sigma_hat[FIDX(nx, ny, nz_half, 2, k_a, k_b, k_c)].y = xi[0] * sigma_hat_y[4] +
                                                                       xi[1] * sigma_hat_y[5] +
                                                                       xi[2] * sigma_hat_y[2];
    }
}

/* compute divergence in Fourier space */
void divergence_fourier(cufftComplex *__restrict__ dev_sigma_hat,
                        cufftComplex *__restrict__ dev_div_sigma_hat,
                        float *__restrict__ dev_xi,
                        const GridSpec grid_spec)
{
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    dim3 grid((nz / 2 + 1 + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X);
    dim3 block(BLOCKSIZE_Z, BLOCKSIZE_Y, BLOCKSIZE_X);
    divergence_fourier_kernel<<<grid, block>>>(dev_sigma_hat, dev_xi, dev_div_sigma_hat, grid_spec);
}

/* kernel for Fourier solve in homogeneous isotropic reference material */
__global__ void fourier_solve_kernel(cufftComplex *__restrict__ dev_tau_hat,
                                     cufftComplex *__restrict__ dev_epsilon_hat,
                                     float *__restrict__ dev_xi_zero,
                                     const float C_A, const float C_B,
                                     const GridSpec grid_spec)
{

    float xi[3];
    cufftComplex tau_hat[6];
    cufftComplex epsilon_hat[6];
    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    size_t nz_half = nz / 2 + 1;
    int k_a = blockDim.z * blockIdx.z + threadIdx.z;
    int k_b = blockDim.y * blockIdx.y + threadIdx.y;
    int k_c = blockDim.x * blockIdx.x + threadIdx.x;
    if ((k_a < nx) && (k_b < ny) && (k_c < nz_half))
    {
        // copy into temporary arrays
        for (int mu = 0; mu < 3; ++mu)
            xi[mu] = dev_xi_zero[FIDX(nx, ny, nz_half, mu, k_a, k_b, k_c)];
        for (int mu = 0; mu < 6; ++mu)
            tau_hat[mu] = dev_tau_hat[FIDX(nx, ny, nz_half, mu, k_a, k_b, k_c)];
        cufftComplex rho;
        rho.x = xi[0] * xi[0] * tau_hat[0].x +
                xi[1] * xi[1] * tau_hat[1].x +
                xi[2] * xi[2] * tau_hat[2].x +
                2 * (xi[0] * xi[1] * tau_hat[3].x +
                     xi[0] * xi[2] * tau_hat[4].x +
                     xi[1] * xi[2] * tau_hat[5].x);
        rho.y = xi[0] * xi[0] * tau_hat[0].y +
                xi[1] * xi[1] * tau_hat[1].y +
                xi[2] * xi[2] * tau_hat[2].y +
                2 * (xi[0] * xi[1] * tau_hat[3].y +
                     xi[0] * xi[2] * tau_hat[4].y +
                     xi[1] * xi[2] * tau_hat[5].y);
        epsilon_hat[0].x = C_A * xi[0] * (xi[0] * tau_hat[0].x + xi[2] * tau_hat[4].x + xi[1] * tau_hat[3].x) +
                           C_B * rho.x * xi[0] * xi[0];
        epsilon_hat[0].y = C_A * xi[0] * (xi[0] * tau_hat[0].y + xi[2] * tau_hat[4].y + xi[1] * tau_hat[3].y) +
                           C_B * rho.y * xi[0] * xi[0];
        epsilon_hat[1].x = C_A * xi[1] * (xi[1] * tau_hat[1].x + xi[2] * tau_hat[5].x + xi[0] * tau_hat[3].x) +
                           C_B * rho.x * xi[1] * xi[1];
        epsilon_hat[1].y = C_A * xi[1] * (xi[1] * tau_hat[1].y + xi[2] * tau_hat[5].y + xi[0] * tau_hat[3].y) +
                           C_B * rho.y * xi[1] * xi[1];
        epsilon_hat[2].x = C_A * xi[2] * (xi[2] * tau_hat[2].x + xi[1] * tau_hat[5].x + xi[0] * tau_hat[4].x) +
                           C_B * rho.x * xi[2] * xi[2];
        epsilon_hat[2].y = C_A * xi[2] * (xi[2] * tau_hat[2].y + xi[1] * tau_hat[5].y + xi[0] * tau_hat[4].y) +
                           C_B * rho.y * xi[2] * xi[2];
        epsilon_hat[3].x = 0.5 * C_A * (xi[0] * xi[1] * (tau_hat[0].x + tau_hat[1].x) + (xi[0] * xi[0] + xi[1] * xi[1]) * tau_hat[3].x + xi[2] * (xi[0] * tau_hat[5].x + xi[1] * tau_hat[4].x)) + C_B * rho.x * xi[0] * xi[1];
        epsilon_hat[3].y = 0.5 * C_A * (xi[0] * xi[1] * (tau_hat[0].y + tau_hat[1].y) + (xi[0] * xi[0] + xi[1] * xi[1]) * tau_hat[3].y + xi[2] * (xi[0] * tau_hat[5].y + xi[1] * tau_hat[4].y)) + C_B * rho.y * xi[0] * xi[1];
        epsilon_hat[4].x = 0.5 * C_A * (xi[0] * xi[2] * (tau_hat[0].x + tau_hat[2].x) + (xi[0] * xi[0] + xi[2] * xi[2]) * tau_hat[4].x + xi[1] * (xi[0] * tau_hat[5].x + xi[2] * tau_hat[3].x)) + C_B * rho.x * xi[0] * xi[2];
        epsilon_hat[4].y = 0.5 * C_A * (xi[0] * xi[2] * (tau_hat[0].y + tau_hat[2].y) + (xi[0] * xi[0] + xi[2] * xi[2]) * tau_hat[4].y + xi[1] * (xi[0] * tau_hat[5].y + xi[2] * tau_hat[3].y)) + C_B * rho.y * xi[0] * xi[2];
        epsilon_hat[5].x = 0.5 * C_A * (xi[1] * xi[2] * (tau_hat[1].x + tau_hat[2].x) + (xi[1] * xi[1] + xi[2] * xi[2]) * tau_hat[5].x + xi[0] * (xi[1] * tau_hat[4].x + xi[2] * tau_hat[3].x)) + C_B * rho.x * xi[1] * xi[2];
        epsilon_hat[5].y = 0.5 * C_A * (xi[1] * xi[2] * (tau_hat[1].y + tau_hat[2].y) + (xi[1] * xi[1] + xi[2] * xi[2]) * tau_hat[5].y + xi[0] * (xi[1] * tau_hat[4].y + xi[2] * tau_hat[3].y)) + C_B * rho.y * xi[1] * xi[2];
        // copy back into solution vector
        for (int mu = 0; mu < 6; ++mu)
            dev_epsilon_hat[FIDX(nx, ny, nz_half, mu, k_a, k_b, k_c)] = epsilon_hat[mu];
    }
}

/* Fourier solve for homogeneous isotropic reference material */
void fourier_solve_device(cufftComplex *__restrict__ dev_tau_hat,
                          cufftComplex *__restrict__ dev_epsilon_hat,
                          float *__restrict__ dev_xi_zero,
                          const float lambda_0, const float mu_0,
                          const GridSpec grid_spec)
{

    size_t nx = grid_spec.nx;
    size_t ny = grid_spec.ny;
    size_t nz = grid_spec.nz;
    dim3 grid((nz / 2 + 1 + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z,
              (ny + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y,
              (nx + BLOCKSIZE_X - 1) / BLOCKSIZE_X);
    dim3 block(BLOCKSIZE_Z, BLOCKSIZE_Y, BLOCKSIZE_X);
    const float C_A = -1.0 / mu_0;
    const float C_B = (lambda_0 + mu_0) / (mu_0 * (lambda_0 + 2 * mu_0));
    fourier_solve_kernel<<<grid, block>>>(dev_tau_hat, dev_epsilon_hat, dev_xi_zero,
                                          C_A, C_B, grid_spec);
}