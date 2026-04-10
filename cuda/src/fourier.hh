/** @brief Support for computations in Fourier space */
#ifndef FOURIER_HH
#define FOURIER_HH FOURIER_HH
#include "common.hh"
#include "cufft.h"

/** @brief Construct un-normalised Fourier vectors tilde{xi}_j on device
 *
 * The vectors are defined as
 *
 *    tilde(xi)_0 = 2/h_0 sin(xi_0/2) cos(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_1 = 2/h_1 cos(xi_0/2) sin(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_2 = 2/h_2 cos(xi_0/2) cos(xi_1/2) sin(xi_2/2)
 *
 * with (xi_d)_j = 2 pi k_j / N_d and k_j = 0,1,...,N_d-1
 *
 *
 * @param[out] dev_xi vectors tilde{xi} (device array, size 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xi_device(float *dev_xi, const GridSpec grid_spec);

/** @brief Construct normalised Fourier vectors ring{tilde{xi}}_j on device
 *
 * The vectors are defined as
 *
 *    tilde(xi)_0 = 2/h_0 sin(xi_0/2) cos(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_1 = 2/h_1 cos(xi_0/2) sin(xi_1/2) cos(xi_2/2)
 *    tilde(xi)_2 = 2/h_2 cos(xi_0/2) cos(xi_1/2) sin(xi_2/2)
 *
 * with (xi_d)_j = 2 pi k_j / N_d and k_j = 0,1,...,N_d-1
 *
 * We then set
 *
 *    ring{tilde{xi}}_j = tilde{xi}_j / ||tilde{xi}_j|| if ||tilde{xi}_j|| != 0
 *
 * and ring{tilde{xi}}_j = 0 if ||tilde{xi}_j|| = 0
 *
 * @param[out] dev_xi_zero vectors ring{tilde{xi}} (device array, size
 * 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xizero_device(float *dev_xi_zero, const GridSpec grid_spec);

/** @brief Construct normalised Fourier vectors ring{tilde{xi}}_j on host
 *
 * Equivalent host implementation of initialize_xizero()
 *
 * @param[out] dev_xi_zero vectors ring{tilde{xi}} (host array, size 3*nvoxels)
 * @param[in] grid_spec specification of grid
 */
void initialize_xizero_host(float *xi_zero, const GridSpec grid_spec);

/** @brief Compute sum of squared absolute values of complex-Hermitian Fourier
 * array
 *
 * The array dev_u is assumed to represent a four-dimensional complex-Hermitian
 * Fourier field of shape (B,nx,ny,nz/2+1), i.e. n = B*nx*ny*(nz/2+1) entries in
 * total. The storage format is row-major, with the final index running fastest.
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
__global__ void reduce_fourier_kernel(cufftComplex *dev_u, float *dev_sum,
                                      const int n, const int nz);

/** @brief Compute norm of complex-Hermitian Fourier field
 *
 *
 * The array dev_u is assumed to represent a four-dimensional complex-Hermitian
 * Fourier field of shape (B,nx,ny,nz/2+1), i.e. n = B*nx*ny*(nz/2+1) entries in
 * total. The storage format is row-major, with the final index running fastest.
 *
 * @param[in] dev_u the device array to be summed, size n
 * @param[inout] dev_sum temporary scratch space for sum on device
 * @param[inout] sum temporary scratch space for sum on host
 * @param[in] batchsize number of fields B
 * @param[in]  grid_spec Specification of computational grid
 */
float reduce_fourier(cufftComplex *dev_u, float *dev_sum, float *sum,
                     const size_t batchsize, const GridSpec grid_spec);

/** @brief Compute divergence in Fourier space
 *
 * Compute xi.sigma
 *
 * @param[in] dev_sigma_hat stress in Fourier space (device array, size
 * 6*Nvoxels)
 * @param[out] dev_div_sigma_hat resulting divergence xi.sigma (device array,
 * size 3*Nvoxels)
 * @param[in] dev_xi Fourier vectors (device array, size 3*Nvoxels)
 * @param[in] grid_spec specification of computational grid
 */
void divergence_fourier(cufftComplex *__restrict__ dev_sigma_hat,
                        cufftComplex *__restrict__ dev_div_sigma_hat,
                        float *__restrict__ dev_xi, const GridSpec grid_spec);

/** @brief Solve elasticity equation for homogeneous isotropic reference
 * material
 *
 * Compute hat{epsilon} = -Gamma^0 hat{tau}
 *
 * @param[in] dev_tau_hat right hand side hat{tau} (device array, size
 * 6*nvoxels)
 * @param[out] dev_epsilon_hat resulting hat{epsilon} (device array, size
 * 6*nvoxels)
 * @param[in] dev_xi_zero Fourier vectors xi_zero (device array, size 3*n),
 *                        as computed by initialize_xizero()
 * @param[in] lambda_0 Lame parameter lambda_0 of reference material
 * @param[in] mu_0 Lame parameter mu_0 of reference material
 * @param[in] grid_spec Grid specification
 */
void fourier_solve_device(cufftComplex *__restrict__ dev_tau_hat,
                          cufftComplex *__restrict__ dev_epsilon_hat,
                          float *__restrict__ dev_xi_zero, const float lambda_0,
                          const float mu_0, const GridSpec grid_spec);

/** @brief Compute anisotropic acoustic tensor
 *
 * Compute the 3x3 acoustic tensor A_{ij} = C_{ijkl} xi_k xi_l for each Fourier mode
 *
 * @note Implemented by GitHub Copilot (Raptor mini Preview); reviewed by Eike Mueller
 *
 * @param[out] dev_acoustic_tensor acoustic tensor (device array, size 9*nmodes)
 * @param[in] dev_xi_zero Fourier vectors xi_zero (device array, size 3*nmodes)
 * @param[in] dev_stiffness_tensor0 stiffness tensor of reference material (device array, size 21)
 * @param[in] grid_spec Grid specification
 */
void get_anisotropic_acoustic_tensor_device(float *dev_acoustic_tensor,
                                            float *dev_xi_zero,
                                            const float *dev_stiffness_tensor0,
                                            const GridSpec grid_spec);

/** @brief Compute inverse of the anisotropic acoustic tensor
 *
 * Invert the 3x3 acoustic tensor A_{ij} = C_{ijkl} xi_k xi_l for each Fourier mode.
 * The acoustic tensor should be pre-computed using get_anisotropic_acoustic_tensor_device()
 * and passed to this function.
 *
 * @note Implemented by GitHub Copilot; Version 2.0; reviewed by Eike Mueller
 *
 * @param[out] dev_inverse_acoustic_tensor inverse acoustic tensor (device array, size 9*nmodes)
 * @param[in] dev_acoustic_tensor acoustic tensor (device array, size 9*nmodes)
 * @param[in] dev_xi_zero Fourier vectors xi_zero (device array, size 3*nmodes)
 * @param[in] grid_spec Grid specification
 */
void get_inverse_anisotropic_acoustic_tensor_device(float *dev_inverse_acoustic_tensor,
                                                    const float *dev_acoustic_tensor,
                                                    float *dev_xi_zero,
                                                    const GridSpec grid_spec);

#endif // FOURIER_HH