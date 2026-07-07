/** @brief Some common functionality
 *
 * Provides general definition and auxilliary functions
 */
#ifndef COMMON_HH
#define COMMON_HH COMMON_HH

#include "cufft.h"
#include <math.h>
#include <stdio.h>

// Size of a warp
#define WARPSIZE 32

// Block size used for 3d kernel launches
#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8
#define BLOCKSIZE_Z 8

// Block size used for 1d kernel launches
#define BLOCKSIZE 1024

// maximal domain size in any direction
#define NMAX 4096

// Convert 3d index to linear index
#define IDX(Nx, Ny, Nz, i, j, k) ((k) + (Nz) * ((j) + (Ny) * (i)))
// Convert (1+3)d index to linear index
#define FIDX(Nx, Ny, Nz, mu, i, j, k) \
  ((k) + (Nz) * ((j) + (Ny) * ((i) + (Nx) * (mu))))

// Error checking, as defined in CUDA Programming guide, Section 2.1.7
#define CUDA_CHECK(expr_to_check)                                      \
  do                                                                   \
  {                                                                    \
    cudaError_t result = expr_to_check;                                \
    if (result != cudaSuccess)                                         \
    {                                                                  \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, \
              __LINE__, result, cudaGetErrorString(result));           \
    }                                                                  \
  } while (0)

// Corresponding cuFFT error checking
#define CUFFT_CHECK(expr_to_check)                                           \
  do                                                                         \
  {                                                                          \
    cufftResult_t result = expr_to_check;                                    \
    if (result != CUFFT_SUCCESS)                                             \
    {                                                                        \
      fprintf(stderr, "cuFFT Runtime Error: %s:%i:%d\n", __FILE__, __LINE__, \
              result);                                                       \
    }                                                                        \
  } while (0)

// Corresponding cuBLAS error checking
#define CUBLAS_CHECK(expr_to_check)                                           \
  do                                                                          \
  {                                                                           \
    cublasStatus_t stat = expr_to_check;                                      \
    if (stat != CUBLAS_STATUS_SUCCESS)                                        \
    {                                                                         \
      fprintf(stderr, "cuBLAS Runtime Error: %s:%i:%d\n", __FILE__, __LINE__, \
              stat);                                                          \
    }                                                                         \
  } while (0)

/** @brief Specification of computational grid
 *
 * Describes grid of the domain Lx x Ly x Lz with nx, ny, nz grid voxels
 * (or voxels) in the different coordinate directions
 */
struct GridSpec
{
  /** @brief number of voxels in x-direction */
  size_t nx;
  /** @brief number of voxels in y-direction */
  size_t ny;
  /** @brief Number of voxels in z-direction */
  size_t nz;
  /** @brief extent of domain in x-direction */
  float Lx;
  /** @brief extent of domain in y-direction */
  float Ly;
  /** @brief extent of domain in z-direction */
  float Lz;
  /** @brief Return total number of voxels */
  size_t number_of_voxels() const { return nx * ny * nz; }
  /** @brief Return total number of Fourier modes */
  size_t number_of_modes() const { return nx * ny * (nz / 2 + 1); }
};

/** @brief Compute relative L2 norm
 *
 * returns ||u-u_{ref}||_2 / ||u_{ref}||_2 which can be used for testing and
 * debugging
 *
 * @param[out] u: field (host pointer)
 * @param[out] u_ref: reference field to compare to (host pointer)
 * @param[in] ndof number of unknowns */
float relative_difference(float *u, float *u_ref, const size_t ndof);

/** @brief Compute norm of real-valued vector field
 *
 * returns the norm
 *
 *    sqrt( sum_{n=0}^{nvoxels-1} ||u_n||^2 )
 *
 * where
 *
 *     ||u_n|| = sqrt( sum_{j=0}^{2} u_{n,j}^2 )
 *
 * is the norm of the vector field u_j in a given cell with index j.
 *
 * @param[in] u vector field, host array of size 3*nvoxels
 * @param[in] nvoxels number of voxels
 */
float vector_norm(float *u, const size_t nvoxels);

/** @brief Compute norm of complex-valued vector field
 *
 * returns the norm
 *
 *    sqrt( sum_{n=0}^{nvoxels-1} ||u_n||^2 )
 *
 * where
 *
 *     ||u_n|| = sqrt( sum_{j=0}^{2} |u_{n,j}|^2 )
 *
 * is the norm of the vector field u_j in a given cell with index j.
 *
 * @param[in] u vector field, host array of size 3*nvoxels
 * @param[in] nvoxels number of voxels
 */
float vector_norm(cufftComplex *u, const size_t nvoxels);

/** @brief Compute norm of real-valued tensor field in Voigt notation
 *
 * returns the norm
 *
 *    sqrt( sum_{n=0}^{nvoxels-1} ||tau_n||^2 )
 *
 * where
 *
 *     ||tau_j|| = sqrt( sum_{i!=j} tau_{n,ij}^2 )
 *               = sqrt( tau_{n,0}^2 + tau_{n,1}^2 + tau_{n,2}^2
 *                       + 2 (tau_{n,1}^2 + tau_{n,4}^2 + tau_{n,5}^2) )
 *
 * is the norm of the vector field tau_j in a given cell with index j.
 *
 * @param[in] tau tensor field in Voigt notation, host array of size 6*nvoxels
 * @param[in] nvoxels number of voxels
 */
float tensor_norm(float *tau, const size_t nvoxels);

/** @brief Compute norm of complex-valued tensor field in Voigt notation
 *
 * returns the norm
 *
 *    sqrt( sum_{n=0}^{nvoxels-1} ||tau_n||^2 )
 *
 * where
 *
 *     ||tau_j|| = sqrt( sum_{i!=j} tau_{n,ij}^2 )
 *               = sqrt( |tau_{n,0}|^2 + |tau_{n,1}|^2 + |tau_{n,2}|^2
 *                       + 2 (|tau_{n,1}|^2 + |tau_{n,4}|^2 + |tau_{n,5}|^2) )
 *
 * is the norm of the vector field tau_j in a given cell with index j.
 *
 * @param[in] tau tensor field in Voigt notation, host array of size 6*nvoxels
 * @param[in] nvoxels number of voxels
 */
float tensor_norm(cufftComplex *tau, const size_t nvoxels);

#endif // COMMON_HH
