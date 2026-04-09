/** @brief Lippmann Schwinger solver */
#ifndef LIPPMANN_SCHWINGER_HH
#define LIPPMANN_SCHWINGER_HH LIPPMANN_SCHWINGER_HH

#include <algorithm>
#include "cufft.h"
#include "cublas_v2.h"
#include "common.hh"
#include "derivatives.hh"
#include "fourier.hh"
#include "hooke.hh"

/** @brief Class for Lippmann Schwinger solver
 *
 * Provides functionality for solving the equations of linear elasticity on a fixed computational
 * grid.
 *
 * The apply routine can be called for different Lame parameters lambda, mu and different mean
 * strain values.
 */
class LippmannSchwingerSolver
{
public:
    /** @brief Constructor
     *
     * Create new instance,;initialise all state variables and allocate required memory
     *
     * @param[in] grid_spec specification of computational grid
     * @param[in] verbose verbosity level: 0 = no output, 1 = print summary, >1 = print at every iteration
     */
    LippmannSchwingerSolver(const GridSpec grid_spec, const int verbose = 0);

    /** @brief Destructor
     *
     * Free all allocated memory
     */
    ~LippmannSchwingerSolver();

    /** @brief Solve for a given set of Lame parameters and mean strain
     *
     * Apply the Lippmann Schwinger iteration for a given set of Lame parameters lambda, mu
     * and mean strain field bar(epsilon). The equation is solved to a given tolerance on the
     * normalised divergence, as defined in relative_divergence_norm().
     *
     * @param[in] lambda Lame parameter lambda (host array, size nvoxels)
     * @param[in] mu Lame parameter mu (host array, size nvoxels)
     * @param[in] epsilon_bar average value of epsilon (host array, size 6)
     * @param[out] epsilon Resulting strain (host array, size 6*nvoxels)
     * @param[out] sigma Resulting stress (host array, size 6*nvoxels)
     * @param[in] rtol relative tolerance on normalised divergence
     * @param[in] atol absolute tolerance on normalised divergence
     * @param[in] maxiter maximum number of iterations
     */
    int apply(float *__restrict__ lambda,
              float *__restrict__ mu,
              float *__restrict__ epsilon_bar,
              float *__restrict__ epsilon,
              float *__restrict__ sigma,
              float rtol, float atol, int maxiter = 100);

    /** @brief Compute normalised divergence for stopping criterion in Fourier space
     *
     * Compute the relative divergence norm
     *
     *      sqrt(<||div(sigma)||^2>) / ||<sigma>||
     *
     * which in Fourier space is given by
     *
     *      sqrt(N <||xi.hat(sigma)||^2>) / ||hat(sigma)(0)||
     *
     * @param[in] dev_sigma_hat stress in Fourier space
     */
    float relative_divergence_norm(cufftComplex *__restrict__ dev_sigma_hat);

protected:
    /** @brief Increment solution
     *
     * Auxilliary function to increment
     *
     *      epsilon -> epsilon + 1/nvoxels * r
     *
     * where the factor 1/nvoxels arises since the inverse Fourier transformation
     * in cuFFT is not normalised.
     *
     * @param[inout] dev_epsilon solution (device array, size 6*nvoxels)
     * @param[in] dev_increment increment (device array, size 6*nvoxels)
     */
    void increment_solution(float *__restrict__ dev_epsilon,
                            float *__restrict__ dev_increment);

    /** @brief Set the values of epsilon to bar(epsilon) on the device
     *
     * Auxilliary function for setting epsilon to the constant value of bar(epsilon)
     * on the device.
     *
     * @param[out] dev_epsilon strain field to to set (device array of size 6*nvoxels)
     * @param[in] epsilon_bar constant mean strain field (device array of size 6)
     */
    void set_epsilon_bar(float *__restrict__ dev_epsilon,
                         float *__restrict__ dev_epsilon_bar);

    /* Class variables */
    /** @brief specification of computational grid */
    const GridSpec grid_spec;
    /** @brief verbosity level */
    const int verbose;
    /** @brief Fourier vectors */
    float *dev_xi;
    /** @brief normalised Fourier vectors */
    float *dev_xi_zero;
    /** @brief Lame parameter lamba on device */
    float *dev_lambda;
    /** @brief Lame parameter mu on device */
    float *dev_mu;
    /** @brief temporary for sum on device */
    float *dev_sum;
    /** @brief temporary for sum on host */
    float *sum;
    /** @brief real-valued strain epsilon on device */
    float *dev_epsilon;
    /** @brief real-valued mean strain epsilon on device */
    float *dev_epsilon_bar;
    /** @brief real-valued stress sigma on device */
    float *dev_sigma;
    /** @brief divergence of sigma on device */
    float *dev_div_sigma;
    /** @brief complex-valued Fourier-stress on device */
    cufftComplex *dev_sigma_hat;
    /** @brief complex-valued divergence of Fourier-stress on device */
    cufftComplex *dev_div_sigma_hat;
    /** @brief complex-valued residual on device */
    float *dev_residual;
    /** @brief complex-valued residual in Fourier space on device */
    cufftComplex *dev_residual_hat;
    /** @brief temporary storage for zero mode of sigma in Fourier space */
    cufftComplex *sigma_0;
    /** @brief cuFFT plan for forward FFT */
    cufftHandle plan_forward;
    /** @brief cuFFT plan for inverse FFT */
    cufftHandle plan_inverse;
};

/** @brief Solve linear elasticity problem with Lippmann-Schwinger iteration
 *
 * Provides an interface which can be called externally
 *
 * @param[in] lambda Lame parameter lambda (host array, size nvoxels)
 * @param[in] mu Lame parameter mu (host array, size nvoxels)
 * @param[in] epsilon_bar average value of epsilon (host array, size 6)
 * @param[out] epsilon Resulting strain (host array, size 6*nvoxels)
 * @param[out] sigma Resulting stress (host array, size 6*nvoxels)
 * @param[in] voxels Number of voxels (nx,ny,nz)
 * @param[in] extents Size of domain in each direction (Lx,Ly,Lz)
 * @param[in] rtol relative tolerance on normalised divergence
 * @param[in] atol absolute tolerance on normalised divergence
 * @param[in] maxiter maximum number of iterations
 * @param[in] verbose verbosity level
 *
 * Returns the actual number of iterations
 */
extern "C"
{
    int lippmann_schwinger_solve(float *lambda, float *mu, float *epsilon_bar,
                                 float *epsilon, float *sigma,
                                 int *voxels,
                                 float *extents,
                                 float rtol, float atol, int maxiter,
                                 int verbose);
}

#endif // LIPPMANN_SCHWINGER_HH