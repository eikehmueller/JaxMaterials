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

/** @brief Base class for Lippmann Schwinger solvers
 *
 * Provides functionality for solving the equations of linear elasticity on a fixed computational
 * grid.
 */
class LippmannSchwingerSolverBase
{
public:
    /** @brief Constructor
     *
     * Create new instance,;initialise all state variables and allocate required memory
     *
     * @param[in] grid_spec specification of computational grid
     * @param[in] verbose verbosity level: 0 = no output, 1 = print summary, >1 = print at every iteration
     */
    LippmannSchwingerSolverBase(const GridSpec grid_spec, const int verbose = 0);

    /** @brief Destructor
     *
     * Free all allocated memory
     */
    virtual ~LippmannSchwingerSolverBase();

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
     * @param[in] delta_epsilon_initial non-constant part of initial strain (device array of size 6*nvoxels)
     */
    void set_epsilon_initial(float *__restrict__ dev_epsilon,
                             float *__restrict__ dev_epsilon_bar,
                             float *__restrict__ dev_delta_epsilon_initial);

    /* Class variables */
    /** @brief specification of computational grid */
    const GridSpec grid_spec;
    /** @brief verbosity level */
    const int verbose;
    /** @brief Fourier vectors */
    float *dev_xi;
    /** @brief normalised Fourier vectors */
    float *dev_xi_zero;
    /** @brief temporary for sum on device */
    float *dev_sum;
    /** @brief temporary for sum on host */
    float *sum;
    /** @brief real-valued strain epsilon on device */
    float *dev_epsilon;
    /** @brief real-valued mean strain epsilon on device */
    float *dev_epsilon_bar;
    /** @brief real-valued non-constant part of initial strain */
    float *dev_delta_epsilon_initial;
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

/** @brief Lippmann Schwinger solver for isotropic materials
 *
 * The apply routine can be called for different Lame parameters lambda, mu and different mean
 * strain values.
 *
 * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
 */
class LippmannSchwingerSolver : public LippmannSchwingerSolverBase
{
public:
    /** @brief Constructor
     *
     * Create a new isotropic Lippmann-Schwinger solver instance, initialise all
     * isotropic state variables and allocate required memory.
     *
     * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
     *
     * @param[in] grid_spec specification of computational grid
     * @param[in] verbose verbosity level: 0 = no output, 1 = print summary,
     *                    >1 = print at every iteration
     */
    LippmannSchwingerSolver(const GridSpec grid_spec, const int verbose = 0);

    /** @brief Destructor */
    ~LippmannSchwingerSolver();

    /** @brief Solve for a given set of Lame parameters and mean strain
     *
     * Apply the Lippmann-Schwinger iteration for a given set of Lame parameters
     * lambda, mu and mean strain field bar(epsilon). The equation is solved to a
     * given tolerance on the normalised divergence, as defined in
     * relative_divergence_norm().
     *
     * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
     *
     * @param[in] lambda Lame parameter lambda (host array, size nvoxels)
     * @param[in] mu Lame parameter mu (host array, size nvoxels)
     * @param[in] epsilon_bar average value of epsilon (host array, size 6)
     * @param[in] delta_epsilon_initial initial value of non-constant part of epsilon
     * @param[out] epsilon resulting strain (host array, size 6*nvoxels)
     * @param[out] sigma resulting stress (host array, size 6*nvoxels)
     * @param[in] rtol relative tolerance on normalised divergence
     * @param[in] atol absolute tolerance on normalised divergence
     * @param[in] maxits maximum number of iterations
     */
    int apply(float *__restrict__ lambda,
              float *__restrict__ mu,
              float *__restrict__ epsilon_bar,
              float *__restrict__ delta_epsilon_initial,
              float *__restrict__ epsilon,
              float *__restrict__ sigma,
              float rtol, float atol, int maxits = 100);

private:
    /** @brief Lame parameter lambda on device */
    float *dev_lambda;
    /** @brief Lame parameter mu on device */
    float *dev_mu;
};

/** @brief Lippmann Schwinger solver for anisotropic materials
 *
 * The apply routine can be called for different stiffness tensors and mean
 * strain values.
 *
 * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
 */
class LippmannSchwingerAnisotropicSolver : public LippmannSchwingerSolverBase
{
public:
    /** @brief Constructor
     *
     * Create a new anisotropic Lippmann-Schwinger solver instance, initialise
     * anisotropic state variables and allocate required memory.
     *
     * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
     *
     * @param[in] grid_spec specification of computational grid
     * @param[in] verbose verbosity level: 0 = no output, 1 = print summary,
     *                    >1 = print at every iteration
     */
    LippmannSchwingerAnisotropicSolver(const GridSpec grid_spec, const int verbose = 0);

    /** @brief Destructor */
    ~LippmannSchwingerAnisotropicSolver();

    /** @brief Solve for a given stiffness tensor and mean strain
     *
     * Apply the Lippmann-Schwinger iteration for a given stiffness tensor C and
     * mean strain field bar(epsilon). The equation is solved to a given
     * tolerance on the normalised divergence, as defined in
     * relative_divergence_norm().
     *
     * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
     *
     * @param[in] stiffness stiffness tensor C (host array, size 21*nvoxels)
     * @param[in] epsilon_bar average value of epsilon (host array, size 6)
     * @param[in] delta_epsilon_initial initial value of non-constant part of epsilon
     * @param[in] lambda_ref reference Lame parameter lambda
     * @param[in] mu_ref reference Lame parameter mu
     * @param[out] epsilon resulting strain (host array, size 6*nvoxels)
     * @param[out] sigma resulting stress (host array, size 6*nvoxels)
     * @param[in] rtol relative tolerance on normalised divergence
     * @param[in] atol absolute tolerance on normalised divergence
     * @param[in] maxits maximum number of iterations
     */
    int apply(float *__restrict__ stiffness,
              float *__restrict__ epsilon_bar,
              float *__restrict__ delta_epsilon_initial,
              const float lambda_ref,
              const float mu_ref,
              float *__restrict__ epsilon,
              float *__restrict__ sigma,
              float rtol, float atol, int maxits = 100);

private:
    /** @brief stiffness tensor on device */
    float *dev_stiffness;
};

/** @brief Solve linear elasticity problem with Lippmann-Schwinger iteration
 *
 * Provides an interface which can be called externally
 *
 * @param[in] lambda Lame parameter lambda (host array, size nvoxels)
 * @param[in] mu Lame parameter mu (host array, size nvoxels)
 * @param[in] epsilon_bar average value of epsilon (host array, size 6)
 * @param[in] delta_epsilon_initial initial value of non-constant part of epsilon
 * @param[out] epsilon Resulting strain (host array, size 6*nvoxels)
 * @param[out] sigma Resulting stress (host array, size 6*nvoxels)
 * @param[in] voxels Number of voxels (nx,ny,nz)
 * @param[in] extents Size of domain in each direction (Lx,Ly,Lz)
 * @param[in] rtol relative tolerance on normalised divergence
 * @param[in] atol absolute tolerance on normalised divergence
 * @param[in] maxits maximum number of iterations
 * @param[in] verbose verbosity level
 *
 * Returns the actual number of iterations
 */
extern "C"
{
    int lippmann_schwinger_solve_isotropic(float *lambda, float *mu,
                                           float *epsilon_bar,
                                           float *delta_epsilon_initial,
                                           float *epsilon, float *sigma,
                                           int *voxels,
                                           float *extents,
                                           float rtol, float atol, int maxits,
                                           int verbose);

    /** @brief Solve anisotropic linear elasticity problem with Lippmann-Schwinger iteration
     *
     * Provides an anisotropic interface which can be called externally.
     *
     * @note Implemented by GitHub Copilot (GPT-5.3-Codex); reviewed by Eike Mueller.
     *
     * @param[in] stiffness stiffness tensor C (host array, size 21*nvoxels)
     * @param[in] epsilon_bar average value of epsilon (host array, size 6)
     * @param[in] delta_epsilon_initial initial value of non-constant part of epsilon
     * @param[in] lambda_ref reference Lame parameter lambda
     * @param[in] mu_ref reference Lame parameter mu
     * @param[out] epsilon resulting strain (host array, size 6*nvoxels)
     * @param[out] sigma resulting stress (host array, size 6*nvoxels)
     * @param[in] voxels number of voxels (nx,ny,nz)
     * @param[in] extents size of domain in each direction (Lx,Ly,Lz)
     * @param[in] rtol relative tolerance on normalised divergence
     * @param[in] atol absolute tolerance on normalised divergence
     * @param[in] maxits maximum number of iterations
     * @param[in] verbose verbosity level
     *
     * Returns the actual number of iterations.
     */
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
                                             int verbose);
}

#endif // LIPPMANN_SCHWINGER_HH