/** @brief Tests of derivatives */
#ifndef TEST_DERIVATIVES_HH
#define TEST_DERIVATIVES_HH TEST_DERIVATIVES_HH
#include <random>
#include <string>
#include <algorithm>
#include "derivatives.hh"
#include <gtest/gtest.h>

/** @brief Class for testing derivatives
 */
class DerivativeTest : public ::testing::TestWithParam<std::string>
{
public:
    /** @brief Constructor
     * Create a new instance */
    DerivativeTest() {}

protected:
    /** @brief initialise test parameters */
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
        tolerance = 1.E-6;
        rng.seed(7812481);
    }
    /** @brief Test backward derivative in a particular direction
     *
     * @param[in] direction direction in which the derivative is to be taken
     */
    void test_derivative(const int direction)
    {
        size_t nvoxels = grid_spec.number_of_voxels();

        // allocate host memory
        float *u = nullptr;
        float *du_dx = nullptr;
        float *du_dx_ref = nullptr;
        CUDA_CHECK(cudaMallocHost(&u, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&du_dx_ref, nvoxels * sizeof(float)));

        // initialise data
        std::generate(u, u + nvoxels, [&]()
                      { return distribution(rng); });

        // allocate device memory
        float *dev_u = nullptr;
        float *dev_du_dx = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_u, nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_du_dx, nvoxels * sizeof(float)));

        // copy data to device
        CUDA_CHECK(cudaMemcpy(dev_u, u, nvoxels * sizeof(float), cudaMemcpyDefault));

        // compute derivative
        backward_derivative_device(dev_u, dev_du_dx, direction, grid_spec);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_derivative_host(u, du_dx_ref, direction, grid_spec);

        // copy data back to host
        CUDA_CHECK(cudaMemcpy(du_dx, dev_du_dx, nvoxels * sizeof(float),
                              cudaMemcpyDefault));
        float rel_diff = relative_difference(du_dx, du_dx_ref, nvoxels);

        // free memory
        CUDA_CHECK(cudaFree(dev_u));
        CUDA_CHECK(cudaFree(dev_du_dx));
        CUDA_CHECK(cudaFreeHost(u));
        CUDA_CHECK(cudaFreeHost(du_dx));
        CUDA_CHECK(cudaFreeHost(du_dx_ref));
        EXPECT_NEAR(rel_diff, 0.0, tolerance);
    }

    /** @brief Test backward divergence*/
    void test_divergence()
    {
        size_t nvoxels = grid_spec.number_of_voxels();
        // allocate host memory
        float *sigma = nullptr;
        float *div_sigma = nullptr;
        float *div_sigma_ref = nullptr;
        CUDA_CHECK(cudaMallocHost(&sigma, 6 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&div_sigma, 3 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&div_sigma_ref, 3 * nvoxels * sizeof(float)));

        // initialise data
        std::generate(sigma, sigma + 6 * nvoxels, [&]()
                      { return distribution(rng); });

        // allocate device memory
        float *dev_sigma = nullptr;
        float *dev_div_sigma = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_sigma, 6 * nvoxels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_div_sigma, 3 * nvoxels * sizeof(float)));

        // copy data to device
        CUDA_CHECK(cudaMemcpy(dev_sigma, sigma, 6 * nvoxels * sizeof(float), cudaMemcpyDefault));

        // compute derivatives
        backward_divergence_device(dev_sigma, dev_div_sigma, grid_spec);
        CUDA_CHECK(cudaDeviceSynchronize());
        backward_divergence_host(sigma, div_sigma_ref, grid_spec);

        // copy data back to host
        CUDA_CHECK(cudaMemcpy(div_sigma, dev_div_sigma, 3 * nvoxels * sizeof(float),
                              cudaMemcpyDefault));
        float rel_diff = relative_difference(div_sigma, div_sigma_ref, 3 * nvoxels);

        // free memory
        CUDA_CHECK(cudaFree(dev_sigma));
        CUDA_CHECK(cudaFree(dev_div_sigma));
        CUDA_CHECK(cudaFreeHost(sigma));
        CUDA_CHECK(cudaFreeHost(div_sigma));
        CUDA_CHECK(cudaFreeHost(div_sigma_ref));
        EXPECT_NEAR(rel_diff, 0.0, tolerance);
    }

    /* Class variables */
    /** @brief grid specification */
    GridSpec grid_spec;
    /** @brief tolerance */
    float tolerance;
    /** @brief random number generator */
    std::default_random_engine rng;
    /** @brief random number distribution used for initialisation */
    std::normal_distribution<float> distribution;
};

INSTANTIATE_TEST_SUITE_P(Derivative, DerivativeTest, testing::Values("even", "odd"));

/** @brief Check whether derivative in x-direction agrees between device and host
 */
TEST_P(DerivativeTest, TestXDerivative) { test_derivative(0); }

/** @brief Check whether derivative in y-direction agrees between device and host
 */
TEST_P(DerivativeTest, TestYDerivative) { test_derivative(1); }

/** @brief Check whether derivative in z-direction agrees between device and host
 */
TEST_P(DerivativeTest, TestZDerivative) { test_derivative(2); }

/** @brief Check whether divergence agrees between device and host
 */
TEST_P(DerivativeTest, TestDivergence) { test_divergence(); }

#endif // TEST_DERIVATIVES_HH