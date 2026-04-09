/** @brief Main test program */
#include "common.hh"
#include "tests/test_derivatives.hh"
#include "tests/test_fourier.hh"
#include "tests/test_hooke.hh"
#include "tests/test_lippmann_schwinger.hh"
#include <gtest/gtest.h>

/* Main program */
int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}