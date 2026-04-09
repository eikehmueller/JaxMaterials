#ifndef TEST_TYPES_HH
#define TEST_TYPES_HH TEST_TYPES_HH

#include <gtest/gtest.h>
#include <type_traits>

template <typename T>
T add(const T x, const T y)
{
    return x + y;
}

template <typename T>
class FooTest : public testing::Test
{
};

template <typename T>
struct Tolerance
{
};

template <>
struct Tolerance<int>
{
    static const int value = 0;
};

template <>
struct Tolerance<float>
{
    static const int value = 1.E-6;
};

template <>
struct Tolerance<double>
{
    static const int value = 1.E-14;
};

using AdditionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(FooTest, AdditionTypes);

TYPED_TEST(FooTest, TestAddition)
{
    TypeParam x = 3;
    TypeParam y = 3;
    TypeParam z = add<TypeParam>(x, y);
    TypeParam tolerance = std::is_same<TypeParam, double>::value ? 1.E-12 : 1.E-4;
    printf("%e\n", tolerance);
    EXPECT_NEAR(z - (x + y), 0, tolerance);
}
#endif // TEST_TYPES_HH