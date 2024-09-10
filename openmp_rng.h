#ifndef _OPENMP_RNG_IMPL
#define _OPENMP_RNG_IMPL

//Options for _RNG_impl:
//RNG_IMPL_BASIC              :not using any external libraries, only use std::random for CPU
//RNG_IMPL_RANDOM123:         :using RANDOM123 library for CPU
//RNG_IMPL_CURAND             :using CUDA CURAND library for NVIDIA GPU
//RNG_IMPL_ROCRAND            :using HIP ROCRAND library for AMD GPU
//RNG_IMPL_MKL_C              :using Intel MKL C backend for Intel GPU

//FIXME: Later need to find the best flag/macro to be used

#if defined(USE_RANDOM123)
//Use ramdom123 version, make sure include random123 (no need to link since it is header-only)
#define RNG_IMPL_RANDOM123

#elif defined(ARCH_CUDA)
//Use cuda version, make sure to link curand
#define RNG_IMPL_CURAND

#elif defined(ARCH_HIP)
//Use hip version, make sure to link rocrand
#define RNG_IMPL_ROCRAND

#elif defined(ARCH_INTEL)
//Use intel MKL C backend, make sure to link MKL
#define RNG_IMPL_MKL_C

#else
//Default to std::random version (not parallized!)
#define RNG_IMPL_BASIC
#warning "Since we did not specify the RNG to use, we use std::random, which is not and should not be parallized!"

#endif  


#if defined(RNG_IMPL_ROCRAND)
#  include "implementation/openmp_rng_rocrand.h"
#elif defined(RNG_IMPL_CURAND)
#  include "implementation/openmp_rng_curand.h"
#elif defined(RNG_IMPL_MKL_C)
#  include "implementation/openmp_rng_mkl_c.h"
#elif defined(RNG_IMPL_RANDOM123)
#  include "implementation/openmp_rng_random123.h"
#elif defined(RNG_IMPL_BASIC)
#  include "implementation/openmp_rng_basic.h"
#else
#  error Must specify a RNG_IMPL_* in openmp_rng.h
#endif


//Important usage: Only call these function ONCE! 
//Everytime you call these functions, they will automatically update the status of the RNG
//so the subsequent calls will generate the same random number sequence unless you manually change the seed/offset
//
//Be careful about the position of the data. For example, basic/random123 implementation suggest that the data are on host, while CUDA/HIP suggest that the data are on device!

//Generate uniform distribution so that each output element is a 32-bit unsigned int where all bits are random.
void omp_get_rng_uniform_uint(unsigned int* data,
                              const size_t sz,
                              unsigned long long seed = 1234ull,
                              const generator_enum rng_type_enum = generator_enum::philox,
                              const size_t offset = 0, const size_t dimensions = 1);

//Generate standard uniform distribution between (0.0, 1.0] with type float
void omp_get_rng_uniform_float(float* data,
                               const size_t sz,
                               unsigned long long seed = 1234ull,
                               const generator_enum rng_type_enum = generator_enum::philox,
                               const size_t offset = 0, const size_t dimensions = 1);

//Generate standard uniform distribution between (0.0, 1.0] with type double
void omp_get_rng_uniform_double(double* data,
                                const size_t sz,
                                unsigned long long seed = 1234ull,
                                const generator_enum rng_type_enum = generator_enum::philox,
                                const size_t offset = 0, const size_t dimensions = 1);

//Generate standard normal distribution with type float
void omp_get_rng_normal_float(float* data,
                              const size_t sz,
                              float mean = 0.0f, float stddev = 1.0f,
                              unsigned long long seed = 1234ull,
                              const generator_enum rng_type_enum = generator_enum::philox,
                              const size_t offset = 0, const size_t dimensions = 1);

//Generate standard normal distribution with type double
void omp_get_rng_normal_double(double* data,
                               const size_t sz,
                               double mean = 0.0, double stddev = 1.0,
                               unsigned long long seed = 1234ull,
                               const generator_enum rng_type_enum = generator_enum::philox,
                               const size_t offset = 0, const size_t dimensions = 1);

#endif
