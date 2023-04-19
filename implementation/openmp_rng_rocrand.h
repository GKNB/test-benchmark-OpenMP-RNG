#ifndef _OPENMP_RNG_ROCRAND_H
#define _OPENMP_RNG_ROCRAND_H

#define __HIP_PLATFORM_AMD__

#include <iostream>
#include <cassert>
#include <rocrand/rocrand.h>

#include <omp.h>

#include "useful_enum.h"

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status _status = condition;           \
    if(_status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << "ROCRAND error: " << _status << " line: " << __LINE__ << std::endl; \
        exit(_status); \
    } \
  }

inline rocrand_rng_type get_rng_type(const generator_enum rng_type_enum)
{
  rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
  switch(rng_type_enum)
  {
    case generator_enum::philox:
      rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
      break;
    case generator_enum::xorwow:
      rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
      break;
    case generator_enum::mrg32k3a:
      rng_type = ROCRAND_RNG_PSEUDO_MRG32K3A;
      break;
    case generator_enum::sobol32:
      rng_type = ROCRAND_RNG_QUASI_SOBOL32;
      break;
    case generator_enum::sobol64:
      rng_type = ROCRAND_RNG_QUASI_SOBOL64;
      break;
    case generator_enum::mtgp32:
      rng_type = ROCRAND_RNG_PSEUDO_MTGP32;
      break;
    default:
      assert(0 && "Error: rng_type set incorrectly, can not find desired rng_type_enum.\n");      
  }
  return rng_type;
}

inline void _set_up_generator(rocrand_generator &generator,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
//  double tt = -omp_get_wtime();
  rocrand_rng_type rng_type = get_rng_type(rng_type_enum);
  ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

  if(rng_type_enum != generator_enum::sobol32 && rng_type_enum != generator_enum::sobol64)
    ROCRAND_CHECK(rocrand_set_seed(generator, seed));

  if(rng_type_enum == generator_enum::sobol32 || rng_type_enum == generator_enum::sobol64)
    ROCRAND_CHECK(rocrand_set_quasi_random_generator_dimensions(generator, dimensions));

  if(rng_type_enum != generator_enum::mtgp32)
    ROCRAND_CHECK(rocrand_set_offset(generator, offset));
//  tt += omp_get_wtime();
//  std::cout << "Time for setting up generator is " << tt * 1e3 << " ms" << std::endl;
}

//FIXME: Need to figure out if device sync is needed!!!!!

void omp_get_rng_uniform_uint(unsigned int* data_d, 
                              const size_t sz, 
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum != generator_enum::sobol64 && "Error: omp_get_rng_uniform_uint does not support sobol64 generator type!\n");

  rocrand_generator generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  ROCRAND_CHECK(rocrand_generate(generator, data_d, sz));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

void omp_get_rng_uniform_float(float* data_d, 
                               const size_t sz, 
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  rocrand_generator generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  ROCRAND_CHECK(rocrand_generate_uniform(generator, data_d, sz));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

void omp_get_rng_uniform_double(double* data_d, 
                                const size_t sz, 
                                unsigned long long seed,
                                const generator_enum rng_type_enum,
                                const size_t offset, const size_t dimensions)
{
  rocrand_generator generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  ROCRAND_CHECK(rocrand_generate_uniform_double(generator, data_d, sz));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

void omp_get_rng_normal_float(float* data_d, 
                              const size_t sz, 
                              float mean, float stddev,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  rocrand_generator generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  ROCRAND_CHECK(rocrand_generate_normal(generator, data_d, sz, mean, stddev));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

void omp_get_rng_normal_double(double* data_d, 
                               const size_t sz,
                               double mean, double stddev,
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  rocrand_generator generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  ROCRAND_CHECK(rocrand_generate_normal_double(generator, data_d, sz, mean, stddev));

  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

#endif
