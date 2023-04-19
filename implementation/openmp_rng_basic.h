#ifndef _OPENMP_RNG_BASIC_H
#define _OPENMP_RNG_BASIC_H

#include <iostream>
#include <cassert>
#include <random>

//#include <omp.h>

#include "useful_enum.h"

void omp_get_rng_uniform_uint(unsigned int* data_h,
                              const size_t sz,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::mt19937 && "Error: OpenMP-RNG-BASIC only support mt19937 generator type!\n");

  std::mt19937 generator(seed);
  for(int i=0; i<sz; i++)
  {
    data_h[i] = generator();
  }
}

void omp_get_rng_uniform_float(float* data_h,
                               const size_t sz,
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::mt19937 && "Error: OpenMP-RNG-BASIC only support mt19937 generator type!\n");

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution (-1.0, 0.0);
  for(int i=0; i<sz; i++)
  {
    data_h[i] = -distribution(generator);
  }
}

void omp_get_rng_uniform_double(double* data_h,
                                const size_t sz,
                                unsigned long long seed,
                                const generator_enum rng_type_enum,
                                const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::mt19937 && "Error: OpenMP-RNG-BASIC only support mt19937 generator type!\n");

  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution (-1.0, 0.0);
  for(int i=0; i<sz; i++)
  {
    data_h[i] = -distribution(generator);
  }
}

void omp_get_rng_normal_float(float* data_h,
                              const size_t sz,
                              float mean, float stddev,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::mt19937 && "Error: OpenMP-RNG-BASIC only support mt19937 generator type!\n");

  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(mean, stddev);
  for(int i=0; i<sz; i++)
  {
    data_h[i] = distribution(generator);
  }
}

void omp_get_rng_normal_double(double* data_h,
                               const size_t sz,
                               double mean, double stddev,
                               unsigned long long seed,
                               const generator_enum rng_type_enum,
                               const size_t offset, const size_t dimensions)
{
  assert(rng_type_enum == generator_enum::mt19937 && "Error: OpenMP-RNG-BASIC only support mt19937 generator type!\n");

  std::mt19937 generator(seed);
  std::normal_distribution<double> distribution(mean, stddev);
  for(int i=0; i<sz; i++)
  {
    data_h[i] = distribution(generator);
  }
}

#endif
