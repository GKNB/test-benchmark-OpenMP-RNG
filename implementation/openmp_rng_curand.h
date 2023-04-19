#ifndef _OPENMP_RNG_CURAND_H
#define _OPENMP_RNG_CURAND_H

#include <iostream>
#include <cassert>
#include <curand.h>

#include <omp.h>

#include "useful_enum.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

inline curandRngType get_rng_type(const generator_enum rng_type_enum)
{
  curandRngType rng_type = CURAND_RNG_PSEUDO_XORWOW;
  switch(rng_type_enum)
  {
    case generator_enum::philox:
      rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10;
      break;
    case generator_enum::xorwow:
      rng_type = CURAND_RNG_PSEUDO_XORWOW;
      break;
    case generator_enum::mrg32k3a:
      rng_type = CURAND_RNG_PSEUDO_MRG32K3A;
      break;
    case generator_enum::sobol32:
      rng_type = CURAND_RNG_QUASI_SOBOL32;
      break;
    case generator_enum::sobol64:
      rng_type = CURAND_RNG_QUASI_SOBOL64;
      break;
    case generator_enum::mtgp32:
      rng_type = CURAND_RNG_PSEUDO_MTGP32;
      break;
    case generator_enum::mt19937:
      rng_type = CURAND_RNG_PSEUDO_MT19937;
      break;
    default:
      assert(0 && "Error: rng_type set incorrectly, can not find desired rng_type_enum.\n");      
  }
  return rng_type;
}

inline void _set_up_generator(curandGenerator_t &generator,
                              unsigned long long seed,
                              const generator_enum rng_type_enum,
                              const size_t offset, const size_t dimensions)
{
//  double tt = -omp_get_wtime();
  curandRngType rng_type = get_rng_type(rng_type_enum);
  CURAND_CALL(curandCreateGenerator(&generator, rng_type));

  if(rng_type_enum != generator_enum::sobol32 && rng_type_enum != generator_enum::sobol64)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

  if(rng_type_enum == generator_enum::sobol32 || rng_type_enum == generator_enum::sobol64)
    CURAND_CALL(curandSetQuasiRandomGeneratorDimensions(generator, dimensions));

  if(rng_type_enum != generator_enum::mtgp32 && rng_type_enum != generator_enum::mt19937)
    CURAND_CALL(curandSetGeneratorOffset(generator, offset)); 
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

  curandGenerator_t generator;

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  CURAND_CALL(curandGenerate(generator, data_d, sz));

  CURAND_CALL(curandDestroyGenerator(generator));
}

void omp_get_rng_uniform_float(float* data_d, 
         			                 const size_t sz, 
                               unsigned long long seed,
         			                 const generator_enum rng_type_enum,
         			                 const size_t offset, const size_t dimensions)
{
  curandGenerator_t generator;	

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  CURAND_CALL(curandGenerateUniform(generator, data_d, sz));

  CURAND_CALL(curandDestroyGenerator(generator));
}


void omp_get_rng_uniform_double(double* data_d, 
         			                  const size_t sz, 
                                unsigned long long seed,
         			                  const generator_enum rng_type_enum,
         			                  const size_t offset, const size_t dimensions)
{
  curandGenerator_t generator;	

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  CURAND_CALL(curandGenerateUniformDouble(generator, data_d, sz));

  CURAND_CALL(curandDestroyGenerator(generator));
}


void omp_get_rng_normal_float(float* data_d, 
         			                const size_t sz, 
                              float mean, float stddev,
                              unsigned long long seed,
         			                const generator_enum rng_type_enum,
         			                const size_t offset, const size_t dimensions)
{
  curandGenerator_t generator;	

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  CURAND_CALL(curandGenerateNormal(generator, data_d, sz, mean, stddev));

  CURAND_CALL(curandDestroyGenerator(generator));
}

void omp_get_rng_normal_double(double* data_d, 
         			                 const size_t sz, 
                               double mean, double stddev,
                               unsigned long long seed,
         			                 const generator_enum rng_type_enum,
         			                 const size_t offset, const size_t dimensions)
{
  curandGenerator_t generator;	

  _set_up_generator(generator, seed, rng_type_enum, offset, dimensions);

  CURAND_CALL(curandGenerateNormalDouble(generator, data_d, sz, mean, stddev));

  CURAND_CALL(curandDestroyGenerator(generator));
}

#endif
