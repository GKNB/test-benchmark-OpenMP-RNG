#ifndef _OPENMP_RNG_MKL_C_H
#define _OPENMP_RNG_MKL_C_H

#include <iostream>
#include <cassert>
#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_omp_offload.h>

#include <omp.h>

#include "useful_enum.h"


//FIXME: Need to figure out if this is correct
//#define MKL_CHECK(condition)                                    \
//  {                                                             \
//    if(condition != VSL_ERROR_OK && condition != VSL_STATUS_OK) \
//    {                                                           \
//        std::cout << "MKL VSL error: " << condition << " line: " << __LINE__ << std::endl; \
//        exit(condition); \
//    } \
//  }


//FIXME: Need to figure out the type and replace auto with that type
inline auto get_rng_type(const generator_enum rng_type_enum)
{
  auto rng_type = VSL_BRNG_PHILOX4X32X10;
  switch(rng_type_enum)
  {
    case generator_enum::philox:
      rng_type = VSL_BRNG_PHILOX4X32X10;
      break;
    case generator_enum::mrg32k3a:
      rng_type = VSL_BRNG_MRG32K3A;
      break;
//Need to test if the following two rng works on GPU      
    case generator_enum::sobol32:
      rng_type = VSL_BRNG_SOBOL;
      break;
    case generator_enum::mt19937:
      rng_type = VSL_BRNG_MT19937;
      break;
    default:
      std::cerr << "RNG type: " << rng_type << " is not supported in MKL-C backend" << std::endl;
      assert(0);
  }
  return rng_type;
}

void omp_get_rng_uniform_uint(unsigned int* data_d, 
         			                const size_t sz,
                              unsigned long long seed,
         			                const generator_enum rng_type_enum,
         			                const size_t offset, const size_t dimensions)
{
  VSLStreamStatePtr stream;
  auto vsl_rng_type = get_rng_type(rng_type_enum);
    
#pragma omp target variant dispatch
  {
    vslNewStream(&stream, vsl_rng_type, seed);
  }

#pragma omp target variant dispatch use_device_ptr(data_d)
  {
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, sz, data_d, 0, INT_MAX);
  }

#pragma omp target variant dispatch  
  {
    vslDeleteStream(&stream);
  }
}


void omp_get_rng_uniform_float(float* data_d, 
         			                 const size_t sz, 
                               unsigned long long seed,
         			                 const generator_enum rng_type_enum,
         			                 const size_t offset, const size_t dimensions)
{
  VSLStreamStatePtr stream;
  auto vsl_rng_type = get_rng_type(rng_type_enum);
    
#pragma omp target variant dispatch
  {
    vslNewStream(&stream, vsl_rng_type, seed);
  }

#pragma omp target variant dispatch use_device_ptr(data_d)
  {
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, sz, data_d, 0.0, 1.0);
  }

#pragma omp target variant dispatch  
  {
    vslDeleteStream(&stream);
  }
}


void omp_get_rng_uniform_double(double* data_d, 
         			                  const size_t sz, 
                                unsigned long long seed,
         			                  const generator_enum rng_type_enum,
         			                  const size_t offset, const size_t dimensions)
{
  VSLStreamStatePtr stream;
  auto vsl_rng_type = get_rng_type(rng_type_enum);
    
#pragma omp target variant dispatch
  {
    vslNewStream(&stream, vsl_rng_type, seed);
  }

#pragma omp target variant dispatch use_device_ptr(data_d)
  {
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, sz, data_d, 0.0, 1.0);
  }

#pragma omp target variant dispatch  
  {
    vslDeleteStream(&stream);
  }
}


void omp_get_rng_normal_float(float* data_d, 
         			                const size_t sz, 
                              float mean, float stddev,
                              unsigned long long seed,
         			                const generator_enum rng_type_enum,
         			                const size_t offset, const size_t dimensions)
{
  VSLStreamStatePtr stream;
  auto vsl_rng_type = get_rng_type(rng_type_enum);
    
#pragma omp target variant dispatch
  {
    vslNewStream(&stream, vsl_rng_type, seed);
  }

#pragma omp target variant dispatch use_device_ptr(data_d)
  {
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, sz, data_d, mean, stddev);
  }

#pragma omp target variant dispatch  
  {
    vslDeleteStream(&stream);
  }
}

void omp_get_rng_normal_double(double* data_d, 
         			                 const size_t sz, 
                               double mean, double stddev,
                               unsigned long long seed,
         			                 const generator_enum rng_type_enum,
         			                 const size_t offset, const size_t dimensions)
{
  VSLStreamStatePtr stream;
  auto vsl_rng_type = get_rng_type(rng_type_enum);
    
#pragma omp target variant dispatch
  {
    vslNewStream(&stream, vsl_rng_type, seed);
  }

#pragma omp target variant dispatch use_device_ptr(data_d)
  {
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, sz, data_d, mean, stddev);
  }

#pragma omp target variant dispatch  
  {
    vslDeleteStream(&stream);
  }
}

#endif
