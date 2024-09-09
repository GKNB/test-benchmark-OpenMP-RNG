#include<iostream>
#include<cstdlib>
#include<vector>
#include<iomanip>

#include <omp.h>

#include "openmp_rng.h"

#define MYMALLOC(T, S) T* data = (T*)malloc(sizeof(T) * (S));

int main()
{
  int ntrials = 1;
  size_t sz = 64;

#if defined(ARCH_CUDA)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#elif defined(ARCH_HIP)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#elif defined(USE_RANDOM123)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#else
  std::vector<generator_enum> gen_type = {generator_enum::mt19937};
#endif

  double tt = 0;
  //Test the most general usage
  for(int i=0; i<gen_type.size(); i++)
  {
    std::cout << "Start to do gen_type " << gen_type[i] << std::endl;
    if(gen_type[i] != generator_enum::sobol64)
    {	
      std::cout << "Benchmark uniform_uint" << std::endl;
      MYMALLOC(unsigned int, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(to:data[0:sz])
#endif  
      {
      //warm-up
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_uint(data, sz, 1234ull, gen_type[i]);

        tt = -omp_get_wtime();
        for(int j=0; j<ntrials; j++)
        {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
          omp_get_rng_uniform_uint(data, sz, 1234ull, gen_type[i]);
        }
        tt += omp_get_wtime();
      }
      for(int i=0; i<sz; i++)
      {
        std::cout << data[i] << std::endl;
      }
    }
    {	
      std::cout << "Benchmark uniform_float" << std::endl;
      MYMALLOC(float, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
        //warm-up
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_float(data, sz, 1234ull, gen_type[i]);
        tt = -omp_get_wtime();
        for(int j=0; j<ntrials; j++)
        {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
          omp_get_rng_uniform_float(data, sz, 1234ull, gen_type[i]);
        }
        tt += omp_get_wtime();
      }
      for(int i=0; i<sz; i++)
      {
        std::cout << data[i] << std::endl;
      }
    }
    {	
      std::cout << "Benchmark uniform_double" << std::endl;
      MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
        //warm-up
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_double(data, sz, 1234ull, gen_type[i]);
        tt = -omp_get_wtime();
        for(int j=0; j<ntrials; j++)
        {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
          omp_get_rng_uniform_double(data, sz, 1234ull, gen_type[i]);
        }
        tt += omp_get_wtime();
      }
      for(int i=0; i<sz; i++)
      {
        std::cout << data[i] << std::endl;
      }
    }
    {	
      std::cout << "Benchmark normal_float" << std::endl;
      MYMALLOC(float, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
        //warm-up
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_normal_float(data, sz, 0.0f, 1.0f, 1234ull, gen_type[i]);
        tt = -omp_get_wtime();
        for(int j=0; j<ntrials; j++)
        {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
          omp_get_rng_normal_float(data, sz, 0.0f, 1.0f, 1234ull, gen_type[i]);
        }
        tt += omp_get_wtime();
      }
      for(int i=0; i<sz; i++)
      {
        std::cout << data[i] << std::endl;
      }
    }
    {	
      std::cout << "Benchmark normal_double" << std::endl;
      MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
        //warm-up
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_normal_double(data, sz, 0.0, 1.0, 1234ull, gen_type[i]);
        tt = -omp_get_wtime();
        for(int j=0; j<ntrials; j++)
        {
#if defined(ARCH_CUDA) || defined(ARCH_HIP)
  #pragma omp target data use_device_ptr(data)
#endif
          omp_get_rng_normal_double(data, sz, 0.0, 1.0, 1234ull, gen_type[i]);
        }
        tt += omp_get_wtime();
      }
      for(int i=0; i<sz; i++)
      {
        std::cout << data[i] << std::endl;
      }
    }
  }
}
