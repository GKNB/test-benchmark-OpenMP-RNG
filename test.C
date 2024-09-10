#include<iostream>
#include<cstdlib>
#include<vector>

#include "openmp_rng.h"

#define MYMALLOC(T, S) T* data = (T*)malloc(sizeof(T) * (S));

int main()
{
  int output_interval = 10;

#if defined(ARCH_CUDA)
  std::vector<generator_enum> gen_type = {generator_enum::philox, generator_enum::xorwow, generator_enum::mrg32k3a, generator_enum::sobol32, generator_enum::sobol64, generator_enum::mtgp32, generator_enum::mt19937};
#elif defined(ARCH_HIP)
  std::vector<generator_enum> gen_type = {generator_enum::philox, generator_enum::xorwow, generator_enum::mrg32k3a, generator_enum::sobol32, generator_enum::sobol64, generator_enum::mtgp32};
#elif defined(ARCH_INTEL)
  std::vector<generator_enum> gen_type = {generator_enum::philox, generator_enum::mt19937, generator_enum::mrg32k3a, generator_enum::mtgp32};
#elif defined(USE_RANDOM123)
  std::vector<generator_enum> gen_type = {generator_enum::philox};
#else
  std::vector<generator_enum> gen_type = {generator_enum::mt19937};
#endif

  //Test the most general usage
  for(int i=0; i<gen_type.size(); i++)
  {
    std::cout << "Start to do gen_type " << i << std::endl;
    if(gen_type[i] != generator_enum::sobol64)
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(unsigned int, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_uint(data, sz, 1234ull, gen_type[i]);
      }
      std::cout << "Test uniform_uint" << std::endl;
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(float, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_float(data, sz, 1234ull, gen_type[i]);
      }
    
      std::cout << "Test uniform_float" << std::endl;
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_uniform_double(data, sz, 1234ull, gen_type[i]);
      }
      std::cout << "Test uniform_double" << std::endl;
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(float, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_normal_float(data, sz, 1.0f, 1.0f, 1234ull, gen_type[i]);
      }
      std::cout << "Test normal_float" << std::endl;
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
      {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
        omp_get_rng_normal_double(data, sz, 0.0, 10.0, 1234ull, gen_type[i]);
      }
      std::cout << "Test normal_double" << std::endl;
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
  }
  
  //Test omp_target_alloc
  {	
    size_t sz = 1024 * 1024 * 128;
    MYMALLOC(double, sz);
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data map(tofrom:data[0:sz])
#endif
    {
#if defined(ARCH_CUDA) || defined(ARCH_HIP) || defined(ARCH_INTEL)
  #pragma omp target data use_device_ptr(data)
#endif
      omp_get_rng_normal_double(data, sz, 0.0, 10.0, 1234ull, gen_type[0]);
    }

    int device_id = omp_get_default_device();
    int host_id = omp_get_initial_device();
    double* data_h = (double*)malloc(sizeof(double) * (sz));
    double* data_d = (double*)omp_target_alloc(sizeof(double) * sz, device_id);
    {
      omp_get_rng_normal_double(data_d, sz, 0.0, 10.0, 1234ull, gen_type[0]);
    }
    omp_target_memcpy(data_h, data_d, sizeof(double) * sz, 0, 0, host_id, device_id);
    std::cout << "Test normal_double omp_target_alloc" << std::endl;
    for(int i=0; i<sz; i+=sz/output_interval)
      std::cout << data[i] << "\t\t\t" << data_h[i] << std::endl;
  }
}
