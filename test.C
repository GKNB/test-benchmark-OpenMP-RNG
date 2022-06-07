#include<iostream>
#include<cstdlib>
#include<vector>

#include "openmp_rng.h"

#define MYMALLOC(T, SZ) T* data = (T*)malloc(sizeof(T) * (SZ));

int main()
{
  int output_interval = 10;
  std::vector<generator_enum> gen_type = {generator_enum::philox, generator_enum::xorwow, generator_enum::mrg32k3a, generator_enum::sobol32, generator_enum::sobol64, generator_enum::mtgp32, generator_enum::mt19937};
  for(int i=0; i<gen_type.size(); i++)
  {
    if(gen_type[i] != generator_enum::sobol64)
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(unsigned int, sz);
  
  #pragma omp target data map(tofrom:data[0:sz])
      {
        omp_get_rng_uniform_uint(data, sz, 1234ull, gen_type[i]);
      }
    
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(float, sz);
  
  #pragma omp target data map(tofrom:data[0:sz])
      {
        omp_get_rng_uniform_float(data, sz, 1234ull, gen_type[i]);
      }
    
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(double, sz);
  
  #pragma omp target data map(tofrom:data[0:sz])
      {
        omp_get_rng_uniform_double(data, sz, 1234ull, gen_type[i]);
      }
    
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(float, sz);
  
  #pragma omp target data map(tofrom:data[0:sz])
      {
        omp_get_rng_normal_float(data, sz, 1.0f, 1.0f, 1234ull, gen_type[i]);
      }
    
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
    {	
      size_t sz = 1024 * 1024 * 128;
      MYMALLOC(double, sz);
  
  #pragma omp target data map(tofrom:data[0:sz])
      {
        omp_get_rng_normal_double(data, sz, 0.0, 10.0, 1234ull, gen_type[i]);
      }
    
      for(int i=0; i<sz; i+=sz/output_interval)
        std::cout << data[i] << std::endl;
    }
  }

}