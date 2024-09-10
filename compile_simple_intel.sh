mpicxx -O3 -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc" -lmkl_rt -DARCH_INTEL simple_use.C
#echo 'Before running the program, do export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-4.5.2/lib/'
