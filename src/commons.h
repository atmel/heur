#ifndef __HEURISTICS_COMMONS__
#define __HEURISTICS_COMMONS__

//debug version!!
#define USE_CUDA 1
#define MAX_THREADS_PER_BLOCK 512

#if !USE_CUDA
//#define __device__
//#define __host__
#endif

#if USE_CUDA

//to compute number of required repetitions within kernel to process all data in pop/offspr/mate...
#define REQUIRED_RUNS(SET_SIZE) ((((SET_SIZE)-1)/blockDim.x)+1)

#define ALLIGN_64(SIZE) (((SIZE)/8 + 1)*8)

#else
#endif

#define H_DEBUG 1

//debug print
#if H_DEBUG
  #include <iostream>
  #include <cstdio>
  #define D(...) {char s[400]; sprintf(s,__VA_ARGS__); std::cout << "DGB info: " << s << "\n";}
  #define EXIT0(...) {D(__VA_ARGS__) return 0;}
  #define CUDA_CALL(NAME,CALL) {std::cout <<"DGB info: KERNEL: "<< NAME << "\n"; CALL; std::cout<< cudaGetErrorString(cudaGetLastError()) << "\n";}
  #define CUDA_CHECK(NAME) {std::cout<< "DGB info: CUDA ERR CHECK: "<< NAME << ":-- "<< cudaGetErrorString(cudaGetLastError()) << "\n";}
#else
  #define D(...) //debug info
  #define EXIT0(...) {return 0;}
  #define CUDA_CALL(NAME,CALL) {CALL;}
  #define CUDA_CHECK(NAME) //nothing
#endif

#endif