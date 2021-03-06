#ifndef __HEURISTICS_RANDOM__
#define __HEURISTICS_RANDOM__

#include "heuristics.h"
#include<cuda.h>
#include<curand_kernel.h>
#include <iostream>

#if USE_CUDA

/* This kernel is expected to run with same number of threads and blocks as the kernel which will use 
	random generation. Size of array is determined in the same fashion.
*/
__global__ void cudaRandomSetupKernel(curandState *state, int seed){
	int id = threadIdx .x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed , a different sequence
		number , no offset */
	curand_init (seed + id , 0 , 0, state + id);
}

//wrapper that can be called from other files :/ plus allocation
inline void cudaRandomAllocateAndSetup(curandState **state, int seed, int threads, int blocks){
	cudaMalloc(state, blocks*threads*sizeof(curandState));
	CUDA_CALL("Random generator init",(cudaRandomSetupKernel<<<blocks,threads>>>(*state,seed)))
}


//inline __device__ int2 curand_normal2int(curandState *state, int sigma){
//	float2 tmp = curand_normal2(state);
//	return make_int2(rintf(tmp.x*sigma),rintf(tmp.y*sigma));
//}

inline __device__ int curand_cauchyInt(curandState *state, int scale){
#define PI 3.1415926535897932f
	return rintf(tanf(curand_uniform(state)*2*PI));
#undef PI
}
#endif //cuda

//CPU equivalent functions -----------------------------------------------------------
#include<stdlib.h>
#include<cmath>

//in its namespace heuristics rand
namespace hrand{

typedef struct _int2 int2;
struct _int2{
	int x,y;

	_int2(const int _x, const int _y):x(_x),y(_y){}
	inline _int2 &operator=(const int2 &rval){
		x=rval.x;
		y=rval.y;
		return *this;
	}
};

typedef struct _float2 float2;
struct _float2{
	float x,y;

	_float2(const float _x, const float _y):x(_x),y(_y){}
	_float2():x(0),y(0){};
	inline _float2 &operator=(const float2 &rval){
		x=rval.x;
		y=rval.y;
		return *this;
	}
};

//this expects srand(..) was called before, performs standard Box-Muller transform
inline float2 rand_normal2(){
#define PI 3.1415926535897932f
	//get x from (0,1] independent on implementation (consider RAND_MAX = 32767 = 0x7FFF)
	float x = (float)((int)(rand() & 0x7FFF) + 1)/(0x7FFF + 1); // i.e. from (0,1]
	//std::cout << x << ", ";
	float R = sqrt(-2*log(x));
	float theta = 2*PI*(float)(rand() & 0x7FFF)/(0x7FFF + 1); // i.e. from [0,2pi)
	//std::cout << sin(theta) << ", " << cos(theta) << "; ";
	return float2(R*sin(theta),R*cos(theta));
#undef PI
}

//on GPU or CPU -- this is often routine
void SetupRandomGeneration(curandState_t **state, int seed, int threads, int blocks){
#if USE_CUDA
//init random generators on GPU -- allocate space for states
	cudaRandomAllocateAndSetup(state,seed,threads,blocks);
#else
	srand(seed);
#endif
}

#if USE_CUDA
//deallocate
void cudaRandomFinalize(curandState_t *state){
	cudaFree(state);
}
#endif


inline int rand_cauchyInt(int scale){
#define PI 3.1415926535897932f
	float theta = 2*PI*(rand() & 0x7FFF)/(0x7FFF + 1); // i.e. from [0,2pi)
	return (int) (tan(theta)*scale);
#undef PI
}

} //hrand namespace

#endif