#ifndef __HEURISTICS_ROUNDING__
#define __HEURISTICS_ROUNDING__

#include "heuristics.h"
#include <math.h>

template<typename vectorType>
class probabilisticRounding{
public:
	#if USE_CUDA
	static __device__ vectorType round(double x, curandState* st){
		double t = floor(x);
		return (curand_uniform(st) > (x-t)) ? t+1:t;
	}
	static __device__ vectorType round(float x, curandState* st){
		float t = floor(x);
		return (curand_uniform(st) > (x-t)) ? t+1:t;
	}
	#else
	static inline vectorType round(double x){
		double t = floor(x);
		return (rand() > (vectorType)(RAND_MAX*(x-t))) ? t+1:t;
	}
	static inline vectorType round(float x){
		float t = floor(x);
		return (rand() > (vectorType)(RAND_MAX*(x-t))) ? t+1:t;
	}
	#endif
};

#endif