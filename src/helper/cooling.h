#ifndef __HEURISTICS_COOLING__
#define __HEURISTICS_COOLING__

#include "heuristics.h"

class classicCooling{
public:
	#if USE_CUDA
	static __device__ bool accept(float T, float delta, curandState *st){
		return curand_uniform(st) < exp(-delta/T);
	}
	#else
	static bool accept(float T, float delta){
		return (float)((int)(rand() & 0x7FFF) + 1)/(0x7FFF + 1) < exp(-delta/T);
	}
	#endif
	
	static float cool(float T0, int step){
		return T0/log(step);
	}
	
};

#endif