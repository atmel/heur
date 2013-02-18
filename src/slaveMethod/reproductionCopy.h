#ifndef __HEURISTICS_COPY_REPRO__
#define __HEURISTICS_COPY_REPRO__

#include "heuristics.h"

#if USE_CUDA

template<class popContainer>
__global__ void copyReproductionKernel(popContainer pop, range sourceRng, range destRng){
	int id = threadIdx.x;

	//proces candidates
	for(int i=0; i < REQUIRED_RUNS(destRng.length); i++, id += blockDim.x){
		if(id >= destRng.length){ //complete
			__syncthreads();
			return;
		}
		for(int j=0; j < pop.GetDim(); j++){
			//dest = source
			pop.RangeComponent(blockIdx.x,i+destRng.lo,j) = pop.RangeComponent(blockIdx.x,i+sourceRng.lo,j);
		}
		__syncthreads();
	}
}

#endif

/*
	Performs annealed merging -- maybe add some constatns to probablitites, or let T0 be different than other component's T0
*/

template<class popContainer>
class copyReproduction: public slaveMethod<popContainer>{
	int threadCount;

public:
	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("copyReproduction Method: slave init unsuccessfull")
		threadCount = std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length);
		return 1;
	}

	int Perform(){
	D("performing reproduction copy from: %d -- %d  to %d -- %d", this->fullRange.lo,this->fullRange.hi, this->workingRange.lo,this->workingRange.hi)
	#if USE_CUDA
		CUDA_CALL("copyReproduction kernel",( copyReproductionKernel<popContainer> <<<this->pop->GetPopsPerKernel(), threadCount>>>
			(*(this->pop),this->fullRange, this->workingRange)));
	#else
		int wlo = this->workingRange.lo, flo = this->fullRange.lo;

		for(int i = 0; i++ < this->workingRange.length; i++){
			for(int j=0; j < this->pop->GetDim(); j++){
				//dest = source
				this->pop->RangeComponent(i+wlo,j) = this->pop->RangeComponent(i+flo,j);
			}
		}
	#endif
	return 1;
	}

};

#endif
