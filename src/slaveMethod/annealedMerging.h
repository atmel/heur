#ifndef __HEURISTICS_ANNEALED_MERGE__
#define __HEURISTICS_ANNEALED_MERGE__

#include "heuristics.h"

#if USE_CUDA

template<class popContainer>
__global__ void annealedMergeKernel(popContainer pop, curandState *state, range sourceRng, range destRng, float T){
	int id = threadIdx.x;
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	curandState localState;
	localState = state[stableId]; //load generator state

	//proces candidates
	for(int i=0; i < REQUIRED_RUNS(destRng.length); i++, id += blockDim.x){
		if(id >= destRng.length){ //complete
			state[stableId]=localState; //save state
			__syncthreads();
			return;
		}

		//delta = new - old ... wRange is dest range (=old)
		if(cool::accept(T, std::max(0,pop.RangeFitness(blockIdx.x,i+sourceRng.lo) - pop.RangeFitness(blockIdx.x,i+destRng.lo)), localState)){
			for(int j=0; j < pop.GetDim(); j++){
				//dest = source
				pop.RangeComponent(blockIdx.x,i+destRng.lo,j) = pop.RangeComponent(blockIdx.x,i+sourceRng.lo,j);
			}
		}
		__syncthreads();
	}
	//save state
	state[stableId] = localState;
}

#endif

/*
	Performs annealed merging -- maybe add some constatns to probablitites, or let T0 be different than other component's T0
*/

template<class popContainer, class cool>
class annealedMerging: public slaveMethod<popContainer>, public stochasticMethod{
	int threadCount;
	const float T0;
	int gen;
public:
	annealedMerging<popContainer,cool>(float Temperature0):T0(Temperature0), gen(1){};
	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("annealedMerging Method: slave init unsuccessfull")
		threadCount = std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length);
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("annealedMerging Method: stochastic method init unsuccessfull")
		return 1;
	}

	int Perform(){
		float T = cool::cool(T0,gen);
	#if USE_CUDA
		CUDA_CALL("annealedMerging kernel",( annealedMergeKernel<popContainer> <<<this->pop->GetPopsPerKernel(), threadCount>>>
			(*(this->pop),this->devStates,this->fullRange, this->workingRange,T)));
	#else
		int wlo = this->workingRange.lo, flo = this->fullRange.lo;

		for(int i = 0; i++ < this->workingRange.length; i++){
			//delta = new - old ... wRange is dest range (=old)
			if(cool::accept(T, std::max(0,this->pop->RangeFitness(i+flo) - this->pop->RangeFitness(i+wlo)){
				for(int j=0; j < this->pop->GetDim(); j++){
					//dest = source
					this->pop->RangeComponent(i+wlo,j) = this->pop->RangeComponent(i+flo,j);
				}
			}
		}
	#endif
		//inc gen
		gen++;
	return 1;
	}

};

#endif
