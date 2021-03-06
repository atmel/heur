#ifndef __HEURISTICS_INITIALIZATION__
#define __HEURISTICS_INITIALIZATION__

#include "heuristics.h"
#include<time.h>
#include<cstdlib>
#include<algorithm>


#if USE_CUDA
template<class popContainer>
__global__ void PseudouniformRandomInitializationKernel(popContainer pop, curandState *state, range rng){
	int id = threadIdx.x + rng.lo;
	const int stableId = threadIdx.x + blockIdx.x*blockDim.x; //for accessing curand states
	curandState localState = state[stableId]; //load generator state

	for(int i=0; i < REQUIRED_RUNS(rng.length); i++, id += blockDim.x){
		if(id >= rng.hi){  //threads that are over save state here
			state[stableId]=localState;
			return;
		}
		for(int j=0; j< pop.GetDim(); j++){
			//here uniform is used so insufficient RAND_MAX is solved internally
			//constant is substracted because curand_uniform returns values from [0,1] ... check that!! and we want [0,1)
			pop.RangeComponent(blockIdx.x, id,j) = floor(curand_uniform(&localState)*(pop.GetUpperLimit(j) - pop.GetLowerLimit(j) - 0.0001)) 
				+ pop.GetLowerLimit(j);
		}
	}
	//threads from the last run save state here
	state[stableId]=localState;
}
#endif

/* Performs random nearly uniform initialization of candidates within limits [lower,upper)
	!!!
	This expercts RAND_MAX to be large enough (at least upper-lower)
	!!!
*/
template<class popContainer>
class pseudouniformRandomInitialization : public slaveMethod<popContainer>, public stochasticMethod{
	int threadCount;

	public:
	//we must inicialize number of seeds in stochastic method
	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("randomInitMethod: slave init unsuccessfull")
		threadCount = std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length);
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("randomInitMethod: stochastic method init unsuccessfull")
		return 1;
	}
	
	int Perform(){
	  D("performing ranged random init in range: %d, %d", this->workingRange.lo,this->workingRange.hi)
#if USE_CUDA
		CUDA_CHECK("Before Kernel run")
		D("Running kernel in %d blocks, %d threads. Required runs: %d",this->pop->GetPopsPerKernel(),threadCount,((100-1)/100)+1)
		CUDA_CALL("pseudorandom init",(PseudouniformRandomInitializationKernel<popContainer>
		  <<<this->pop->GetPopsPerKernel(),threadCount>>>(*(this->pop),this->devStates,this->workingRange)))
#else	
		D("starting for cycle")
		for(int i= this->workingRange.lo; i < this->workingRange.hi; i++){
			for(int j=0; j< this->pop->GetDim(); j++){
				this->pop->RangeComponent(i,j) = ((float)rand()/RAND_MAX)*(this->pop->GetUpperLimit(j) - this->pop->GetLowerLimit(j))
					+ this->pop->GetLowerLimit(j);
			}
		}
		
		D("exiting random initialization")
#endif
		return 1;
	}
};

#endif