#ifndef __VYZKUM_SELECTION__
#define __VYZKUM_SELECTION__

#include "heuristics.h"
#include<time.h>
#include<stdlib.h>
#include<algorithm>

//-----------------------------------------------------------------------------------------
/*
	Fills mating pool with indices from range
*/
#if USE_CUDA
template<class popContainer>
__global__ void TwoTournamentSelection(popContainer pop, curandState *state, range rng, int *mate, int mateSize){
	int id = threadIdx.x;
	const int stableId = threadIdx.x + blockIdx.x*blockDim.x; //for accessing curand states
	curandState localState = state[stableId]; //load generator state
	int one, two;

	for(int i=0; i < REQUIRED_RUNS(mateSize); i++, id += blockDim.x){
		if(id >= mateSize){  //threads that are over save state here
			state[stableId]=localState;
			return;
		}
		//pick tournament participants
		one = curand(&localState) % rng.length + rng.lo;
		two = curand(&localState) % rng.length + rng.lo;
		//access right block of mate!
		//__syncthreads(); maybe
		//winner has lower fitness
		mate[id + blockIdx.x*mateSize] = (pop.RangeFitness(blockIdx.x,one) < pop.RangeFitness(blockIdx.x,two))?one:two;
	}
	//threads from the last run save state here
	state[stableId]=localState;
}
#endif

/*===================================================================================================================================
	performs selection -- fills mate with indices from working range
*/
template<class popContainer>
class twoTournamentSelection : public slaveMethod<popContainer>, public stochasticMethod{
	//retrieved from master
	int *mate;
	int mateSize, threadCount;

	public:
	int Init(generalInfoProvider *p){
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("tournament slection method: slave init unsuccessfull")
		mateProvider *mp = dynamic_cast<mateProvider*>(p);
		if(mp == NULL) EXIT0("tournament slection method: mate provider cast unsuccessfull")
		mate = mp->GetMatingPool();
		mateSize = mp->GetMatingPoolSize();
		//init stochasticMethod
		threadCount = std::min(MAX_THREADS_PER_BLOCK,mateSize);
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("tournament slection method: stochastic method init unsuccessfull")
		return 1;
	}
	int Perform(){			//basic 2-tournament, pop must be smaller than randmax!!
		#if USE_CUDA
			CUDA_CALL("tournament selection",(TwoTournamentSelection<popContainer><<<this->pop->GetPopsPerKernel() , threadCount>>>
				(*(this->pop),this->devStates,this->workingRange, mate, mateSize)));
		#else		
			int one,two;
			for(int i=0;i<mateSize;i++){
				one = rand() % this->workingRange.length + this->workingRange.lo;
				two = rand() % this->workingRange.length + this->workingRange.lo;
				mate[i] = (this->pop->RangeFitness(one) < this->pop->.RangeFitness(two))?one:two;
			}
		#endif
		return 1;
	}
};

#endif