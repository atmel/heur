#ifndef __VYZKUM_CROSSOVER__
#define __VYZKUM_CROSSOVER__

#include "heuristics.h"
#include<time.h>
#include<stdlib.h>
#include<algorithm>

//-----------------------------------------------------------------------------------------
/*
	Performs one point crossover, crossover point is chosen randomly, copy the rest
	Expects pop.GetDim be at least 2
*/

#if USE_CUDA
template<class popContainer>
__global__ void OnePointCrossoverKernel(popContainer pop, curandState *state, range crossRng, range fullRng, 
																								int *mate, int mateSize){
	int id = threadIdx.x;
	
	const int stableId = threadIdx.x + blockIdx.x*blockDim.x; //for accessing curand states
	curandState localState = state[stableId]; //load generator state

	for(int i=0; i < REQUIRED_RUNS(crossRng.length); i++, id += blockDim.x){
		if(id >= crossRng.length){  //threads that are over goto the next stage
			break;
		}
		//crossover
		int j, point = floor(curand_uniform(&localState)*(pop.GetDim()-2))+1;	//not all components from one candidate
		for(j=0; j<point; j++){
			pop.RangeComponent(blockIdx.x, id+crossRng.lo, j) = 
				pop.RangeComponent(blockIdx.x, mate[2*id + blockIdx.x*mateSize], j);
		}
		for(/*j is already==point*/; j < pop.GetDim(); j++){
			pop.RangeComponent(blockIdx.x, id+crossRng.lo, j) = 
				pop.RangeComponent(blockIdx.x, mate[2*id+1 + blockIdx.x*mateSize], j);
		}
	}
	__syncthreads();
	state[stableId]=localState;
	
	//copy part
	id = threadIdx.x;
	for(int i=0; i < REQUIRED_RUNS(fullRng.length - crossRng.length); i++, id += blockDim.x){
		if(id >= fullRng.length - crossRng.length){  //threads that are over, exit
			return;
		}
		for(int j=0; j < pop.GetDim(); j++){
			pop.RangeComponent(blockIdx.x, id + crossRng.hi, j) = 
				pop.RangeComponent(blockIdx.x, mate[id + crossRng.length*2 + blockIdx.x*mateSize], j);
		}
	}
	
}
#endif

/*
	performs 2-parent one-point crossover. 
	Probability of crossover is determined by retrieved ranges:
	We treat mating pool as:
	2*|working range|, |full range - working range|
	From the firs part we make offspring by crossover, 
	form the second by pure copy from population by mating pool indices
 */
template<class popContainer>
class onePointCrossover : public slaveMethod<popContainer>, public stochasticMethod, public arityProvider{
	//retrieved from master
	int *mate;
	int mateSize, threadCount;

	public:
	int Init(generalInfoProvider *p){
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("onePointCrossover method: slave init unsuccessfull")
		mateProvider *mp = dynamic_cast<mateProvider*>(p);
		if(mp == NULL) EXIT0("onePointCrossover method: mate provider cast unsuccessfull")
		mate = mp->GetMatingPool();
		mateSize = mp->GetMatingPoolSize();
		//init stochasticMethod -- we need only so many random gens as the count of offsrings creted by crossover
		threadCount = std::min(MAX_THREADS_PER_BLOCK,std::max(this->workingRange.length,this->fullRange.length - this->workingRange.length));
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("onePointCrossover method: stochastic method init unsuccessfull")
		return 1;
	}

	//override virtual
	int GetArity(){return 2;}

	int Perform(){	
		#if USE_CUDA
			CUDA_CALL("onepoint crossover kernel",(OnePointCrossoverKernel<popContainer><<<this->pop->GetPopsPerKernel(),threadCount>>>
				(*(this->pop), this->devStates, this->workingRange, this->fullRange, mate, mateSize)));
		#else		
			int i,j,mIdx;
			//crossover part
			for(i=this->workingRange.lo, mIdx = 0; i<this->workingRange.hi; i++, mIdx +=2){
				int point = rand() % (this->pop->GetDim()-1) + 1;	//not all components from one candidate
				for(j=0; j<point; j++){
					this->pop->RangeComponent(i,j) = this->pop->RangeComponent(mate[mIdx],j);
				}
				for(/*j is already==point*/; j < this->pop->GetDim(); j++){
					this->pop->RangeComponent(i,j) = this->pop->RangeComponent(mate[mIdx+1],j);
				}
			}
			//from work.hi to full.hi just copy. For mate indexing use mIdx which points to the beginning of "copy part" 
			for(; i<this->fullRange.hi; i++, mIdx++){
				for(j=0; j < this->pop->GetDim(); j++){
					this->pop->RangeComponent(i,j) = this->pop->RangeComponent(mate[mIdx],j);
				}
			}
		#endif
		return 1;
	}
};

#endif