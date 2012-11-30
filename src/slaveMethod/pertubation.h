#ifndef __HEURISTICS_PERTUBATION__
#define __HEURISTICS_PERTUBATION__

#include "heuristics.h"
#include<time.h>
#include<cstdlib>
#include<algorithm>

#if USE_CUDA
template<class popContainer>
__global__ void PeriodicPertubationKernel(popContainer pop, range rng){
	int id = threadIdx.x + rng.lo;

	for(int i=0; i < REQUIRED_RUNS(rng.length); i++, id += blockDim.x){
		if(id >= rng.hi) return; //complete
		for(int j=0; j< pop.GetDim(); j++){
			//similar two phases as on cpu, just without if
			/* we want x within [a,b), so
				x = (x-a)%(b-a)  // in (-b+a,b-a) due to modulo behaviour
				x = (x+b-a)%(b-a) + a
			*/
			pop.RangeComponent(blockIdx.x,id,j) = 
				(pop.RangeComponent(blockIdx.x,id,j) - pop.GetLowerLimit(j))%
				(pop.GetUpperLimit(j) - pop.GetLowerLimit(j));
			pop.RangeComponent(blockIdx.x,id,j) = 
				(pop.RangeComponent(blockIdx.x,id,j) + pop.GetUpperLimit(j) - pop.GetLowerLimit(j))%
				(pop.GetUpperLimit(j) - pop.GetLowerLimit(j)) + pop.GetLowerLimit(j);
		}
	}
}
#endif


/* Performs periodic pertubation
	To be range safe, limits should be few times smaller, than whole type range!
	This metod expects vectorType to be integer (modulo)
*/
template<class popContainer>
class periodicPertubation : public slaveMethod<popContainer>{

	public:
	int Perform(){
#if USE_CUDA
		D("Running kernel in %d blocks, %d threads.",this->pop->GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length))
		CUDA_CALL("pertubation",(PeriodicPertubationKernel<popContainer>
				<<<this->pop->GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length)>>>(*(this->pop),this->workingRange)))
#else
		for(int i=this->workingRange.lo; i< this->workingRange.hi; i++){
			for(int j=0; j< this->pop->GetDim(); j++){
				//modulo works with sign so this can be still out of range
				this->pop->RangeComponent(i,j) = 
					(this->pop->RangeComponent(i,j) - this->pop->GetLowerLimit(j))%
					(this->pop->GetUpperLimit(j) - this->pop->GetLowerLimit(j));
				//correct if negative -- after substraction of lower limit modulo should stay positive
				if(this->pop->RangeComponent(i,j) < 0){
					this->pop->RangeComponent(i,j) +=
						(this->pop->GetUpperLimit(j) - this->pop->GetLowerLimit(j));
				}
				this->pop->RangeComponent(i,j) += this->pop->GetLowerLimit(j);
			}
		}
#endif
		return 1;
	}
};


#endif