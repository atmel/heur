#ifndef __HEURISTICS_MOVE__
#define __HEURISTICS_MOVE__

#include "heuristics.h"

#if USE_CUDA

/*
	Move candidates using shared memory and one thread per candidate
	blockDim.x is assumed = rng.length
*/
template<class popContainer, typename vectorType, typename evalType>
__global__ void MoveKernelCandidatePerThread(popContainer pop, range rng, int *gInd){
	extern __shared__ vectorType dynamic[];

	int id = threadIdx.x;
	//load to shared
	for(int i=0; i<pop.GetDim();i++){
		dynamic[i*blockDim.x + id] = pop.RangeComponent(blockIdx.x, gInd[id], i);
	}
	evalType fitness = pop.RangeFitness(blockIdx.x, gInd[id]);
	__syncthreads();

	//copy to destination
	for(int i=0; i<pop.GetDim();i++){
		pop.RangeComponent(blockIdx.x, rng.lo + id, i) = dynamic[i*blockDim.x + id];
	}
	pop.RangeFitness(blockIdx.x, rng.lo + id) = fitness;
}

/*
	Move candidates using shared memory and one thread per component
	blockDim.x is assumed = rng.length*pop.GetDim()
*/
template<class popContainer, typename vectorType, typename evalType>
__global__ void MoveKernelComponentPerThread(popContainer pop, range rng, int *gInd){
	extern __shared__ vectorType dynamic[];

	int cand = threadIdx.x/pop.GetDim();
	int comp = threadIdx.x % pop.GetDim();
	//load to shared
	dynamic[comp*blockDim.x + cand] = pop.RangeComponent(blockIdx.x, gInd[cand], comp);
	__syncthreads();

	//copy to destination
	pop.RangeComponent(blockIdx.x, rng.lo + cand, comp) = dynamic[comp*blockDim.x + cand];
	__syncthreads();

	//copy fitness:
	if(threadIdx.x >= rng.length) return;
	evalType fit;
	fit = pop.RangeFitness(blockIdx.x, gInd[threadIdx.x]);
	__syncthreads();
	pop.RangeFitness(blockIdx.x, rng.lo + threadIdx.x) = fit;
}

#endif


/*
	Moves workingRange.length candidates from positions specified by *indices to places 
	specified by working range
*/
template<class popContainer, typename vectorType, typename evalType>
class moveToRange : public slaveMethod<popContainer>{
	//retrieved from master (for GPU result)
	int *indices;

public:

	int Init(generalInfoProvider *p){
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("moveToRangeMethod: slave init unsuccessfull")
		sortResourceProvider *srp = dynamic_cast<sortResourceProvider*>(p);
		if(srp == NULL) EXIT0("moveToRangeMethod: resource provider cast unsuccessfull")
		indices = srp->GetIndexArray();
		return 1;
	}

	int Perform(){
	#if USE_CUDA
		int memSize = ALLIGN_64(this->workingRange.length * this->pop->GetDim() * sizeof(vectorType));

		if(this->workingRange.length * this->pop->GetDim() > MAX_THREADS_PER_BLOCK){
			int threads = this->workingRange.length; 
			CUDA_CALL("Candidate move kernel", (MoveKernelCandidatePerThread<popContainer, vectorType, evalType>
				<<<this->pop->GetPopsPerKernel(), threads, memSize>>>
				(*(this->pop), this->workingRange, indices)))
		}else{
			int threads = this->workingRange.length * this->pop->GetDim(); 
			CUDA_CALL("Component move kernel", (MoveKernelComponentPerThread<popContainer, vectorType, evalType>
				<<<this->pop->GetPopsPerKernel(), threads, memSize>>>
				(*(this->pop), this->workingRange, indices)))
		}
	#else

		//odstranit cykly, presunout


	#endif
		return 1;
	}
};

#endif
