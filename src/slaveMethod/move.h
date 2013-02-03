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
	int stride = blockDim.x/pop.GetDim();
	//load to shared
	dynamic[comp*stride + cand] = pop.RangeComponent(blockIdx.x, gInd[cand], comp);
	__syncthreads();

	//copy to destination
	pop.RangeComponent(blockIdx.x, rng.lo + cand, comp) = dynamic[comp*stride + cand];
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

		/*reshuffle index so destination of each move is not needed at time the move is performed:
		  i.e. seqence 3,1,10 is bad, because when we move third element onto the fitst place, we
		  lose the first element needed next step.
		  
		  Imagine the edges runing from element's desired position to actual position. Our goal is 
		  to remove all edges running backwards. This is done by swapping the elements.
		*/
		int tmp;
		#define SWAP(X,Y){tmp = X; X=Y; Y=tmp;}
		int c = this->workingRange.lo;
		for(int i=0; i<this->workingRange.length; i++){
			while((c <= indices[i]) && (indices[i] < c+i)){ 
				SWAP(indices[indices[i]-c] , indices[i]); // order matters here
			}
		}
		#undef SWAP
		//now move -- partial sorting is preserved
		for(int i=0; i<this->workingRange.length; i++){
			for(int j=0; j<this->pop->GetDim(); j++){
				this->pop->RangeComponent(i+c,j) = this->pop->RangeComponent(indices[i],j);
			}
			this->pop->RangeFitness(i+c) = this->pop->RangeFitness(indices[i]);
		}
	#endif
		return 1;
	}
};

#endif
