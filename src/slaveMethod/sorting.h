#ifndef __HEURISTICS_SORTING__
#define __HEURISTICS_SORTING__

#include "heuristics.h"
#include <algorithm>
#include <iostream>
/*bitonic sorter*/

#define MOD2(X,M) ((X) & ((M)-1))	//low bits, M is power of 2
#define DIV22(X,D) (((X) & (~((D)-1)))*2) // shifted high bits, D is power of 2
//compare and swap items in fit, ind with these indexes
#define CMP_SWAP(X,Y) {\
	evalType ftmp; int itmp;\
	if(fit[(X)] > fit[(Y)]){\
		ftmp = fit[(X)]; fit[(X)] = fit[(Y)]; fit[(Y)] = ftmp;\
		itmp = ind[(X)]; ind[(X)] = ind[(Y)]; ind[(Y)] = itmp;\
	}}

#if USE_CUDA
/* performs full bitonic sort
	Number of sorted elements = 2x blockDim.x (number of threads).
	No checks, minimalistic code
	Box representation is taken from wiki

	rngLo is basically offset of indices (whole range is not needed as we sort fixed amount of elements)
	gInd is array of first <resLength> indices (part that the we originally wanted to be sorted)
*/
template<class popContainer, typename evalType>
__global__ void FullBitonicSortKernel(popContainer pop, int rngLo, int *gInd, int resLength, evalType *gFit){
	//setup shared
	extern __shared__ int dynamic[];
	const int allignedFitSize = ALLIGN_64((blockDim.x*2)*sizeof(evalType)); //allign to 8 bytes

	evalType *fit = reinterpret_cast<evalType*>(dynamic);
	int *ind = reinterpret_cast<int*> (&dynamic[allignedFitSize/sizeof(int)]); // /sizeof(int) -- 'dynamic' is int

	int id = threadIdx.x;
	//load to shared
	#pragma unroll 2
	for(int i=0; i<2; i++, id += blockDim.x){
		fit[id] = pop.RangeFitness(blockIdx.x,id + rngLo);
		ind[id] = id + rngLo;
	}

	id = threadIdx.x;
	//soooooort
	int m,d;
	for(int p=1; p <= blockDim.x; p*=2){
		//brown box
		//compute ids to swap between
		m = MOD2(id,p);
		d = DIV22(id,p);
		CMP_SWAP(d+m, d + p*2 - m - 1); //magic!
		__syncthreads();
		for(int q=p/2; q > 0; q/=2){
			//pink box
			m = MOD2(id,q);
			d = DIV22(id,q);
			CMP_SWAP(d+m, d+m + q); //magic!
			__syncthreads();
		}
	}

	//copy sorted index to global
	//id = threadIdx; already is
	#pragma unroll 2
	//max 2 times (do not use REQUIRED_RUNS)
	for(int i=0; i<2; i++, id += blockDim.x){
		if(id < resLength){  //complete - copy only working range
			gInd[id + blockIdx.x*resLength] = ind[id];
			gFit[id + blockIdx.x*resLength] = fit[id];
		}
	}
}
#endif

//===========================================================================================================
/* partially sorts full range, working range is sorted fully.
	On GPU, all the full range is fully sorted
	sorted index (fitness and indices) is saved to RAM/GPURAM
*/
template<class popContainer, typename evalType>
class rangedSorting : public slaveMethod<popContainer>{
	//retrieved from master (for GPU result)
	int *indices;
	//DEBUG:
	//evalType *fitnesses;

	//for local sorting
	indexElement<evalType> *els;
public:
	//dummy constructor for safe memory deletion in destructor
	rangedSorting<popContainer,evalType>() : slaveMethod<popContainer>(), els(0){};

	int Init(generalInfoProvider *p){
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("RangedSorting method: slave init unsuccessfull")
		sortResourceProvider *srp = dynamic_cast<sortResourceProvider*>(p);
		if(srp == NULL) EXIT0("RangedSorting method: resource provider cast unsuccessfull")
		//els = srp->GetIndexStorage();
		indices = srp->GetIndexArray();
		
		#if USE_CUDA
			//DEBUG:
			//D("Sorting: length of indices: %d",this->pop->GetPopsPerKernel()*this->workingRange.length );
			//cudaMalloc(&fitnesses,this->pop->GetPopsPerKernel()*sizeof(evalType)*(this->workingRange.length));
			//cudaMemset(fitnesses,0,this->pop->GetPopsPerKernel()*sizeof(evalType)*(this->workingRange.length));
		#else
			//allocate internal array
			els = new indexElement<evalType>[this->fullRange.length];
		#endif
		return 1;
	}

	//dealocate array
	~rangedSorting<popContainer,evalType>(){
		//DEBUG
		//cudaFree(fitnesses);

		#if !USE_CUDA
			if(els != 0) delete [] els;
		#endif
	}

	int Perform(){
	#if USE_CUDA
		int threads = this->fullRange.length/2;
		//D("size of indexElement<evalType> is: %d",(int)sizeof(indexElement<evalType>))
		CUDA_CALL("Sorting Kernel",(FullBitonicSortKernel<popContainer, evalType>
			<<<this->pop->GetPopsPerKernel(),threads, ALLIGN_64((threads*2)*sizeof(evalType))+ALLIGN_64((threads*2)*sizeof(int))>>>
			(*(this->pop), this->workingRange.lo, indices, this->workingRange.length, fitnesses)))

		//DEBUG:
		//D("Error?: %s",cudaGetErrorString(cudaGetLastError()))
		/*D("Sorting: printing fitnesses")
		evalType *fitToPrint = new evalType[this->pop->GetPopsPerKernel()*this->workingRange.length];
		CUDA_CALL("Sorting Memcpy",(cudaMemcpy(fitToPrint, fitnesses, this->pop->GetPopsPerKernel()*this->workingRange.length * sizeof(evalType), 
		  cudaMemcpyDeviceToHost)))
		
		for(int i = 0; i< this->workingRange.length*this->pop->GetPopsPerKernel();i++){
			std::cout << fitToPrint[i] << ", ";
		}*/
	#else
		//fill structure
		for(int i = 0; i < this->fullRange.length; i++){
			els[i].fit = this->pop->RangeFitness(i + this->fullRange.lo);
			els[i].ind = i + this->fullRange.lo;
		}
		//partially sort
		std::partial_sort(els, els + this->workingRange.length, els + this->fullRange.length);

		//save for later use
		for(int i=0; i<this->workingRange.length; i++){
			indices[i] = els[i].ind;
		}
	#endif
	return 1;
	}
};

#endif
