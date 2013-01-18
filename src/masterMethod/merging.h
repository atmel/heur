#ifndef __VYZKUM_MERGING__
#define __VYZKUM_MERGING__

#include "heuristics.h"


/*
	This method performs merging by replacement:
	old population is replaced with |P| best candidates from offspring
*/
template<class popContainer, typename vectorType, typename evalType>
class replaceMerging : public masterMethod<popContainer>, public sortResourceProvider{
	
	// created on CPU or GPU
	int *indices;
	range wRange, fRange;

public:

	replaceMerging<popContainer, vectorType, evalType>(){
		this->Add(new rangedSorting<popContainer,evalType>())
			->Add(new moveToRange<popContainer, vectorType, evalType>());
	}

	int Init(generalInfoProvider *p){
		//init population
		if(!abstractGeneralMethod<popContainer>::Init(p)) EXIT0("Merging Init: general method init unsuccessfull")
		
		//create index array:
		#if USE_CUDA
			CUDA_CALL("Merging Malloc",cudaMalloc(&indices, sizeof(int) * this->pop->GetPopSize() * this->pop->GetPopsPerKernel()))
		#else
			indices = new int[this->pop->GetPopSize()* this->pop->GetPopsPerKernel()];
		#endif

		//init sorting with appropriate ranges
		fRange = this->pop->GetOffsprRange();
		wRange = makeRange(this->pop->GetPopSize(), 2*this->pop->GetPopSize());
		//DEBUG:
		//fRange = wRange = this->pop->GetPopRange();
		if(! this->methods[0]->Init(this)) EXIT0("Merging Init: sort submethod init unsuccessfull")

		//init moving (only wRange needed)
		wRange = this->pop->GetPopRange();
		if(! this->methods[1]->Init(this)) EXIT0("Merging Init: move submethod init unsuccessfull")
		return 1;
	}

	//propper deallocation
	~replaceMerging<popContainer, vectorType, evalType>(){
		#if USE_CUDA
			cudaFree(indices);
		#else
			delete [] indices;
		#endif
	}

	//override virtual method
	int* GetIndexArray(){return indices;}
	//theese are changing in Init
	range GetWorkingRange(){return wRange;}
	range GetFullRange(){return fRange;}

};
#endif
