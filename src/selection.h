#ifndef __VYZKUM_SELECTION__
#define __VYZKUM_SELECTION__

#include "heuristics.h"
#include<time.h>
#include<stdlib.h>
#include<algorithm>

template<class popContainer, int mateSize>
class selectionMethod{
	protected:
	//reference initialized in constructor
	popContainer &pop;
	int *mate;

	public:
	virtual int PerformSelection() = 0;
	virtual int Init(const basicPopulation<popContainer> &p, int *_mate){
		pop = p.GetPopulationContainer();
		mate = _mate;
		return 1;
	}
	virtual int Finalize(){};
};

//Takes care of random initialization, in case of GPU even care of devStates allocation, initialization and cleanup!
template<class popContainer, int mateSize>
class stochasticSelectionMethod : public selectionMethod<popContainer>{
	protected:
	curandState *devStates;

	public:
	virtual int PerformSelection() = 0;
	virtual int Init(const basicPopulation<popContainer> &p, int *_mate){
		selectionMethod<popContainer,matesize>::Init(p,_mate);
		//allocates space for max(MAX THREADS, <choose the right size for the object>) states
		const int characteristicSize = mateSize;
		hrand::SetupRandomGeneration(devStates,time(NULL),std::min(MAX_THREADS_PER_BLOCK,characteristicSize),p.GetPopsPerKernel());
		return 1;
	}
	virtual int Finalize(){
#if USE_CUDA
		cudaRandomFinalize(devStates);
#endif
	}
};

//-----------------------------------------------------------------------------------------

#if USE_CUDA
__global__ void TwoTournamentSelection<class popContainer, int mateSize>(popContainer pop, curandState *state){
	int id = threadIdx.x;
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	int one, two;

	for(int i=0; i < REQUIRED_RUNS(mateSize); i++, id += blockDim.x){
		if(id >= mateSize) return; //complete
		//try local copy and then moving state 2 fotward?? ... faster?
		one = curand(&state[stableId]) % pop.GetPopSize();
		two = curand(&state[stableId]) % pop.GetPopSize();
		//access right block of mate!
		//__syncthreads();
		mate[id + blockIdx.x*mateSize] = (pop.PopFitness(blockIdx.x,one) > pop.PopFitness(blockIdx.x,two))?one:two;
	}
}
#endif

template<class popContainer, int mateSize>
class tournamentSelection : public selectionMethod<popContainer,mateSize>{
	//to search in templated parent
	using this->pop;
	using this->mate;
	using this->devStates;

	public:
	int PerformSelection(){			//basic 2-tournament, pop must be smaller than randmax!!
		//specSingleObjCandidate *one, *two;
		#if USE_CUDA
			TwoTournamentSelection<popContainer,mateSize>
				<<<pop.GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,mateSize)>>>(pop,devStates);
		#else		
			int one,two;
			for(int i=0;i<mateSize;i++){
				one = rand() % pop.GetPopSize();
				two = rand() % pop.GetPopSize();
				mate[i] = (pop.PopFitness(one) > pop.PopFitness(two))?one:two;
			}
		#endif
		return 1;
	}
};

#endif