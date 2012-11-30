#ifndef __HEURISTICS_ABSTRACT_METHOD__
#define __HEURISTICS_ABSTRACT_METHOD__

#include "heuristics.h" 
#include <algorithm>
#include <ctime>

//See UML diagram
template<class popContainer>
class abstractGeneralMethod{
	protected:
	popContainer *pop;

	public:
	virtual int Init(generalInfoProvider *p){
		populationProvider<popContainer>* pp = dynamic_cast< populationProvider<popContainer>* >(p);
		//check for right support
		if(pp == NULL) return 0;

		pop = pp->GetPopulationContainer();
		return 1;
	}
	virtual int Perform() = 0;
};

//------------------------------------------------------------------------------------------------------------------
//Takes care of random initialization, in case of GPU even care of devStates allocation, initialization and cleanup!
class stochasticMethod{
	protected:
	curandState *devStates;

	public:
	int Init(const int size, const int popsPerKernel){
		D("Stochastic method initializing")
		//allocates space for min(MAX THREADS, <choose the right size for the object>) states
		hrand::SetupRandomGeneration(&devStates,time(NULL),std::min(MAX_THREADS_PER_BLOCK,size),popsPerKernel);
		return 1;
	}
	~stochasticMethod(){
#if USE_CUDA
		hrand::cudaRandomFinalize(devStates);
#endif
	}
};

//------------------------------------------------------------------------------------------------------------------
class rangedMethod{
	protected:
	range workingRange,fullRange;

	public:
	int Init(generalInfoProvider *p){
		rangeProvider* rp = dynamic_cast< rangeProvider* >(p);
		//check for right support
		if(rp == NULL) return 0;

		workingRange = rp->GetWorkingRange();
		fullRange = rp->GetFullRange();
		return 1;
	}
};
#endif