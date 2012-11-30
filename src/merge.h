#ifndef __VYZKUM_MERGE__
#define __VYZKUM_MERGE__

#include "heuristics.h"
#include<stdlib.h>
#include<time.h>
#include<math.h>

/* General merging methods
	Here, for example the annealing process is taken care of
*/

template<class popContainer>
class mergingMethod{
	protected:
	//reference initialized in constructor
	popContainer &pop;

	public:
	virtual int PerformMerge() = 0;
	virtual int Init(const basicPopulation<popContainer> &p){
		pop = p.GetPopulationContainer();
		return 1;
	}
	virtual int Finalize(){};
};

//Takes care of random initialization, in case of GPU even care of devStates allocation, initialization and cleanup!
template<class popContainer>
class stochasticMergingMethod : public mergingMethod<popContainer>{
	protected:
	curandState *devStates;

	public:
	virtual int PerformMerge() = 0;
	virtual int Init(const basicPopulation<popContainer> &p){
		mergingMethod<popContainer>::Init(p);
		//allocates space for max(MAX THREADS, <choose the right size for the object>) states
		// how many here?! .. popSize is more logical
		const int characteristicSize = p.GetPopSize();
		hrand::SetupRandomGeneration(devStates,time(NULL),std::min(MAX_THREADS_PER_BLOCK,characteristicSize),p.GetPopsPerKernel());
		return 1;
	}
	virtual int Finalize(){
#if USE_CUDA
		cudaRandomFinalize(devStates);
#endif
	}
};
//---------------------------------------------------------------------------------------------------------------
#if USE_CUDA
template<class popContainer>
__global__ void fightParentKernel(popContainer pop){
	int id = threadIdx.x;

	for(int i=0; i < REQUIRED_RUNS(pop.GetPopSize()); i++, id += blockDim.x){
		if(pop.PopFitness(blockIdx.x,id) > pop.OffsprFitness(blockIdx.x,id)){
			pop.MoveToPop(blockIdx.x,id,id);
		}
	}
};
#endif

// Does pure comparison one to one, expects popSize == offsprSize
template<class popContainer>
class fightParent: mergingMethod<popContainer>{
public:
	int PerformMerge(){
#if USE_CUDA
		fightParentKernel<popContainer>
			<<<this->pop.GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,this->pop.GetPopSize())>>>(this->pop);
#else
		for(int i=0; i < this->pop.GetPopSize(); i++){
			if(this->pop.PopFitness(i) > this->pop.OffsprFitness(i)){
				pop.MoveToPop(i,i);
			}
		}
#endif
	}
};

//------------------------------------------------------------------------------------------------------------------
#if USE_CUDA
template<class popContainer, class acceptanceRule>
__global__ void fightParentAnnealedKernel(popContainer p, float temperature, acceptanceRule ar, curandState *state){
	int id = threadIdx.x;

	for(int i=0; i < REQUIRED_RUNS(pop.GetPopSize()); i++, id += blockDim.x){
		if(ar(pop.PopFitness(blockIdx.x,id) - pop.OffsprFitness(blockIdx.x,id), temperature, state)){
			pop.MoveToPop(blockIdx.x,id,id);
		}
	}
}

#endif

// Annealed merge - templatable acceptance rule and cooling schedule
template<class popContainer, class acceptanceRule, class coolingSchedule>
class fightParentAnnealed: stochasticMergingMethod<popContainer>{
	float temperature;
	acceptanceRule accRule;
	coolingSchedule cooling;

	public:
	int PerformMerge(){
#if USE_CUDA
#else
		for(int i=0; i < this->pop.GetPopSize(); i++){
			if( accRule(this->pop.PopFitness(i) - this->pop.OffsprFitness(i),temperature) ){
				pop.MoveToPop(i,i);
			}
		}
#endif
		//for both CPU, GPU, upgrade temperature
		temperature = cooling(this->pop);
	}
};

//#define SWAPO_P(X) tmp = this->pop[(X)]; this->pop[(X)] = this->offspr[(X)]; this->offspr[(X)] = tmp;
//
//template<int dim, typename vectorType, typename evalType>
//class fightParent : public mergingMethod<dim,vectorType,evalType>{
//	HEURISTICS_COMMON_TYPEDEFS_THREE_PARAMS
//	
//	public:
//	int PerformMerge(){
//		if(this->offsprSize != this->popSize) return 0;
//		specSingleObjCandidate *tmp;
//
//		for(int i=0; i < this->popSize; i++){
//			if(this->pop[i]->fitness > this->offspr[i]->fitness){		//minimizing!!
//				SWAPO_P(i);
//			}
//		}
//		return 1;
//	}
//};
//
//#define SWAP(X,Y) {tmp = this->pop[(X)]; this->pop[(X)] = this->pop[(Y)]; this->pop[(Y)] = tmp;}
//#define FIT(X) (this->pop[(X)]->fitness)
//
//template<int dim, typename vectorType, typename evalType>
//class fightPopulation : public mergingMethod<dim,vectorType,evalType>{
//	HEURISTICS_COMMON_TYPEDEFS_THREE_PARAMS
//	
//	protected:
//	int qsortPartial(int first, int last, int split){
//	/*
//		Sorts in ascending order!!
//		Modified qsort - partially sorts array from indice 'first' to indice 'last'
//		splitting it into two partially ordered subarrays where first subarray 
//		is 'split' long. Last-first is supposed to be > 1
//	*/
//		specSingleObjCandidate *tmp;
//
//		//run qsort
//		int pivot, rising, falling;
//		int targetPos = first + split;
//		evalType pivotFit;
//		while(1){
//			//find median of 3 (to be on first+1 pos)
//			if(FIT(first) > FIT(last)) SWAP(first,last);
//			if(FIT(first) > FIT(first+1)) SWAP(first,first+1);
//			if(FIT(first+1) > FIT(last)) SWAP(first+1,last);
//		
//			pivotFit = FIT(first+1);//pivot is at 'first+1' pos (first is already smaller)
//			rising = first+1;	//will be immediatelly incremented
//			falling= last;	//will be immediatelly decremented but last is greated than pivot so OK
//			while(1){
//				do falling--; while(FIT(falling) > pivotFit);	//thanks to med. no need to check underrun
//				//rise until first element >= pivot is found
//				do rising++; while(FIT(rising) < pivotFit);		//also, no overrun check
//				//done and continue or swap and go again
//				if(rising > falling){	
//					SWAP(first+1,falling);	//first+1=pivot index
//					break;
//				}else{					
//					SWAP(rising,falling);
//				}
//			}
//			//check and end/run again
//			if((falling == targetPos)||(falling == targetPos-1)){return 1;}
//			else if(falling > targetPos) last = falling-1; //pivot-1;
//			else first = falling+1; //pivot+1;
//		}
//	}
//
//	//sorts first cnt candidates in ascending order by insertion sort
//	int insertionSortPartial(int first, int last, int cnt){
//		evalType minFit;
//		int minIdx, i,j;
//		specSingleObjCandidate *tmp;
//
//		for(i=0;i<cnt;i++){
//			minIdx = i+first;
//			minFit = FIT(minIdx);
//			for(j=first+i+1; j<=last; j++){
//				if(FIT(j) < minFit){
//					minIdx = j; minFit = FIT(minIdx);
//				}
//			}
//			SWAP(minIdx,i+first);
//		}
//		return 1;
//	}
//
//	public:
//	int PerformMerge(){	//modified qsort
//		qsortPartial(0, this->popSize + this->offsprSize - 1,this->popSize);
//		return 1;
//	}
//};
//
////------------------BIA merge -----------------------------
//
///*
//Merges pop and offspr so that elite from pop is preserved and the rest of pop is replaced
//with best offspring. It inherits fightPopulation so that it can use quicksort
//
//eliteCount determines elite volume
//*/
//template<int dim, typename vectorType, typename evalType, int eliteCount>
//class elitismMerge : public fightPopulation<dim,vectorType,evalType>{
//	HEURISTICS_COMMON_TYPEDEFS_THREE_PARAMS
//	
//	public: 
//	int PerformMerge(){
//		//sort out elites in pop and popSize-eliteCount in offstr
//		if(eliteCount < INSERTION_SORT_TRESHOLD) insertionSortPartial(0,this->popSize-1,eliteCount);
//		else qsortPartial(0,this->popSize-1,eliteCount);
//		//also include treshold ??
//		qsortPartial(this->popSize,this->popSize+this->offsprSize-1,this->popSize-eliteCount);
//
//		specSingleObjCandidate *tmp;
//		/*
//		preserve elite and SWAP the rest -- otherwise some pointers would 
//		be duplicit and pop would be corrupted
//		*/
//		for(int i=eliteCount, j=0; i<this->popSize; i++,j++){
//			tmp = this->pop[i];
//			this->pop[i] = this->offspr[j];
//			this->offspr[j] = tmp;
//		}
//		return 1;
//	}
//};
//
//
//
//
//// TO RECONSIDER
////template<int dim, typename vectorType, int evalDim, typename evalType>
////class fightParentAnnealing : public mergingMethod<dim,vectorType,evalDim,evalType>{
////	public:
////	typedef abstractAnnealingPopulation<dim,vectorType,evalDim,evalType> specAbstAnnePopulation;
////	typedef candidate<dim,vectorType,evalDim,evalType> specCandidate;
////	typedef initializablePart<dim,vectorType,evalDim,evalType> specInitializable;
////
////	private:
////	using specInitializable::p;
////
////	public:
////	int Init(specAbstAnnePopulation *pop){ //we need temperature
////		p=pop;
////		srand((int)time(NULL));
////		return 1;
////	}	
////	int PerformMerge(){
////		if(p->offspringSize != p->populationSize) return 0;
////		specCandidate *tmp;
////		evalType delta;
////
////		for(int i=0; i<p->populationSize; i++){
////			delta = p->pop[i]->fitness() - p->offspr[i]->fitness();
////			if((delta > 0)||((double)rand()/RAND_MAX < exp(delta/static_cast<specAbstAnnePopulation *>(p)->temperature))){		//minimizing!!
////				SWAPO_P(i);
////			}
////		}
////		return 1;
////	}
////};


#endif