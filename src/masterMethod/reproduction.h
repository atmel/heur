#ifndef __VYZKUM_REPRODUCTION__
#define __VYZKUM_REPRODUCTION__

#include "heuristics.h"
#include<algorithm>

/*
	Works on population-offspring logic, sets the right ranges for slave selection and crossover.
	Expects methods[0] be selection and method[1] crossover added after construction
*/
template<class popContainer>
class reproductionMethod: public masterMethod<popContainer>, public mateProvider{
	protected:
	int *mate;
	const float crossProb;
	int mateSize, crossFrac;	//how many will be crossovered based on crossProb
	range wRange, fRange;

	public:
	//init crossProb in constructor
	reproductionMethod<popContainer>(float crossProbability):crossProb(crossProbability){};
	int Init(generalInfoProvider *p){
		//init population
		if(!abstractGeneralMethod<popContainer>::Init(p)) EXIT0("Merging Init: general method init unsuccessfull")
		
		//check two methods present
		if(this->methods.size() != 2) EXIT0("Reproduction: does not contain exactly 2 submethods")
		
			//determine mateSize
		arityProvider* ap = dynamic_cast<arityProvider*>(this->methods[1]);
		if(ap == NULL) EXIT0("Reproduction: crossover (method[1]) to arityProvider cast unsuccesfull")
		crossFrac = this->pop->GetOffsprSize()*crossProb;
		mateSize = (this->pop->GetOffsprSize() - crossFrac) + crossFrac*ap->GetArity();

		D("Reproduction init: mateSize is: %d", mateSize)
		//alloc mate
		#if USE_CUDA
			CUDA_CALL("reproduction Malloc",cudaMalloc(&mate, sizeof(int) * mateSize * this->pop->GetPopsPerKernel()))
		#else
			mate = new int[mateSize];
		#endif

		//init selection with appropriate ranges
		wRange = fRange = this->pop->GetPopRange();
		if(! this->methods[0]->Init(this)) EXIT0("Reproduction Init: selection submethod init unsuccessfull")

		//init reproduction
		fRange = this->pop->GetOffsprRange();
		wRange = makeRange(fRange.lo, fRange.lo + crossFrac);
		if(! this->methods[1]->Init(this)) EXIT0("Reproduction Init: crossover submethod init unsuccessfull")
		return 1;
	}

	~reproductionMethod<popContainer>(){
		#if USE_CUDA
			cudaFree(mate);
		#else
			delete [] mate;
		#endif
	}

	//override virtual method
	int* GetMatingPool(){return mate;}
	int GetMatingPoolSize(){return mateSize;}
	//theese are beeing changed in Init
	range GetWorkingRange(){return wRange;}
	range GetFullRange(){return fRange;}
	
};

//-----------------------BIA repro-----------------------------

////classic GO reproduction with 2-tournalent selection and n-point crossover
//template<int dim, typename vectorType, typename evalType, int mateSize, 
//	template<int,typename,typename,int> class _selectionMethod,int parasitismRate, int maxCrossPoints>
//class biaReproduction : public reproductionMethod<dim,vectorType,evalType,mateSize>{
//	typedef _selectionMethod<dim,vectorType,evalType,mateSize> specSelMethod;
//	HEURISTICS_COMMON_TYPEDEFS_TWO_PARAMS
//	HEURISTICS_COMMON_TYPEDEFS_THREE_PARAMS
//	
//	private: 
//		specSelMethod sel;
//		
//
//	public: 
//	int Init(specAbstBasPopulation *p){
//		reproductionMethod<dim,vectorType,evalType,mateSize>::Init(p);
//		sel.Init(reinterpret_cast<specSingleObjPopulation*>(p),reinterpret_cast<specSingleObjCandidate**>(this->mate));
//		srand(time(NULL));
//		return 1;
//	}
//
//	int PerformReproduction(){
//		sel.PerformSelection();
//		//now we have mating pool full of candidates to crossover
//
//		//n-point crossover n from 1 to maxCrossPoints
//		int cross[maxCrossPoints+1], crossPoints;
//		int i,j,k;
//		bool parasitism;
//		vectorType distMax;
//
//		for(i=0; i< this->offsprSize; i++){
//			//randomize number of cross points and initialize them
//			crossPoints = (rand() % maxCrossPoints)+1;
//			for(j=0;j<crossPoints;j++){
//				cross[j] = (rand() % (dim/2))*2;	//aligned to whole atom coordinates
//			}
//			sort(cross,cross+crossPoints);
//			//just as a stopper
//			cross[crossPoints] = dim;
//			
//			//find out if crossover is with parasite or not
//			parasitism = false;
//			if(rand()%100 <= parasitismRate){
//				parasitism=true;
//				//init distMax
//				distMax = abs(this->mate[2*i]->components[0]);
//				for(j=1;j<dim; j++){
//					if(distMax < abs(this->mate[2*i]->components[j])) distMax = abs(this->mate[2*i]->components[j]);
//				}
//			}
//			//crossover
//			k=0;
//			for(j=0;j<crossPoints+1;j++){
//				if(!(j % 2)){	//copy 1st parent part
//					for(;k<cross[j];k++) this->offspr[i]->components[k] = this->mate[2*i]->components[k];
//				}else{			//copy 2nd parent or parasite
//					if(!parasitism){	
//						for(;k<cross[j];k++) this->offspr[i]->components[k] = this->mate[2*i+1]->components[k];
//					}else{
//						//parasite
//						for(;k<cross[j];k++) this->offspr[i]->components[k] = 
//							(rand()%((vectorType)(2*distMax*1.1))) - (vectorType)(distMax*1.1);
//					}
//				}
//			}
//		}
//		return 1;
//	}
//};

#endif