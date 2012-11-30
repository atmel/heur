#ifndef __VYZKUM_REPRODUCTION__
#define __VYZKUM_REPRODUCTION__

#include "heuristics.h"
#include<stdlib.h>
#include<time.h>
#include<algorithm>

template<class popContainer, int mateSize>
class reproductionMethod{
	protected:
	//initialized in constructor
	popContainer &pop;
	int *mate;

	public:
	virtual int PerformReproduction() = 0;
	virtual int Init(const basicPopulation<popContainer> &p){
		pop = p.GetPopulationContainer();
		//through the mating pool can be unnecesary for some types of reproduction
		#if USE_CUDA
			cudaMalloc(&mate,sizeof(int)*mateSize*pop.GetPopsPerKernel());
		#else
			mate = new int[mateSize];
		#endif
		return 1;
	}

	virtual int Finalize(){
		#if USE_CUDA
			cudaFree(mate);
		#else
			delete [] mate;
		#endif
	}
};


/* Fill offspring with population copy.
	popSize == offsprSize is expected
*/
template<class popContainer>
class plainCopyReproduction : public reproductionMethod<popContainer,0>{
	
	public:
	int PerformReproduction(){
#if USE_CUDA
		cudaMemcpy(pop.GetOffsprComponentArray(),pop.GetPopComponentArray(),
			pop.GetVectorTypeSize()*pop.GetPopSize()*pop.GetDim(), cudaMemcpyDeviceToDevice);
#else
		memcpy(pop.GetOffsprComponentArray(),pop.GetPopComponentArray(),
			pop.GetVectorTypeSize()*pop.GetPopSize()*pop.GetDim());
#endif
		return 1;
	}
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