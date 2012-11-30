#ifndef __VYZKUM_POPULATION__
#define __VYZKUM_POPULATION__

#include "heuristics.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

/*	Basic population -- no sence to be without archive
*/
template<class popContainer, class archContainer>
class basicPopulation : public populationProvider<popContainer>, public archiveProvider<archContainer>{
	protected:
	int generationCount;
	std::vector< abstractGeneralMethod<popContainer>* > initialization;
	std::vector< abstractGeneralMethod<popContainer>* > loop;

	popContainer *pop;
	archContainer *arch;

	public:
	basicPopulation<popContainer,archContainer>(popContainer* p, archContainer* a):
	  generationCount(0){
		
		initialization.clear();
		loop.clear();
		pop = p;
		arch = a;
	}
	
	inline int GetGeneration() const {return generationCount;}
	popContainer* GetPopulationContainer(){return pop;}
	archContainer* GetArchiveContainer(){return arch;}

	int AddInitialization(abstractGeneralMethod<popContainer>* m){
		initialization.push_back(m);
		D("Added new inititalization")
		return 1;
	}
	int AddExecution(abstractGeneralMethod<popContainer>* m){
		loop.push_back(m);
		D("Added new execution")
		return 1;
	}

/*	virtual int Restart(){
	
	}
*/
	virtual int Init(){
		//initilize init part
		int result = 1;
		for(int i=0;i<initialization.size();i++){
			result *= initialization[i]->Init(this);
		}
		if(!result) return 0;

		//initialize loop part
		for(int i=0;i<loop.size();i++){
			result *= loop[i]->Init(this);
		}
		if(!result) return 0;

		//run init part
		for(int i=0;i<initialization.size();i++){
			result *= initialization[i]->Perform();
		}
		return result;
	}

	virtual int NextGeneration(){
		int result = 1;
		for(int i=0;i<loop.size();i++){
			result *= loop[i]->Perform();
		}
		return result;
	}
};

//===================================================================================
/*
template<class popContainer, class archives>
class archivedPopulation : public basicPopulation<popContainer>{
	protected:

	archives *arch;

	public:
	archivedPopulation<popContainer,archives>(int pSize, int oSize, std::string outFile):
	  basicPopulation<popContainer>(pSize, oSize){
		arch = new archives();
	}
	
	//convenience methods for process stages
	inline archives& GetArchives() const {return *arch;}

	virtual bool Create() = 0;
};*/

//------------------------------------------------------------------------------------------------------------------------
/*
	More spetialized populations follow:
		- if some part does not need more specific information, it can use pointer to theese populations
*/
/*
template<class popContainer>
class basicSingleObjPopulation : public abstractBasicPopulation<popContainer>{
	public:
	//typedef singleObjectiveCandidate<dim,vectorType,evalType> specSingleObjCandidate; 

	protected:
	//history
	//vector<evalType> fitnessArchive;

	public:
	basicSingleObjPopulation<popContainer>(int pSize, int oSize) :
		abstractBasicPopulation<popContainer>(pSize,oSize){};

	//inline specSingleObjCandidate** GetPopulation(){return reinterpret_cast<specSingleObjCandidate**>(this->pop);}
	//inline specSingleObjCandidate** GetOffspring(){return reinterpret_cast<specSingleObjCandidate**>(this->offspr);}
	//inline int GetGeneration(){return generationCount;}
	//inline evalType GetHistoricFitness(int idx){return this->fitnessArchive[idx];}
};
*/
/*
template<int dim, typename vectorType, typename evalType>
class basicSingleObjAnnePopulation : public  basicSingleObjPopulation<dim,vectorType,evalType>{
	double temperature;
	public:
	basicSingleObjAnnePopulation<dim,vectorType,evalType>(int pSize, int oSize, int bestA, double t) :
		basicSingleObjPopulation<dim,vectorType,evalType>(pSize,oSize,bestA), temperature(t){}
};
*/

//--------------------------------------------------------------------------------------------

////vectorType should be signed int. After mutation all candidates are normalised to have their center of gravity at (0,0)
//template<int dim, typename vectorType, typename evalType, 
//	template<int,typename,typename> class _evaluationMethod, 
//	template<int,typename> class _mutationMethod,
//	template<int,typename,typename> class _mergingMethod, 
//	template<int,typename,typename> class _reproductionMethod>
//class testingClassicGOPopulation : public basicSingleObjPopulation<dim,vectorType,evalType>{
//	bool firstCreation;
//
//	public:
//	typedef _mutationMethod<dim,vectorType> specMutMethod;
//	typedef _reproductionMethod<dim,vectorType,evalType> specRepMethod;
//	typedef _evaluationMethod<dim,vectorType,evalType> specEvaMethod;
//	typedef _mergingMethod<dim,vectorType,evalType> specMerMethod;
//	typedef singleObjectiveCandidate<dim,vectorType,evalType> specSingleObjCandidate; 
//
//	//typedef abstractPopulation<dim,vectorType,evalDim,evalType> specAbstPopulation;
//	//typedef candidate<dim,vectorType,evalDim,evalType> specCandidate;
//	protected:
//		specEvaMethod eva;
//		specMerMethod mer;
//		specMutMethod mut;
//		specRepMethod rep;
//		//plainCopyReproduction<dim,vectorType,evalType,1> pcrep;
//
//	public:
//	testingClassicGOPopulation<dim,vectorType,evalType,
//		_evaluationMethod,_mutationMethod,
//		_mergingMethod,_reproductionMethod>(int pSize, int oSize, int bestA) :
//		basicSingleObjPopulation<dim,vectorType,evalType>(pSize,oSize,bestA),firstCreation(true){}
//
//	bool Create(){
//		if(firstCreation){
//			eva.Init(this);
//			mer.Init(this);
//			mut.Init(this);
//			rep.Init(this);
//		}
//		this->fitnessArchive.clear();
//		this->generationCount = 0;
//		//pcrep.Init(this);
//
//		//create initial population
//		#define bigrand() (rand() | rand() << 16)
//		for(int i=0;i < this->populationSize; i++){
//			if(firstCreation) this->pop[i] = new specSingleObjCandidate();
//			for(int j=0; j<dim; j++){
//				this->pop[i]->components[j] = (rand()-RAND_MAX/2); //bigrand();
//			}
//		}
//		//initialize offspring
//		for(int i=0;i < this->offspringSize; i++){
//			if(firstCreation) this->offspr[i] = new specSingleObjCandidate();
//			for(int j=0; j<dim; j++){
//				this->offspr[i]->components[j] = 0;
//			}
//		}
//		//initially evaluate population
//		eva.PerformEvaluation(true);
//		
//		//prepare graphics
//		//if(firstCreation) InitGraphics();
//
//		//to prevent reallocation in other runs
//		firstCreation = false;
//		return true;
//	}
//	//error check
//	/*
//	bool BadCand(){
//		for(int i=0;i<populationSize+offspringSize;i++){
//			for(int j=0;j<dim;j++){
//				if(pop[i]->components[j] == -2147483648) return true;
//			}
//		}
//		return false;
//	}
//	*/
//	void PrintPop(){
//		cout.fill(' ');
//		cout <<'\n';
//		for(int i=0;8*i < this->populationSize;i++){
//			for(int k=0;k<dim;k++){
//				for(int j=0;j<8;j++){
//					if(i*8+j >= this->populationSize){cout<<'\n';break;}
//					cout << setw(10) << right << this->GetPopulation()[i*8+j]->components[k];
//				}
//			}
//			cout << "--------------------------------\n";
//		}
//	}
//
//	bool NextGeneration(){
//		char b;
//		//PrintPop();
//		/*if(GetGeneration()<1000)*/ rep.PerformReproduction();
//		//else pcrep.PerformReproduction();
//		//if(BadCand()){
//		//	cout << "BAD after reproduction";
//		//	//PrintPop();
//		//	cin >> b;
//		//}
//		mut.PerformMutation();
//		//if(BadCand()){
//		//	cout << "BAD after mutation";
//		//	//PrintPop();
//		//	cin >> b;
//		//}
//		eva.PerformEvaluation();
//		//if(BadCand()){
//		//	cout << "BAD after evaluation";
//		//	//PrintPop();
//		//	cin >> b;
//		//}
//		mer.PerformMerge();
//		//if(BadCand()){
//		//	cout << "BAD after merge";
//		//	//PrintPop();
//		//	cin >> b;
//		//}
//		//print info about population
//		cout.precision(2);
//		cout << fixed;
//		cout << ".";
//		for(int i=0;i < this->populationSize; i++){
//			//if(i==populationSize) cout << "---\n";
//			//cout << /*GetPopulation()[i]->components[0] << " " <<*/ GetPopulation()[i]->fitness << '\n';
//		}
//		//DrawScene<dim,vectorType,evalType>(populationSize,pop);
//		
//	//perform restart check, update history
//		//compute average fitness
//		evalType avgFit=0;
//		for(int i=0;i< this->populationSize;i++){
//			avgFit += ((specSingleObjCandidate**)this->pop)[i]->fitness;
//		}
//		avgFit /= this->populationSize;
//		this->fitnessArchive.push_back(avgFit);
//
//		this->generationCount++;
//		return true;
//	}
//};


#endif