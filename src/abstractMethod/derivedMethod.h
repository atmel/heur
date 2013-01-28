#ifndef __HEURISTICS_DERIVED_ABSTRACT_METHOD__
#define __HEURISTICS_DERIVED_ABSTRACT_METHOD__

#include "heuristics.h" 
#include <vector>

template<class popContainer>
class composedMethod : public abstractGeneralMethod<popContainer>, public populationProvider<popContainer>{
	protected:
	std::vector< abstractGeneralMethod<popContainer>* > methods;

	public:
	composedMethod<popContainer>(){
		methods.clear();
	}

	virtual int Init(generalInfoProvider *p){
		//methods.clear(); NO, they are registered during construction!!
		//this obejct must be initialized first (submethods will need it's pop and so on)
		if(!abstractGeneralMethod<popContainer>::Init(p)) EXIT0("Composed Init: general method init unsuccessfull")

		//init all submethods with this object!
		for(int i=0;i < methods.size(); i++) 
			if(! methods[i]->Init(this)) EXIT0("Composed Init: submethod init unsuccessfull, method number: %d", i)

		return 1;
	}

	//return this so the calls can be chained
	composedMethod<popContainer>* Add(abstractGeneralMethod<popContainer> *m){
		methods.push_back(m);
		D("Composed: added submethod, total %d methods", (int)methods.size())
		return this;
	}

	//must be virtual -- for adaptive master methods like annealed merge
	virtual int Perform(){
		D("Composed: perform. (%d methods)",(int)methods.size())
		//returns 0 if some submethod returns zero, but still performs all
		int result = 1;
		for(int i=0; i < methods.size();i++){
			result *= methods[i]->Perform();
		}
		return result;
	}

	~composedMethod<popContainer>(){
		//delete subsequent
		for(int i=0; i < methods.size();i++){
			delete methods[i];
		}
		//reset vector
		methods.clear();
	}

	//override abstract method
	popContainer* GetPopulationContainer(){return this->pop;}
};

//------------------------------------------------------------------------------------------------------------------
template<class popContainer>
class slaveMethod : public abstractGeneralMethod<popContainer>, public rangedMethod {
	public:
	//using rangedMethod::workingRange;
	//using rangedMethod::fullRange;
	
	virtual int Init(generalInfoProvider *p){
		if(!abstractGeneralMethod<popContainer>::Init(p)) return 0;
		if(!rangedMethod::Init(p)) return 0;
		return 1;
	}
	virtual int Perform() = 0;
};

//------------------------------------------------------------------------------------------------------------------
template<class popContainer>
class masterMethod : public composedMethod<popContainer>, public rangeProvider {
	public:
	virtual range GetWorkingRange() = 0;
	virtual range GetFullRange() = 0;
};


//-----------------------------------------------------------------------------------------------------------------

#endif