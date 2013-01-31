#ifndef __HEURISTICS_CANDIDATE_CONTAINER__
#define __HEURISTICS_CANDIDATE_CONTAINER__

#include "heuristics.h"

/* Universal master for all slaves who need to access population as init or non-population methods
	(everything happens in population, there's no offspring)
*/
template<class popContainer>
class popRangedMasterMethod : public masterMethod<popContainer>{
public:
	virtual range GetWorkingRange(){
		return this->pop->GetPopRange();
	}
	virtual range GetFullRange(){
		return this->pop->GetPopRange();
	}
};

//===============================================================================================
/* Universal master for all slaves who need to access offspring, as evaluation method
	(everything happens in population, there's no offspring)
*/
template<class popContainer>
class offsprRangedMasterMethod : public masterMethod<popContainer>{
public:
	virtual range GetWorkingRange(){
		return this->pop->GetOffsprRange();
	}
	virtual range GetFullRange(){
		return this->pop->GetOffsprRange();
	}
};

template<class popContainer, class archContainer>
class popRangedArchivedMasterMethod : public popRangedMasterMethod<popContainer>, public archiveProvider<archContainer>{
protected:
  archiveProvider<archContainer> *ap;
  
public:
	virtual int Init(generalInfoProvider *p){
		ap = dynamic_cast< archiveProvider<archContainer>* >(p);
		if(ap == NULL) EXIT0("popRangedArchivedMasterMethod: unsuccessfull cast to archive provider")
		//methods.clear(); NO, they are registered during construction!!
		//this obejct must be initialized first (submethods will need it's pop and so on)
		if(!popRangedMasterMethod<popContainer>::Init(p)) EXIT0("popRangedArchivedMasterMethod: popRangedMasterMethod init unsuccessfull")
	
		//init all submethods with this object!
		//for(int i=0;i < methods.size(); i++).
		//<-->if(! methods[i]->Init(this)) EXIT0("Composed Init: submethod init unsuccessfull, method number: %d", i)
	
		return 1;
	}
	virtual archContainer* GetArchiveContainer(){return ap->GetArchiveContainer();}
};


#endif