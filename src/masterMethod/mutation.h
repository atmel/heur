#ifndef __HEURISTICS_MUTATION_MASTER__
#define __HEURISTICS_MUTATION_MASTER__

#include "heuristics.h"
#include <cmath>

/*
	This ranged master method is supposed to contain one or chain of slave mutations.
	Use this wrapper if previous stage had probability of happening (i.e. crossover),
	given previous stage probability, this will set ranges to emulate statistical independence 
	of both stages (works only for TWO now) without need to do random permutation and preserving 
	vector character of algorithm (it will work with continuous block).

	Ex: let crossoved have 50% probability (first half of offspring is made by breeding, last half only by copy)
	let this mutation have also prob. of 50%.
	The result will be: 
	1./4 only crossover
	2./4 cross + mutation
	3./4 only mutation
	4./4 unchanged (copy)
	
	Again, it emulates statistical independence only for this and the previous stage ... more will need random permutation
	We expect that previous stage run on the same range as this
*/

template<class popContainer>
class mutationWrapper: public masterMethod<popContainer>, public rangedMethod{
protected:
	const float prevProb, mutProb;
	range wRange;

public:
	mutationWrapper(float crossProb, float mutProb):prevProb(crossProb), mutProb(mutProb){};

	int Init(generalInfoProvider *p){
		//this obejct must be initialized first (submethods will need it's pop and so on)
		if(!abstractGeneralMethod<popContainer>::Init(p)) EXIT0("MutationWrapper Init: general method init unsuccessfull")
		if(!rangedMethod::Init(p)) EXIT0("MutationWrapper Init: ranged method init unsuccessfull")

		//set ranges
		int prevFrac = floor(this->workingRange.length*prevProb);
		int currFrac = floor(this->workingRange.length*mutProb);
		int overlap = floor(this->workingRange.length*mutProb*prevProb); //stat. independent
		wRange = makeRange(this->workingRange.lo + prevFrac - overlap, this->workingRange.lo + prevFrac - overlap + currFrac);

		//init all submethods with this object!
		for(int i=0;i < this->methods.size(); i++) 
			if(! this->methods[i]->Init(this)) EXIT0("Composed Init: submethod init unsuccessfull, method number: %d", i)

		return 1;
	}

	//override virtual
	range GetWorkingRange(){return wRange;}
	range GetFullRange(){return this->fullRange;}

};

#endif