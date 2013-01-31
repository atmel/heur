#ifndef __HEURISTICS_PROVIDERS__
#define __HEURISTICS_PROVIDERS__

#include "heuristics.h"

//general abstract provider
class generalInfoProvider{
  public:
  virtual void foo(){}
};

template<class popContainer>
class populationProvider : public virtual generalInfoProvider{
public:
	virtual popContainer* GetPopulationContainer() = 0;
};

template<class archContainer>
class archiveProvider : public virtual generalInfoProvider{
public:
	virtual archContainer* GetArchiveContainer() = 0;
};

class rangeProvider : public virtual generalInfoProvider{
public:
	virtual range GetWorkingRange() = 0;
	virtual range GetFullRange() = 0;
};

class mateProvider : public virtual generalInfoProvider{
public:
	virtual int* GetMatingPool() = 0;
	virtual int GetMatingPoolSize() = 0;
};

//size of array is provided via workingRange
class sortResourceProvider : public virtual generalInfoProvider{
public:
	virtual int* GetIndexArray() = 0;
};

//returns arity of operation (i.e. crossover)
class arityProvider : public virtual generalInfoProvider{
public:
	virtual int GetArity() = 0;
};
#endif