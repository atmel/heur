#ifndef __VYZKUM_CROSSOVER__
#define __VYZKUM_CROSSOVER__

template<class popComponent, int mateSize>
class crossoverMethod{

	public:
	virtual int Init(){};

	virtual int PerformCrossover() = 0;
	
};

#endif