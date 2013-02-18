#ifndef __HEURISTICS_COOLED_GAUSS_MUT__
#define __HEURISTICS_COOLED_GAUSS_MUT__

#include "heuristics.h"
#include "gaussianMutation.h"

/*
	We will use gaussian mutation kernel... if it is possible due to file scope
*/

template<class popContainer, class round, class cool>
class cooledGaussianNoiseMutation: public slaveMethod<popContainer>, public stochasticMethod{
	int threadCount;
	const float T0;
	int gen;
public:
	cooledGaussianNoiseMutation<popContainer,round,cool>(float Temperature0):T0(Temperature0), gen(10){};

	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("cooled gausian mutation slave Method: slave init unsuccessfull")
		threadCount = std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length);
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("cooled gausian mutation slave Method: stochastic method init unsuccessfull")
		return 1;
	}

	int Perform(){
		float T = cool::cool(T0,gen);
		D("temp: %f",T)
	#if USE_CUDA
		CUDA_CALL("cooled gauss mut kernel",(GaussianNoiseKernel <popContainer,round> <<<this->pop->GetPopsPerKernel(), threadCount>>>
			(*(this->pop),this->devStates,this->workingRange,/*oooo*/ T /*ooo*/)));
	#else
		hrand::float2 rnd;
		for(int id = this->workingRange.lo; id < this->workingRange.hi; id++){
			for(int j=0; j < this->pop->GetDim()-1; j+=2){
				rnd = hrand::rand_normal2();
				this->pop->RangeComponent(id,j) += round::round(rnd.x*T);
				this->pop->RangeComponent(id,j+1) += round::round(rnd.y*T);
				//std::cout << round::round(rnd.x*sigma2) << ", "<< round::round(rnd.y*sigma2) << "; ";
			}
			//proces last odd component if any
			if(this->pop->GetDim()%2){
				//form stack object... ok?
				this->pop->RangeComponent(id,this->pop->GetDim()-1) += round::round(hrand::rand_normal2().x*T);
			}
			//std::cout << round::round(hrand::rand_normal2().x*sigma2) << "\n";
		}
	#endif
		//inc generation
		gen++;
	return 1;
	}
};


#endif