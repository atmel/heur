#ifndef __VYZKUM_MUTATION_BIA__
#define __VYZKUM_MUTATION_BIA__

#include "heuristics.h"
#include<stdlib.h>
#include<time.h>
#include<math.h>

#if USE_CUDA
/* This kernel does the gaussian mutation in the same fashion as CPU and each thread processes single candidate.
	This approach may prove unefficient when mutation rate is low or candidates are large!
*/
__global__ void HighRateGaussianNoiseAndCenter<class popContainer, int sigma>(popContainer pop, curandState *state, int count, int begin=0){
		//FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
	int id = threadIdx.x + begin; //apply shift
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	curandState localState;
	//for not using modulo
	int2 rnd;
	//load generator state
	localState = state[stableId];
	long int w0,w1;

	//proces candidates [begin,begin+count)
	for(int i=0; i < REQUIRED_RUNS(count); i++, id += blockDim.x){
		if(id >= begin+count){ //complete
			state[stableId]=localState; //save state
			return;
		}
		//process even number of dimensions
		w0 = w1 = 0;
		#pragma unroll
		for(int j=0; j < pop.GetDim(); j+=2){
			rnd = curand_normal2int(&localState,sigma);
			pop.OffsprComponent(blockIdx.x,id,j) += rnd.x;
			pop.OffsprComponent(blockIdx.x,id,j+1) += rnd.y;
			w0 += pop.OffsprComponent(blockIdx.x,id,j);
			w1 += pop.OffsprComponent(blockIdx.x,id,j+1);
		}
		w0 /= pop.GetDim()/2;
		w1 /= pop.GetDim()/2;
		#pragma unroll
		for(int j=0; j < pop.GetDim(); j+=2){
			pop.OffsprComponent(blockIdx.x,id,j) -= w0;
			pop.OffsprComponent(blockIdx.x,id,j+1) -= w1;
		}
		__syncthreads();
	}
}
#endif

//For molecule optimization
//adds gaussian noise to each component of candidate, moves its center of mass to
template<class popContainer, int sigma, int rate>
class BIAgaussianNoiseAndCenter: public mutationMethod<popContainer>{
	using this->pop;

	curandState *devStates;
public:
	int Init(const basicPopulation<popContainer> &p){
		mutationMethod<popContainer>::Init(p);
		hrand::SetupRandomGeneration(devStates,time(NULL),pop.GetPopSize(),pop.GetPopsPerKernel());

	}
	int PerformMutation(){
#if USE_GPU
		HighRateGaussianNoiseAndCenter<popContainer,sigma,(rate*128)/100>
			<<<pop.GetPopsPerKernel() , std::max(MAX_THREADS_PER_BLOCK,pop.GetPopSize())>>>(pop,devStates);
#else
		hrand::int2 rnd;
		long int w0,w1;
		for(int i=0; i < pop.GetOffsprSize(); i++{
			if((rand()%100) > rate) continue;
			//process components
			w0 = w1 = 0;
			for(int j=0; j < pop.GetDim(); j+=2){
				rnd = hrand::rand_normal2(sigma);
				pop.OffsprComponent(i,j) += rnd.x;
				pop.OffsprComponent(i,j+1) += rnd.y;
				w0 += pop.OffsprComponent(i,j);
				w1 += pop.OffsprComponent(i,j+1);
			}
			w0 /= pop.GetDim()/2;
			w1 /= pop.GetDim()/2;
			for(int j=0; j < pop.GetDim(); j+=2){
				pop.OffsprComponent(i,j) -= w0;
				pop.OffsprComponent(i,j+1) -= w1;
			}
		}
#endif
	}
};

#endif