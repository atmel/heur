#ifndef __VYZKUM_MUTATION__
#define __VYZKUM_MUTATION__

#include "heuristics.h"
#include<cstdlib>
#include<time.h>
#include<math.h>

//using namespace std;

template<class popContainer>
class mutationMethod{
	protected:
	//reference initialized in constructor
	popContainer &pop;

	public:
	virtual int PerformMutation() = 0;
	virtual int Init(const basicPopulation<popContainer> &p){
		pop = p.GetPopulationContainer();
		return 1;
	}
	virtual int Finalize(){};
};

//Takes care of random initialization, in case of GPU even care of devStates allocation, initialization and cleanup!
template<class popContainer>
class stochasticMutationMethod : public mutationMethod<popContainer>{
	protected:
	curandState *devStates;

	public:
	virtual int PerformMutation() = 0;
	virtual int Init(const basicPopulation<popContainer> &p){
		mutationMethod<popContainer>::Init(p);
		//allocates space for max(MAX THREADS, <choose the right size for the object>) states
		const int characteristicSize = p.GetOffsprSize();
		hrand::SetupRandomGeneration(devStates,time(NULL),std::min(MAX_THREADS_PER_BLOCK,characteristicSize),p.GetPopsPerKernel());
		return 1;
	}
	virtual int Finalize(){
#if USE_CUDA
		cudaRandomFinalize(devStates);
#endif
	};
};

//=============================================================================================================================

#if USE_CUDA
/* This kernel does the gaussian mutation in the same fashion as CPU and each thread processes single candidate.
	As generating offspring is stochastic process and candidates are independent random variables, we need 
	not introduce another randomness and we just process first n candidates, where n is fraction corresponding to mut. rate
*/
__global__ void GaussianNoiseKernel<class popContainer, int sigma>(popContainer pop, curandState *state, int count, int begin=0){
	int id = threadIdx.x + begin; //apply shift in case of multiple mutations?
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	curandState localState;
	int2 rnd;
	localState = state[stableId]; //load generator state

	//proces candidates [begin,begin+count)
	for(int i=0; i < REQUIRED_RUNS(count); i++, id += blockDim.x){
		if(id >= begin+count){ //complete
			state[stableId]=localState; //save state
			return;
		}

		//process even number of dimensions
		#pragma unroll
		for(int j=0; j < pop.GetDim()-1; j+=2){
			rnd = curand_normal2int(&localState,sigma);
			pop.OffsprComponent(blockIdx.x,id,j) += rnd.x;
			pop.OffsprComponent(blockIdx.x,id,j+1) += rnd.y;
		}
		//proces last odd component if any
		if(pop.GetDim()%2){
			//form stack object... ok?
			pop.OffsprComponent(blockIdx.x,id,pop.GetDim()-1) += curand_normal2int(&localState,sigma).x;
		}
		//__syncthreads();
	}
}

//specialized for 100% mutation rate
__global__ void CauchyanNoiseKernel<class popContainer, int scale>(popContainer pop, curandState *state, int count, int begin=0){
	int id = threadIdx.x + begin; //apply shift in case of multiple mutations?
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	curandState localState;
	localState = state[stableId]; //load generator state

	//proces candidates [begin,begin+count)
	for(int i=0; i < REQUIRED_RUNS(count); i++, id += blockDim.x){
		if(id >= begin+count){ //complete
			state[stableId]=localState; //save state
			return;
		}
		#pragma unroll
		for(int j=0; j < pop.GetDim(); j++){
			pop.OffsprComponent(blockIdx.x,id,j) += curand_cauchyInt(&localState,scale);
		}
	}
}

#endif

template<class popContainer, int sigma, int rate>
class gaussianNoise: public stochasticMutationMethod<popContainer>{
	using this->pop;
	using this->devStates;

public:
	int PerformMutation(){
		const int count = (pop.GetOffsprSize()*100)/rate;
#if USE_GPU
		GaussianNoiseKernel<popContainer,sigma>
			<<<pop.GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,count)>>>(pop,devStates,count);
#else
		hrand::int2 rnd;
		for(int id = 0; id < count; id++){
			for(int j=0; j < pop.GetDim()-1; j+=2){
				rnd = hrand::rand_normal2(sigma);
				pop.OffsprComponent(id,j) += rnd.x;
				pop.OffsprComponent(id,j+1) += rnd.y;
			}
			//proces last odd component if any
			if(pop.GetDim()%2){
				//form stack object... ok?
				pop.OffsprComponent(id,pop.GetDim()-1) += hrand::rand_normal2(sigma).x;
			}
		}
#endif
	}
};

template<class popContainer, int scale, int rate>
class cauchyanNoise: public stochasticMutationMethod<popContainer>{
	using this->pop;
	using this->devStates;
public:
	int PerformMutation(){
		const int count = (pop.GetOffsprSize()*100)/rate;
#if USE_GPU
		CauchyanNoiseKernel<popContainer,sigma>
			<<<pop.GetPopsPerKernel() , std::min(MAX_THREADS_PER_BLOCK,count)>>>(pop,devStates,count);
#else
		for(int id = 0; id < count; id++){
			for(int j=0; j < pop.GetDim()-1; j+=2){
				pop.OffsprComponent(id,j) += hrand::rand_cauchyInt(scale);
			}
		}
#endif
	}
};

#include "mutationBIA.h"

//moves only by 1 (Hamming distance), it mutates offspring only, vectorType should be integer
//template<int dim, typename vectorType>
//class randomNbDisplace : public mutationMethod<dim,vectorType>{
//	HEURISTICS_COMMON_TYPEDEFS_TWO_PARAMS
//	
//	public:
//	int Init(specAbstBasPopulation *p){
//		mutationMethod<dim,vectorType>::Init(p);
//		srand((int)time(NULL));
//		return 1;
//	};
//	int PerformMutation(){
//		//component to change -- direction actually
//		int com;
//		int move;
//		for(int i=0; i < this->offsprSize; i++){
//			com = rand() % dim;
//			move = (rand() & 1)*2 - 1;	//= +-1
//			this->offspr[i]->components[com] += move;
//			//pertubation - back to border (bad but this is just example)
//			if(this->offspr[i]->components[com] > this->upperLimit[com]) this->offspr[i]->components[com] -= 2;
//			if(this->offspr[i]->components[com] < this->lowerLimit[com]) this->offspr[i]->components[com] += 2;
//		}
//		return 1;
//	}
//};

//-----------------------BIA----------------------------

//bia gaussian mutation (displacement) of n points

//OLD CODE

//template<int dim, typename vectorType, int sigma, int mutRate, int maxMutPoints>
//class gaussianDisplace : public mutationMethod<dim,vectorType>{
//	HEURISTICS_COMMON_TYPEDEFS_TWO_PARAMS
//	
//	public:
//	int Init(specAbstBasPopulation *p){
//		mutationMethod<dim,vectorType>::Init(p);
//		srand((int)time(NULL));
//		return 1;
//	};
//
//	/*bool BadCand(){
//		for(int i=0;i<offsprSize;i++){
//			for(int j=0;j<dim;j++){
//				if(this->offspr[i]->components[j] == -2147483648) return true;
//			}
//		}
//		return false;
//	}*/
//
//	//mutates one point
//	int PerformMutation(){
//		#define PI 3.1415926535897932
//		#define MAXR 10000
//		//mutate one point
//
//		char b;
//
//		double R,theta,rnd;
//		int pos,mutPoints;
//		for(int i=0; i< this->offsprSize; i++){
//			if((rand() % 100) > mutRate) continue; // no mutation
//			// +1 to eliminate NaN, +-INF
//			rnd=(double)(rand())/(RAND_MAX);
//			R = sqrt(-2*log(rnd==0?0.0001:rnd))*sigma;
//			theta = 2*PI*(double)rand()/RAND_MAX;
//
//			mutPoints = (rand() % maxMutPoints)+1;
//			for(int j=0; j<mutPoints; j++){
//				pos = rand() % (dim/2);
//				//mutate all points with the same value
//				this->offspr[i]->components[2*pos] += R*cos(theta);
//				this->offspr[i]->components[2*pos+1] += R*sin(theta);
//			}
//		}
//
//		/*if(BadCand()){
//			cout << "BAD after primal mutation";
//			cin >> b;
//		}*/
//
//		//normalize ... pertubation done automaticaly by overflow
//		long wx,wy;
//		for(int i=0; i< this->offsprSize; i++){
//			wx=wy=0;
//			for(int j=0; j<dim; j+=2) wx += this->offspr[i]->components[j];
//			for(int j=1; j<dim; j+=2) wy += this->offspr[i]->components[j];
//			wx /= dim/2;
//			wy /= dim/2;
//			for(int j=0; j<dim; j+=2) this->offspr[i]->components[j] -= wx;
//			for(int j=1; j<dim; j+=2) this->offspr[i]->components[j] -= wy;
//		}
//
//		/*if(BadCand()){
//			cout << "BAD after normalisation";
//			cin >> b;
//		}*/
//
//		//and randomly rotate with mutation rate -- just try
//		//double rotM[4], phi;
//		//vectorType oldx, oldy;
//		//for(int i=0; i<offsprSize; i++){
//		//	if((rand() % 100) > mutRate) continue; // no mutation
//		//	//init rotation matrix
//		//	// +1 to eliminate NaN, +-INF
//		//	R = sqrt(-2*log((double)(rand()+1)/(RAND_MAX+1)))*0.3;
//		//	theta = 2*PI*(double)rand()/RAND_MAX;
//		//	//phi with normal distribution.. 0.1 ~ 6 deg.
//		//	phi = R*cos(theta);
//		//	rotM[0]=rotM[3] = cos(phi);
//		//	rotM[1] = -sin(phi);
//		//	rotM[2] = -rotM[1];
//
//		//	//rotate every point
//		//	for(int j=0;j<dim/2;j++){
//		//		oldx = offspr[i]->components[2*j];
//		//		oldy = offspr[i]->components[2*j+1];
//		//		offspr[i]->components[2*j] =  rotM[0]*oldx + rotM[1]*oldy;
//		//		offspr[i]->components[2*j+1] =rotM[2]*oldx + rotM[3]*oldy;
//		//	}
//		//}
//		/*if(BadCand()){
//			cout << "BAD after rotation";
//			cin >> b;
//		}*/
//
//		return 1;
//	}
//};

#endif