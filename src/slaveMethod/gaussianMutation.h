#ifndef __HEURISTICS_GAUSSIAN_MUTATION__
#define __HEURISTICS_GAUSSIAN_MUTATION__

#include "heuristics.h"

//=============================================================================================================================

#if USE_CUDA
/* 

*/
template<class popContainer, class round>
__global__ void GaussianNoiseKernel(popContainer pop, curandState *state, range rng, float sigma2){
	int id = threadIdx.x + rng.lo;
	const int stableId = threadIdx .x + blockIdx .x * blockDim.x; //for accessing curand states
	curandState localState;
	float2 rnd;
	localState = state[stableId]; //load generator state

	//proces candidates
	for(int i=0; i < REQUIRED_RUNS(rng.length); i++, id += blockDim.x){
		if(id >= rng.hi){ //complete
			state[stableId]=localState; //save state
			return;
		}

		//process even number of dimensions
		#pragma unroll
		for(int j=0; j < pop.GetDim()-1; j+=2){
			rnd = curand_normal2(&localState);
			pop.RangeComponent(blockIdx.x,id,j) += round::round(rnd.x*sigma2, &localState);
			pop.RangeComponent(blockIdx.x,id,j+1) += round::round(rnd.y*sigma2, &localState);
		}
		//proces last odd component if any
		if(pop.GetDim()%2){
			//form stack object... ok?
			pop.RangeComponent(blockIdx.x,id,pop.GetDim()-1) += round::round(curand_normal2(&localState).x*sigma2, &localState);
		}
	}
	//save state
	state[stableId] = localState;
}

#endif

template<class popContainer, class round>
class gaussianNoiseMutation: public slaveMethod<popContainer>, public stochasticMethod{
	int threadCount;
	const float sigma2;
public:
	gaussianNoiseMutation<popContainer,round>(float variance):sigma2(variance){};

	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("gausian mutation slave Method: slave init unsuccessfull")
		threadCount = std::min(MAX_THREADS_PER_BLOCK,this->workingRange.length);
		if(!stochasticMethod::Init(threadCount,this->pop->GetPopsPerKernel())) EXIT0("gausian mutation slave Method: stochastic method init unsuccessfull")
		return 1;
	}

	int Perform(){
	#if USE_CUDA
		CUDA_CALL("gauss mut kernel",(GaussianNoiseKernel <popContainer,round> <<<this->pop->GetPopsPerKernel(), threadCount>>>
			(*(this->pop),this->devStates,this->workingRange,sigma2)));
	#else
		hrand::float2 rnd;
		for(int id = this->workingRange.lo; id < this->workingRange.hi; id++){
			for(int j=0; j < this->pop->GetDim()-1; j+=2){
				rnd = hrand::rand_normal2();
				this->pop->RangeComponent(id,j) += round::round(rnd.x*sigma2);
				this->pop->RangeComponent(id,j+1) += round::round(rnd.y*sigma2);
				//std::cout << round::round(rnd.x*sigma2) << ", "<< round::round(rnd.y*sigma2) << "; ";
			}
			//proces last odd component if any
			if(this->pop->GetDim()%2){
				//form stack object... ok?
				this->pop->RangeComponent(id,this->pop->GetDim()-1) += round::round(hrand::rand_normal2().x*sigma2);
			}
			//std::cout << round::round(hrand::rand_normal2().x*sigma2) << "\n";
		}
	#endif
	return 1;
	}
};

//#include "mutationBIA.h"

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