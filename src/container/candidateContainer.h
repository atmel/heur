#ifndef __HEURISTICS_CANDIDATE_CONTAINERR__
#define __HEURISTICS_CANDIDATE_CONTAINERR__

#include "heuristics.h"
#include <cstring>
/*
better memory handling, portability to GPU with advantage of straightforward syntax
and good memory acces pattern "shielded" from a user.
*/

/*
	Basic class for population storage.
	It is meant to be passed to the kernel BY VALUE (plain copies of everything, includung pointers) so all needed variables are
	quickly accessible from const/shared memory -- it is probably the most fast and elegant way how to put these variables in these
	types of memory. Passing it every time may seem abundant but it is more logical since we can have multiple different populations.

	Best Arc, mating pool could be stored here (consider), basic storage supports only population and offspring

	In single population (/offspring/every population )

	All device functions should inline -- check in ptx assembly
*/
template<typename vectorType, typename evalType>
class basicCandidateContainer{
//this pointers points either to RAM or GPURAM 
	/* PopComponent holds componnets in following memory patterns:
		For GPU (multiple pops per kernel):
		pop1components1, off1components1, pop1components2, off1components2, ... ,pop1componentsN, off1componentsN,
		pop2components1, off2components1 ,... (if present)
		because in paralell all first components are accesed at the same time, all second ones and so on. Offspring components are stored 
		right behind for ranged access. *offsprComponent is reseated at the begining of off1components1, which is not its original purpose
		but this is handled internally

		For CPU (single pop always):
		candidate1,candidate2,...candidateN, offspring1, offspring2,... offspringM
		*offsprComponent is reseated at the begining of candidateN so it holds its original purpose
		
	*/  
	vectorType *popComponent;
	vectorType *offsprComponent;

	/* Holds fitness for pop and offspr consecultively in one array.
		In case of mutiple pops, pop and corresponding offspr is paired up for easier loading. Therefore pattern is:
		pop1,offspr1,pop2,offspr2,... *offsprFitness is reseated at the begining of offspr1
	*/
	evalType *popFitness, *offsprFitness;

	const int popSize, offsprSize, dim, popsPerKernel;
	/* In case of multiple populations for faster indexing */
	//const int popStride, offsprStride;
		// stride over whole pop+offspr
		//fitStride;
	//GPU strides -- pop1<->pop2 , comp1<->comp2
	const int wholePopStride, rowStride;

	//limits -- candidate coordinates are in range [lowerLimit, upperLimit) 
	// NOTE THE  [ )  !!! ... pertubation works that way
 	vectorType *upperLimit,*lowerLimit;

	public:
		basicCandidateContainer(int _dim, int _popSize, int _offsprSize, int _popsPerKernel=1):
			popSize(_popSize), offsprSize(_offsprSize), dim(_dim), popsPerKernel(_popsPerKernel),
			rowStride(_popSize+_offsprSize), wholePopStride(_dim*(_popSize+_offsprSize)){
			
#if USE_CUDA
			D("Allocating pop on GPU")
			//allocate multiple populations, offsprings, compute strides
			cudaMalloc(&popComponent, sizeof(vectorType)*dim*(popSize+offsprSize)*popsPerKernel);
			//the first component of the first offspring is just after first compopnent of last candidate in population
			offsprComponent = &popComponent[popSize];
					//cudaMalloc(&offsprComponent, sizeof(vectorType)*dim*offsprSize*popsPerKernel);
			cudaMalloc(&popFitness, sizeof(evalType)*(popSize+offsprSize)*popsPerKernel);
			offsprFitness = &popFitness[popSize];
					//cudaMalloc(&offsprFitness, sizeof(evalType)*offsprSize*popsPerKernel);
			cudaMalloc(&upperLimit, sizeof(vectorType)*dim);
			CUDA_CALL("Last allocation",(cudaMalloc(&lowerLimit, sizeof(vectorType)*dim)))
			
			//memset for testing
			//cudaMemset(popComponent,0,sizeof(vectorType)*dim*(popSize+offsprSize)*popsPerKernel);
			//popStride = dim*popSize;
			//offsprStride = dim*offsprSize;
			//fitStride = popSize+offsprSize;
#else
			popComponent = new vectorType[dim*(popSize+offsprSize)];
			//the first component of the first offspring is after last candidate of population
			offsprComponent = &popComponent[popSize*dim];   //new vectorType[dim*offsprSize];
			popFitness = new evalType[popSize];
			offsprFitness = &popFitness[popSize];  //new evalType[offsprSize];
			upperLimit = new vectorType[dim];
			lowerLimit = new vectorType[dim];
#endif
		}
		
		// propper dealocation
		int Finalize(){
#if USE_CUDA 
			cudaFree(popComponent);
			//cudaFree(offsprComponent);
			cudaFree(popFitness);
			//cudaFree(offsprFitness);
			cudaFree(upperLimit);
			cudaFree(lowerLimit);
#else
			delete [] popComponent;
			//delete [] offsprComponent;
			delete [] popFitness;
			//delete [] offsprFitness;
			delete [] upperLimit;
			delete [] lowerLimit;
#endif
			return 1;
		} 
		//DO NOT DEALOCATE IN DESTRUCTOR -- OBJECT IS PASSED BY VALUE!!
		~basicCandidateContainer(){}

		//common convenience functions
		inline __device__ __host__ int GetPopsPerKernel() const {return popsPerKernel;}
		inline __device__ __host__ int GetDim() const {return dim;}
		inline __device__ __host__ int GetPopSize() const {return popSize;}
		inline __device__ __host__ int GetOffsprSize() const {return offsprSize;}

		inline __device__ __host__ range GetPopRange() const {return makeRange(0,popSize);}
		inline __device__ __host__ range GetOffsprRange() const {return makeRange(popSize,popSize+offsprSize);}

		//meant to use for cudaMemcpy and effective evaluation (passing only one array)
		inline __device__ __host__ int GetVectorTypeSize() const {return sizeof(vectorType);}

		inline __device__ __host__ vectorType GetUpperLimit(int i) const {return upperLimit[i];}
		inline __device__ __host__ vectorType GetLowerLimit(int i) const {return lowerLimit[i];}
		
#if USE_CUDA
		//globIdx determines choosen pop-offspr pair in case popPerKernel > 1
		inline __device__ vectorType& PopComponent(const int globIdx, const int cand, const int comp){
			//column-major!
			return popComponent[wholePopStride*globIdx + cand + rowStride*comp];
		}
		inline __device__ vectorType& OffsprComponent(const int globIdx, const int cand, const int comp){
			//column-major! -- same as above, just different starting point
			return offsprComponent[wholePopStride*globIdx + cand + rowStride*comp];
		}
		inline __device__ evalType& PopFitness(const int globIdx, const int cand){
			return popFitness[rowStride*globIdx + cand];
		}
		inline __device__ evalType& OffsprFitness(const int globIdx, const int cand){
			return offsprFitness[rowStride*globIdx + cand];
		}
		inline __device__ vectorType& RangeComponent(const int globIdx, const int idx, const int comp){
			//due to memory pattern just alias
			return PopComponent(globIdx, idx, comp);
		}
		inline __device__ evalType& RangeFitness(const int globIdx, const int idx){
			return PopFitness(globIdx, idx);
		}

		//only from CPU now
		void SetLimits(vectorType *lo, vectorType *hi){
			for(int i=0; i < dim; i++){
				if(lo[i] == hi[i]) EXIT0("SettingLimits: %d-th lower limit same as upper limit. That is forbidden. Candidate \
					coordinates are in range [lower,upper)", i)
			}
			cudaMemcpy(upperLimit,hi,sizeof(vectorType)*dim,cudaMemcpyHostToDevice);
			cudaMemcpy(lowerLimit,lo,sizeof(vectorType)*dim,cudaMemcpyHostToDevice);
		}
		inline __device__ void MoveToPop(const int globIdx, const int popIdx, const int offsprIdx){
			for(int i=0; i < dim; i++){
				PopComponent(globIdx,popIdx,i) = OffsprComponent(globIdx,offsprIdx,i);
			}
			PopFitness(globIdx,popIdx) = OffsprFitness(globIdx,offsprIdx);
		}
#else
		inline vectorType& PopComponent(const int cand, const int comp){
			//row-major!
			return popComponent[dim*cand + comp];
		}
		inline vectorType& OffsprComponent(const int cand, const int comp){
			//row-major!
			return offsprComponent[dim*cand + comp];
		}
		inline evalType& PopFitness(const int cand){
			return popFitness[cand];
		}
		inline evalType& OffsprFitness(const int cand){
			return offsprFitness[cand];
		}
		//for evaluation
		inline vectorType& RangeComponent(const int idx, const int comp){
			//row-major!
			return popComponent[dim*idx + comp];
		}
		inline evalType& RangeFitness(const int idx){
			return popFitness[idx];
		}
		
		//only from CPU now
		void SetLimits(vectorType *lo, vectorType *hi){
			memcpy(upperLimit,hi,sizeof(vectorType)*dim);
			memcpy(lowerLimit,lo,sizeof(vectorType)*dim);
		}
		inline void MoveToPop(const int popIdx, const int offsprIdx){
			for(int i=0; i < dim; i++){
				PopComponent(popIdx,i) = OffsprComponent(offsprIdx,i);
			}
			PopFitness(popIdx) = OffsprFitness(offsprIdx);
		}
#endif
};


#endif