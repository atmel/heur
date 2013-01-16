#ifndef __HEURISTICS_ARCHIVE__
#define __HEURISTICS_ARCHIVE__

#include "heuristics.h"
/*
better memory handling, portability to GPU with advantage of straightforward syntax
and good memory acces pattern "shielded" from a user.
*/

/*
	Class for storing statistical data or whatever we want to save about population state
	It is supposed to be used for general diagnostics, or as a base for convergence criterion
*/

/* Basic archive is minimalistic, it holds best fitness history and best candidate found so far 
	(one for each population on GPU)
	Results are saved to properly named file, saving mode can be chosen as "at once" or continuous 
	(after each generation) - this should be used only for debugging (crash will not destroy all the data) 
	as it probably will be particularly slow on GPU
	For purposes of use by other methods, this object is also passed by value to GPU.
*/

template<typename vectorType, typename evalType>
class basicArchive{
protected:
//this pointers points either to RAM or GPURAM  
	// stored in pattern pop1cand1, pop1cand2,..., pop2cand1, pop2cand2,...
	vectorType *bestCand;

	// bestPopFitnessArchive storesd as (makes sense only on GPU): pop1_fit1, pop2_fit1, ... , popN_fit1, pop1_fit2, ...
	evalType *bestFitness; //, *bestPopFitnessArchive;
	const int dim, maxGeneration, popsPerKernel;

public:
		basicArchive(int _dim, int _maxGeneration, int _popsPerKernel=1):
			dim(_dim), maxGeneration(_maxGeneration), popsPerKernel(_popsPerKernel){
#if USE_CUDA
			D("bestCand size is %d",(int)sizeof(vectorType)*dim*popsPerKernel*maxGeneration)
			//allocate multiple populations, offsprings, compute strides
			cudaMalloc(&bestCand, sizeof(vectorType)*dim*popsPerKernel*maxGeneration);
			cudaMalloc(&bestFitness, sizeof(evalType)*popsPerKernel*maxGeneration);
			//cudaMalloc(&bestPopFitnessArchive, sizeof(evalType)*maxGeneration*popsPerKernel);
			//cudaMemset(bestCand,0,sizeof(vectorType)*dim*popsPerKernel*maxGeneration);
#else
			bestCand = new vectorType[dim*maxGeneration];
			bestFitness = new evalType[maxGeneration];
			//bestPopFitnessArchive = new evalType[maxGeneration];
#endif
		}
		
		// propper dealocation
		int Finalize(){
#if USE_CUDA 
			cudaFree(bestCand);
			cudaFree(bestFitness);
			//cudaFree(bestPopFitnessArchive);
#else
			delete [] bestCand;
			delete [] bestFitness;
			//delete [] bestPopFitnessArchive;
#endif	
			return 1;
		} 
		
		//DO NOT DEALOCATE IN DTOR -- OBJECT PASSED BY VALUE!!
		~basicArchive(){}
		
		//For kernell to write values
#if USE_CUDA
		//Getters
		inline __device__ vectorType& BestCandComponent(const int globIdx, const int idx, const int comp){
			return bestCand[dim*popsPerKernel*idx + dim*globIdx + comp];}
		inline __device__ evalType& BestPopFitness(const int globIdx, const int idx){
			return bestFitness[globIdx + popsPerKernel*idx];}
		//Setters
		/*
		inline __device__ void BestCandComponent(const int globIdx, const int idx, const int comp, const vectorType val){
			bestCand[dim*popsPerKernel*idx + dim*globIdx + comp] = val;}
		inline __device__ void BestPopFitness(const int globIdx, const int idx, const vectorType val){
			bestFitness[globIdx + popsPerKernel*idx] = val;}*/
			
		//inline __device__ evalType& BestFitnessArchive(const int globIdx, const int idx){return bestPopFitnessArchive[globIdx*popsPerKernel + idx];}
		/* convenience method for copying to RAM
			Copies bestCand to RAM and fitnessArchive, beginning with startIdx ending with endIdx-1
		*/
		void CopyToCPU(vectorType* bestCandCPU, evalType* bestFitnessCPU, int startIdx, int endIdx){
			cudaMemcpy(bestCandCPU, &bestCand[startIdx*dim*popsPerKernel], 
				(endIdx-startIdx)*dim*sizeof(vectorType)*popsPerKernel,cudaMemcpyDeviceToHost);

			cudaMemcpy(bestFitnessCPU, &bestFitness[startIdx*popsPerKernel], 
				(endIdx-startIdx)*sizeof(evalType)*popsPerKernel,cudaMemcpyDeviceToHost); 
			/*
			cudaMemcpy(bestCandCPU, bestCand, 
				dim*sizeof(vectorType),cudaMemcpyDeviceToHost);

			cudaMemcpy(bestFitnessCPU, bestFitness, 
				sizeof(evalType),cudaMemcpyDeviceToHost);*/	
		}
#else
		inline vectorType& BestCandComponent(const int idx, const int comp){return bestCand[dim*idx + comp];}
		inline evalType& BestPopFitness(const int idx){return bestFitness[idx];}
		//inline evalType& BestFitnessArchive(const int idx){return bestPopFitnessArchive[idx];}
#endif
		inline __device__ __host__ int GetMaxGeneration() const {return maxGeneration;}
};


/* Contains structure for holding snapshots of whole population
	Use carefully snapshor archive can get large easily
*//*
template<int dim, typename vectorType, typename evalType, int maxGeneration, int popsPerKernel=1>
class popSnapshotArchive : basicArchive<dim,vectorType,evalType>{
	protected:
	 Contains populations as they are stored in memory
		In case of GPU it contains just snapshots of whole popComponent array
		regardless of popsPerKernel (it stores all at once)
	
	vectorType *popSnapshots;
	const int popSize;

	public:
	popSnapshotArchive<dim,vectorType,evalType>(const int _popSize): 
		basicArchive<dim,vectorType,evalType,maxGeneration,popsPerKernel>(),popSize(_popsize){
#if USE_CUDA
			cudaMalloc(&popSnapshots, sizeof(vectorType)*popsPerKernel*dim*popSize*maxGeneration);
#else
			popSnapshots = new vectorType[dim*popSize*maxGeneration]
#endif
	}

	int Finalize(){
		basicArchive<dim,vectorType,evalType,maxGeneration,popsPerKernel>::Finalize();
#if USE_CUDA 
		cudaFree(popSnapshots);
#else
		delete [] popSnapshots;
#endif
	} 

#if USE_CUDA
		inline __device__ vectorType* GetPopArchivesArray(const int idx){return &popSnapshots[dim*popSize*popsPerKernel*idx];}
#else
		inline vectorType* GetPopArchivesArray(const int idx){return &popSnapshots[dim*popSize*idx];}
#endif

};*/


#endif