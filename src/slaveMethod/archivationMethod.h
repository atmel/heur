#ifndef __VYZKUM_MUTATION__
#define __VYZKUM_MUTATION__

#include "heuristics.h"
#include <fstream>
#include <ios>
#include <typeinfo>

/* General rule for saving to files: save data so that pieces of complete information 
	is found consecultively in rows or columns.
*/

template<class popContainer, class archContainer>
class archivationMethod : public slaveMethod<popContainer>{
	protected:
	archContainer *arch;

	std::string filename;
	// under which index should we store next data (starts form 0)
	int currentIndex;
	// index, from which we should start saving to file (starts form 0 == nothing saved yet)
	int saveToFileIdx;

	public:
	archivationMethod<popContainer,archContainer>(std::string fname):
		currentIndex(0), saveToFileIdx(0), filename(fname){
	}

	int Init(generalInfoProvider *p){
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("Archivation method: slave init unsuccessfull")
		archiveProvider<archContainer> *ap = dynamic_cast<archiveProvider<archContainer>*>(p);
		if(ap == NULL) EXIT0("Archivation method: archive provider cast unsuccessfull")
		arch = ap->GetArchiveContainer();
		return 1;
	}

	virtual int Perform() = 0;
	//Save to file is to be called from outside
	virtual int SaveToFile() = 0;
};

//=========================================================================================================



#if USE_CUDA
/* Basically paralell reduction kernel with min function
	uses dynamically allocated shared mem array containing fitnesses and candidate indices

	lowerPow2size contains power of two closest to PopSize from below
	blockDim is expected to be = lowerPow2size/2:
		first, number of elements to reduce is reduced to lowerPow2size in first run by either minimization, of copy 
		(if there not enough elements to form pairs)
*/

template<class popContainer, class archive, typename evalType>
__global__ void UpdateBestCandFitKernel(popContainer pop, archive arch, const int lowerPow2size, range rng, int currentIndex){
	//contains array of fitnesses and ind array both beginning alligned to 8bytes (64bit) 
	extern __shared__ int dynamic[];
	const int allignedFitSize = ALLIGN_64((lowerPow2size)*sizeof(evalType));
	//const int allignedIndSize = ALLIGN_64((lowerPow2size)*sizeof(int));

	evalType *fit = reinterpret_cast<evalType*>(dynamic); //is is shared PER BLOCK! [blockIdx.x*(allignedFitSize + allignedIndSize)];
	int *ind = reinterpret_cast<int*> (&dynamic[allignedFitSize/sizeof(int)]);

	int id = threadIdx.x;
	//due to prescribed number of threads run exactly 2 times
	#pragma unroll 2
	for(int i=0; i < 2; i++, id += blockDim.x){
		bool first;
		if(id+lowerPow2size+rng.lo >= rng.hi){ // copy, no pair
			fit[id] = pop.RangeFitness(blockIdx.x,id + rng.lo);
			ind[id] = id + rng.lo;
		}else{ //find min and save it to shared
			first = pop.RangeFitness(blockIdx.x,id+rng.lo)<pop.RangeFitness(blockIdx.x,id+lowerPow2size+rng.lo) ? true:false;
			fit[id] = first ? pop.RangeFitness(blockIdx.x,id + rng.lo):pop.RangeFitness(blockIdx.x,id+lowerPow2size+rng.lo);
			ind[id] = first ? (id+rng.lo):(id+lowerPow2size+rng.lo);
		}
	}
	__syncthreads();
	// Do reduction
	ParalellReduceMin(fit,ind,lowerPow2size);

	//now, best fit is at the first position in array
	//update best candidate
	if(threadIdx.x < pop.GetDim()){
		arch.BestCandComponent(blockIdx.x,currentIndex,threadIdx.x) = pop.RangeComponent(blockIdx.x,ind[0],threadIdx.x);
	}
	//archive fitness
	if(threadIdx.x == 0){
		arch.BestPopFitness(blockIdx.x,currentIndex) = fit[0];
	}
}

//selects minimum from fitness values and stores it to fit1, performs the same action (based on fit condition) in idx array
template<typename evalType>
__device__ inline void SubordinateMin(evalType &fit1, const evalType &fit2, int &idx1, const int &idx2){
	const bool first = (fit1<fit2)?true:false;
	idx1 = first?idx1:idx2;
	fit1 = first?fit1:fit2;
}
	
/* Each thread first performs reduction to shared memory to size of power of two,
	therefore each thread performs min of two values from glob. mem, or pure copy from 
	glob mem.

	This function expects already array of power-of-two size, all pointers to shared memory
	It uses pyramidal reduction from Oberhuber's lectures, maximal size is 2048 ((corresponds to 4096 from globmem)/2)
*/	
template<typename evalType>
__device__ inline void ParalellReduceMin(evalType *fit, int *candIdx, const int pow2size){
	const int tid = threadIdx.x;
	//Pyramidal reduction
	if(pow2size==2048){ // all threads
		SubordinateMin(fit[tid],fit[tid+1024],candIdx[tid],candIdx[tid+1024]);
		__syncthreads();
	}
	if(pow2size>=1024){
		if(tid<512){
			SubordinateMin(fit[tid],fit[tid+512],candIdx[tid],candIdx[tid+512]);
		}
		__syncthreads();
	}
	if(pow2size>=512){
		if(tid<256){SubordinateMin(fit[tid],fit[tid+256],candIdx[tid],candIdx[tid+256]);}__syncthreads();}
	if(pow2size>=256){
		if(tid<128){SubordinateMin(fit[tid],fit[tid+128],candIdx[tid],candIdx[tid+128]);}__syncthreads();}
	if(pow2size>=128){
		if(tid<64){SubordinateMin(fit[tid],fit[tid+64],candIdx[tid],candIdx[tid+64]);}__syncthreads();}
	if(tid<32){//warp is synchronized implicitly
		if(pow2size>=64)SubordinateMin(fit[tid],fit[tid+32],candIdx[tid],candIdx[tid+32]);
		if(pow2size>=32)SubordinateMin(fit[tid],fit[tid+16],candIdx[tid],candIdx[tid+16]);
		if(pow2size>=16)SubordinateMin(fit[tid],fit[tid+8],candIdx[tid],candIdx[tid+8]);
		if(pow2size>=8)SubordinateMin(fit[tid],fit[tid+4],candIdx[tid],candIdx[tid+4]);
		if(pow2size>=4)SubordinateMin(fit[tid],fit[tid+2],candIdx[tid],candIdx[tid+2]);
		if(pow2size>=2)SubordinateMin(fit[tid],fit[tid+1],candIdx[tid],candIdx[tid+1]);
	}
}
#endif

template<class popContainer, class archContainer, typename vectorType, typename evalType>
class bestCandArchivedFitnessArchivationMethod : public archivationMethod<popContainer,archContainer>{

	public:
	bestCandArchivedFitnessArchivationMethod<popContainer,archContainer,vectorType,evalType>(std::string fname):
	  archivationMethod<popContainer,archContainer>(fname){};

	int Perform(){
		D("Archive update")
		//check free space
		if(this->currentIndex >= this->arch->GetMaxGeneration()) return 0;
#if USE_CUDA
		// find power of two strictly lower than popSize()
		/* popSize() is expected to be at least 4 => lowerPow2Size = 2 => 1 thread */
		int i = 0;
		for(int t=this->workingRange.length-1; t > 0; t = t >> 1,i++);
		int lowerPow2size = 1 << (i-1);
		const int allignedFitSize = ALLIGN_64((lowerPow2size)*sizeof(evalType));
		const int allignedIndSize = ALLIGN_64((lowerPow2size)*sizeof(int));
		D("Running archivation kernel with %d bytes of dynamic memory, %d threads",(allignedFitSize+allignedIndSize),lowerPow2size/2)
		CUDA_CALL("basic archivation",(UpdateBestCandFitKernel<popContainer,archContainer,evalType>
			<<<this->pop->GetPopsPerKernel(),lowerPow2size/2, (allignedFitSize+allignedIndSize)>>>
			(*(this->pop),*(this->arch),lowerPow2size,this->workingRange,this->currentIndex)))

#else
		//find minimal fitness, remember that candidate
		D("Archivation: finding minimum")
		int bestIdx = this->workingRange.lo;
		evalType best = this->pop->RangeFitness(this->workingRange.lo);
		
		for(int i = this->workingRange.lo + 1; i < this->workingRange.hi; i++){
			if(this->pop->RangeFitness(i) < best){
				bestIdx = i;
				best = this->pop->RangeFitness(i);
			}
		}
		D("Saving fitness, currIdx = %d, best = %d", this->currentIndex, best)
		//save best to array
		this->arch->BestPopFitness(this->currentIndex) = best;
		D("Saving Candidate")
		for(int i=0; i< this->pop->GetDim(); i++){
			this->arch->BestCandComponent(this->currentIndex,i) = this->pop->RangeComponent(bestIdx,i);
		}
		D("Finished saving")
#endif
		//increment current index
		this->currentIndex++;
		return 1;
	}


	int SaveToFile(){
		std::string fnBestCand = this->filename + "__BestCandidate.txt";
		std::string fnFitnessArchive = this->filename + "__BestFitness.txt";
		//open files
		ofstream fBestC(fnBestCand.c_str(),ios_base::out | ios_base::app);
		ofstream fBestF(fnFitnessArchive.c_str(),ios_base::out | ios_base::app);
		int generationsToSave = this->currentIndex- this->saveToFileIdx;
#if USE_CUDA
		//get data from gpu
		int candStride = this->pop->GetDim()*this->pop->GetPopsPerKernel();
		vectorType *candTmp = new vectorType[generationsToSave*candStride];
		evalType *fitTmp = new evalType[generationsToSave*this->pop->GetPopsPerKernel()];
		this->arch->CopyToCPU(candTmp,fitTmp,this->saveToFileIdx,this->currentIndex);
		//save them
		//candTmp[0] = candTmp[1] = 13;
		for(int i=0;i<generationsToSave;i++){
			//do not print letters when char...
			for(int j=0;j<candStride;j++){ 
			  if(typeid(vectorType) == typeid(char)){
				  fBestC << (int)(candTmp[j+i*candStride]) << ", ";
			  }else{
				  fBestC << candTmp[j+i*candStride] << ", ";
			  }
			}
			fBestC << "\n";
		}

		for(int i=0;i<generationsToSave;i++){
			for(int j=0;j < this->pop->GetPopsPerKernel();j++) fBestF << fitTmp[j+i*this->pop->GetPopsPerKernel()] << ", ";
			 fBestF << "\n";
		}
		delete [] candTmp;
		delete [] fitTmp;
#else
		for(int i= this->saveToFileIdx; i< this->currentIndex; i++){
			for(int j=0;j < this->pop->GetDim();j++) fBestC << this->arch->BestCandComponent(i,j) << ", ";
			fBestC << "\n";
		}

		for(int i=this->saveToFileIdx; i< this->currentIndex; i++) fBestF << this->arch->BestPopFitness(i) << ", \n";
		fBestF << "\n";
#endif
		//update what was saved
		this->saveToFileIdx = this->currentIndex;
		//close files
		fBestC.close();
		fBestF.close();
		return 1;
	}
};



/* diagnostic archive - archContainer whole populations snapshots + all things archived by 
	bestCandArchivedFitnessArchivationMethod
*/
//template<class popContainer, class archContainer, typename vectorType, typename evalType>
//class diagnosticArchivationMethod : public bestCandArchivedFitnessArchivationMethod<popContainer,archContainer,vectorType,evalType>{
//	using this->arch;
//	using this->pop;
//
//	public:
//	int Archive(){
//		//perform basic archivation, check free space
//		if(!bestCandArchivedFitnessArchivationMethod<popContainer,archContainer,vectorType,evalType>::Archive()){
//			currentIndex--;
//			return 0;
//		}
//		// decrement currentIndex for later use 
//		currentIndex--;
//
//#if USE_CUDA
//		cudaMemcpy(arch->GetPopArchivesArray(currentIndex),pop->GetPopComponentArray(),
//			pop->GetVectorTypeSize()*pop->GetDim()*pop->GetPopSize()*pop->GetPopsPerKernel(),cudaMemcpyDeviceToDevice);
//#else	
//		memcpy(arch->GetPopArchivesArray(currentIndex),pop->GetPopComponentArray(),
//			pop->GetVectorTypeSize()*pop->GetDim()*pop->GetPopSize());
//#endif
//		//increment current index
//		currentIndex++;
//	}
//
//	/* Snapshots form GPU are saved in format -- therefore same way as pops are hold in memory:
//	+generation1-------------+
//	|+pop1------------------+|
//	|| 1c1, 2c1, 3c1,...	||
//	|| 1c2, 2c2, 3c2,...	||
//	|| <--    PopSize	 -->||
//	|+----------------------+|
//	|+pop2------------------+|
//	|| ...					||
//	|+----------------------+|
//	| ...					 |
//	+------------------------+
//
//	for CPU: 
//	+generation1-------------+
//	|+pop1------------------+|
//	|| 1c1, 1c2, 1c3,...	||
//	|| 2c1, 2c2, 2c3,...	||
//	|| <--      dim 	 -->||
//	|+----------------------+|
//	+------------------------+
//	*/
//	int SaveToFile(){
//		// perform basic save to file
//		int tmpSaveIdx = saveToFileIdx;
//		bestCandArchivedFitnessArchivationMethod<popContainer,archContainer,vectorType,evalType>::SaveToFile();
//		saveToFileIdx = tmpSaveIdx;
//
//		std::string fnPopSnapshot = filename + "__PopSnapshots.txt";
//		//open files
//		ofstream fPopSnapshot(fnBestCand,ios::out | ios::app);
//		int generationsToSave = currentIndex-saveToFileIdx;
//#if USE_CUDA
//		//get data from gpu
//		vetorType *popSnapshot = new vectorType[pop->GetDim()*pop->GetPopsPerKernel()*pop->GetPopSize()*generationsToSave];
//		cudaMemcpy(popSnapshot,arch->GetPopArchivesArray(saveToFile),
//			pop->GetDim()*pop->GetPopsPerKernel()*pop->GetPopSize()*generationsToSave*sizeof(vectorType),cudaMemcpyDeviceToHost);
//		//save them
//		for(int i=0;i<generationsToSave*pop->GetPopsPerKernel()*pop->GetDim();i++){
//			for(int j=0;j<pop->GetPopSize();j++){
//				fPopSnapshot << popSnapshot[i*pop->GetPopSize() + j] << ", ";
//			}
//			fPopSnapshot << "\n";
//		}
//#else
//		for(int i=0;i<generationsToSave;i++){
//			for(int j=0;j<pop->GetPopSize();j++){
//				for(int k=0;k<pop->GetDim();k++){
//					fPopSnapshot << arch->GetPopArchivesArray(i)[j*pop->GetDim() + k] << ", ";
//				}
//				fPopSnapshot << "\n";
//			}
//		}
//#endif
//		//update what was saved
//		saveToFileIdx = currentIndex;
//		//close file
//		fPopSnapshot.close();
//		return 1;
//	}
//
//};

#endif