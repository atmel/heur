#ifndef __HEURISTICS_EVAL_SUDOKU__
#define __HEURISTICS_EVAL_SUDOKU__

#include "heuristics.h"

/*
	There shouls be enough shared memory for 512 threads (84 bytes each)
*/

#if USE_CUDA
template<class popContainer>
__global__ void SudokuEvalKernel(popContainer pop, range rng, int* squares){
	//setup shared
	extern __shared__ int dynamic[];
	char* cand = reinterpret_cast<char*>(&dynamic[84/sizeof(int)*threadIdx.x]);
	int id = threadIdx.x + rng.lo;
	int fit, num, ix; //checked number and universal index

	for(int i=0; i < REQUIRED_RUNS(rng.length); i++, id += blockDim.x){
		if(id >= rng.hi) return; //complete

		//load to shared
		for(int s=0; s< pop.GetDim(); s++){
			//printf("%d ",s);
			cand[s] = pop.RangeComponent(blockIdx.x,id,s);
			//if(threadIdx.x == 0) printf("%d, ", cand[i]);
		}
		//if(threadIdx.x == 0) printf("\n");
		__syncthreads();

		for(int j=0;j< pop.GetDim();j++){
			num = cand[j];
			//rows conflict check
			ix = j/9;
			for(int k=0;k<9;k++){
				fit += (cand[9*ix+k] == num);
			}
			//column conflict check
			ix = j % 9;
			for(int k=0;k<9;k++){
				fit += (cand[ix+9*k] == num);
			}
			//square conflict check
			ix = squares[j];
			for(int k=0;k<3;k++){
				for(int l=0;l<3;l++){
					fit += (cand[ix + 9*k + l] == num);
				}
			}
		}
		pop.RangeFitness(blockIdx.x,id) = fit-27;
	}
}
#endif

/*
	Candidate coordiantes can be [1,9]
*/

template<class popContainer>
class sudokuEvaluation: public slaveMethod<popContainer>{
protected:
	//indices where corresponding small squares start
	static const int squares[81]; 
	int *gSquares;
	int threadCount;


public:
	sudokuEvaluation<popContainer>(){
		#if USE_CUDA
		//move constatns to GPU
		cudaMalloc(&gSquares,sizeof(int)*81);
		cudaMemcpy(gSquares,squares,sizeof(int)*81, cudaMemcpyHostToDevice);
		#endif
	}

	~sudokuEvaluation<popContainer>(){
		#if USE_CUDA
		cudaFree(gSquares);
		#endif
	}

	int Init(generalInfoProvider *p){
		//init slave
		if(!slaveMethod<popContainer>::Init(p)) EXIT0("sudoku evaluation: slave init unsuccessfull")
		threadCount = std::min(512,this->workingRange.length);
		if(this->pop->GetDim() != 81)  EXIT0("sudoku evaluation: candidate does not have 81 dimensions")
		return 1;
	}
	int Perform(){
	#if USE_CUDA
		CUDA_CALL("sudoku eval",(SudokuEvalKernel<popContainer><<<this->pop->GetPopsPerKernel(), threadCount, threadCount*84>>>
			(*(this->pop),this->workingRange,gSquares)))
	#else
		for(int i=this->workingRange.lo; i< this->workingRange.hi;i++){
			int fit = 0;
			int ri,ci,si; //row index, column index, square beginning index
			int num;
			// every filed
			for(int j=0;j < this->pop->GetDim();j++){
				num = this->pop->RangeComponent(i,j);
				//rows conflict check
				ri = j/9;
				for(int k=0;k<9;k++){
					fit += (this->pop->RangeComponent(i, 9*ri+k) == num);
				}
				//column conflict check
				ci = j % 9;
				for(int k=0;k<9;k++){
					fit += (this->pop->RangeComponent(i, ci+9*k) == num);
				}
				//square conflict check
				si = squares[j];
				for(int k=0;k<3;k++){
					for(int l=0;l<3;l++){
						fit += (this->pop->RangeComponent(i, si + 9*k + l) == num);
					}
				}
			}
			//there are 27 conflicts wit itself
			this->pop->RangeFitness(i) = fit - 27;
		}
	#endif
		return 1;
	}
};


template<class popContainer>
const int sudokuEvaluation<popContainer>::squares[] = {
0,0,0,	  3,3,3,	6,6,6,
0,0,0,	  3,3,3,	6,6,6,
0,0,0,	  3,3,3,	6,6,6,
27,27,27, 30,30,30, 33,33,33,
27,27,27, 30,30,30, 33,33,33,
27,27,27, 30,30,30, 33,33,33,
54,54,54, 57,57,57, 60,60,60,
54,54,54, 57,57,57, 60,60,60,
54,54,54, 57,57,57, 60,60,60
}; 


#endif