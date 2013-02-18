#include "heuristics.h"
#include <iostream>
#include <cstdlib>

#define DIM 5
#define POPS 1

typedef int vType;

typedef basicCandidateContainer<vType,int> RScandCont;
typedef basicArchive<RScandCont,vType,int> RSbasArch;

int main(int argc, char* argv[]){

	//decode cmd options: 1. popSize, 2. pops per kernel
	
	if(argc != 3) return -1;
	int pops = atoi(argv[2]), pSize = atoi(argv[1]);
	//cout << "popSize: " << pSize << "popsPerKernel: " << pops << "\n";
	
	timer t;
	RScandCont *cc = new RScandCont(DIM,pSize,pSize,pops);
	RSbasArch *ac = new RSbasArch(cc,100);
	int lo[]={-10000,-10000,-10000,-10000,-10000}, hi[] = {10001,10001,10001,10001,10001};
	/*vType lo[]={
	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1,

	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1,

	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1,
	1,1,1, 1,1,1, 1,1,1
	}, hi[] = {
	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10,

	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10,

	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10,
	10,10,10, 10,10,10, 10,10,10
	};*/
	if(!cc->SetLimits(lo,hi)) return 0;
	//new population
	basicPopulation<RScandCont,RSbasArch> *RSpop = new basicPopulation<RScandCont,RSbasArch>(cc,ac);
	bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,vType,int> *arch = 
		new bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,vType,int>("testPopSA");

	RSpop->AddInitialization((new popRangedMasterMethod<RScandCont>())
							->Add(new pseudouniformRandomInitialization<RScandCont>())
							->Add(new sphericFunction<RScandCont>())//sudokuEvaluation<RScandCont>())
							);
	RSpop->AddExecution((new popToOffsprRangedMasterMethod<RScandCont>())
							->Add(new copyReproduction<RScandCont>())
							);
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add(new gaussianNoiseMutation<RScandCont,probabilisticRounding<vType> >(50)) //cooledGaussianNoiseMutation<RScandCont,probabilisticRounding<vType>,classicCooling>(400))
							->Add(new periodicPertubation<RScandCont>())
							->Add(new sphericFunction<RScandCont>())//sudokuEvaluation<RScandCont>())
							);
	RSpop->AddExecution((new offsprToPopRangedMasterMethod<RScandCont>())
							->Add(new annealedMerging<RScandCont,classicCooling>(1000000))
							);
	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	//std::cout << "initializing\n";
	if(!RSpop->Init()){ 
	  std::cout << "init UNsuccessfull\n";
	  return 0;
	}
	t.Start();
	for(int i=0;i<100;i++){
	  //std::cout << i << "-th generation\n";
	  if(!RSpop->NextGeneration()){
		std::cout << "Error during Next Generation\n";
		return 0;
	  }
	}
	#if USE_CUDA
	cudaDeviceSynchronize();
	#endif
	t.Stop();
	std::cout << t.PrintElapsedTimeRaw() << "\n";
	  
	//cout << "Saving to file\n";
	//arch->SaveToFile();

return 0;
}