#include "heuristics.h"
#include <iostream>
#include <cstdlib>

#define DIM 81
#define POPS 1

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<RScandCont,int,int> RSbasArch;

int main(int argc, char* argv[]){

	//decode cmd options: 1. popSize, 2. pops per kernel
	
	if(argc != 3) return -1;
	int pops = atoi(argv[2]), pSize = atoi(argv[1]);
	//cout << "popSize: " << pSize << "popsPerKernel: " << pops << "\n";
	
	timer t;
	RScandCont *cc = new RScandCont(DIM,pSize,0,pops);
	RSbasArch *ac = new RSbasArch(cc,100);
	//int lo[]={-10,-100,-1000,-10000,-10000}, hi[] = {11,101,1001,10001,10001};
	int lo[]={
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
	};
	if(!cc->SetLimits(lo,hi)) return 0;
	//new population
	basicPopulation<RScandCont,RSbasArch> *RSpop = new basicPopulation<RScandCont,RSbasArch>(cc,ac);
	bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int> *arch = 
		new bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int>("testPop");

	/*RSpop->AddInitialization(new popRangedMasterMethod<popContainer>()
								->Add(new pseudouniformRandomInitialization<popContainer>)
								->Add(new someEvaluation.......)
							);*/
	RSpop->AddExecution((new popRangedMasterMethod<RScandCont>())
							->Add(new pseudouniformRandomInitialization<RScandCont>())
							->Add(new sudokuEvaluation<RScandCont>())
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
	cudaDeviceSynchronize();
	t.Stop();
	std::cout << t.PrintElapsedTimeRaw() << "\n";
	  
	//cout << "Saving to file\n";
	//arch->SaveToFile();

return 0;
}