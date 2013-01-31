#include "heuristics.h"
#include <iostream>

#define DIM 5
#define POPS 1

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<RScandCont,int,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,64,128,POPS);
	RSbasArch *ac = new RSbasArch(cc,1000);
	int lo[]={-10,-100,-1000,-10000,-10000}, hi[] = {10,100,1000,10000,10000};
	cc->SetLimits(lo,hi);
	//new population
	basicPopulation<RScandCont,RSbasArch> *RSpop = new basicPopulation<RScandCont,RSbasArch>(cc,ac);
	bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int> *arch = 
		new bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int>("testPopGO");

	//initialize
	RSpop->AddInitialization((new popRangedMasterMethod<RScandCont>())
							->Add(new pseudouniformRandomInitialization<RScandCont>())
							->Add(new periodicPertubation<RScandCont>())
							->Add(new sphericFunction<RScandCont>())
							);
	//reproduction
	RSpop->AddExecution((new reproductionMethod<RScandCont>(0.9))
							->Add(new twoTournamentSelection<RScandCont>())
							->Add(new onePointCrossover<RScandCont>())
							);
	//mutation
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add((new mutationWrapper<RScandCont>(0.9,0.2))
								->Add(new gaussianNoiseMutation<RScandCont,probabilisticRounding<int> >(100.0)))
							);
							
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add(new periodicPertubation<RScandCont>())
							->Add(new sphericFunction<RScandCont>())
							);
							
				  
	RSpop->AddExecution(new replaceMerging<RScandCont,int,int>());

	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	std::cout << "initializing\n";
	if(!RSpop->Init()){ 
	  std::cout << "init UNsuccessfull\n";
	  return 0;
	}
	for(int i=0;i<10;i++){
	  std::cout << i << "-th generation\n";
	  if(!RSpop->NextGeneration()){
		std::cout << "Error during Next Generation\n";
		return 0;
	  }
	}
	  
	cout << "Saving to file\n";
	arch->SaveToFile();

return 0;
}