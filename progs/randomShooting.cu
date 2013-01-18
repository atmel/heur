#include "heuristics.h"
#include <iostream>

#define DIM 2

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<int,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,16,0,3);
	RSbasArch *ac = new RSbasArch(DIM,1000,3);
	int lo[]={-10000,-10000}, hi[] = {10000,10000};
	cc->SetLimits(lo,hi);
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
							->Add(new periodicPertubation<RScandCont>())
							->Add(new sphericFunction<RScandCont>())
							->Add(new replaceMerging<RScandCont,int,int>())
							);
	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	std::cout << "initializing\n";
	if(!RSpop->Init()){ 
	  std::cout << "init UNsuccessfull\n";
	  return 0;
	}
	for(int i=0;i<1;i++){
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