#include "heuristics.h"
#include <iostream>

#define DIM 81
#define POPS 1

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<RScandCont,int,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,512,0,POPS);
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
	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100,

	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100,

	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100,
	100,100,100, 100,100,100, 100,100,100
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
							//->Add(new periodicPertubation<RScandCont>())      // init respects limits!
							->Add(new evaluationSodoku<RScandCont>())
							);
	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	std::cout << "initializing\n";
	if(!RSpop->Init()){ 
	  std::cout << "init UNsuccessfull\n";
	  return 0;
	}
	for(int i=0;i<3;i++){
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