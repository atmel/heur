#include "heuristics.h"

#define DIM 2

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<int,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,1000,0,5);
	RSbasArch *ac = new RSbasArch(DIM,1000,5);
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
							->Add(new replaceMerging())
						);
	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	cout << "initializing\n";
	if(!RSpop->Init()){ 
	  cout << "init UNsuccessfull\n";
	  return 0;
	}
	for(int i=0;i<100;i++){
	  cout << i << "-th generation\n";
	  if(!RSpop->NextGeneration()){
		cout << "Error during Next Generation\n";
		return 0;
	  }
	}
	  
	cout << "Saving to file\n";
	arch->SaveToFile();

return 0;
}