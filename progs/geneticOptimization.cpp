#include "heuristics.h"
#include <iostream>

#define DIM 10
#define POPS 1

typedef basicCandidateContainer<int,int> RScandCont;
typedef basicArchive<RScandCont,int,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,64,128,POPS);
	RSbasArch *ac = new RSbasArch(cc,1000);
	int lo[]={-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000}, 
		hi[]={10001,10001,10001,10001,10001,10001,10001,10001,10001,10001};
	if(!cc->SetLimits(lo,hi)) return 0;
	
	timer t;
	
	t.Start();
	t.Stop();
	std::cout << "Time between calls to timer: " << t.PrintElapsedTime() << "\n";
  
	//new population
	basicPopulation<RScandCont,RSbasArch> *RSpop = new basicPopulation<RScandCont,RSbasArch>(cc,ac);
	bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int> *arch = 
		new bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,int,int>("testPopGO");

	//initialize
	RSpop->AddInitialization((new popRangedMasterMethod<RScandCont>())
							->Add(new pseudouniformRandomInitialization<RScandCont>())
							//->Add(new periodicPertubation<RScandCont>())  //crossover cannot break limits
							->Add(new sphericFunction<RScandCont>())
							);
	//reproduction
	RSpop->AddExecution((new reproductionMethod<RScandCont>(0.9))
							->Add(new twoTournamentSelection<RScandCont>())
							->Add(new onePointCrossover<RScandCont>())
							);
	//mutation
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add((new mutationWrapper<RScandCont>(0.9,0.5))
								->Add(new gaussianNoiseMutation<RScandCont,probabilisticRounding<int> >(50.0))
              					->Add(new periodicPertubation<RScandCont>())) // we need to pertube only after mutation!
							);
							
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add(new sphericFunction<RScandCont>())
							);
							
				  
	RSpop->AddExecution(new replaceMerging<RScandCont,int,int>());

	RSpop->AddExecution((new popRangedArchivedMasterMethod<RScandCont,RSbasArch>())
							->Add(arch)
						);
	std::cout << "initializing\n";
	t.Start();
	  if(!RSpop->Init()){ 
		std::cout << "init UNsuccessfull\n";
		return 0;
	  }
	t.Stop();
	std::cout << "---- Time of initialization: " << t.PrintElapsedTime() << "\n";
	
	t.Start();
	for(int i=0;i<100;i++){
	  //std::cout << i << "-th generation\n";
	  if(!RSpop->NextGeneration()){
		std::cout << "Error during Next Generation\n";
		return 0;
	  }
	}
	t.Stop();
	std::cout << "---- Time of run: " << t.PrintElapsedTime() << "\n";
	
	cout << "Saving to file\n";
	arch->SaveToFile();

return 0;
}