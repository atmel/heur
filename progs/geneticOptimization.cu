#include "heuristics.h"
#include <iostream>

#define DIM 81
#define POPS 1

typedef char vType;

typedef basicCandidateContainer<vType,int> RScandCont;
typedef basicArchive<RScandCont,vType,int> RSbasArch;

int main(void){

	RScandCont *cc = new RScandCont(DIM,512,1024,POPS);
	RSbasArch *ac = new RSbasArch(cc,1000);
	//int lo[]={-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000,-10000}, 
	//	hi[]={10001,10001,10001,10001,10001,10001,10001,10001,10001,10001};
	vType lo[]={
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
	
	timer t;
	
	t.Start();
	t.Stop();
	std::cout << "Time between calls to timer: " << t.PrintElapsedTime() << "\n";
  
	//new population
	basicPopulation<RScandCont,RSbasArch> *RSpop = new basicPopulation<RScandCont,RSbasArch>(cc,ac);
	bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,vType,int> *arch = 
		new bestCandArchivedFitnessArchivationMethod<RScandCont,RSbasArch,vType,int>("testPopGO");

	//initialize
	RSpop->AddInitialization((new popRangedMasterMethod<RScandCont>())
							->Add(new pseudouniformRandomInitialization<RScandCont>())
							//->Add(new periodicPertubation<RScandCont>())  //crossover cannot break limits
							->Add(new sphericFunction<RScandCont>())
							);
	//reproduction
	RSpop->AddExecution((new reproductionMethod<RScandCont>(0.3))
							->Add(new twoTournamentSelection<RScandCont>())
							->Add(new onePointCrossover<RScandCont>())
							);
	//mutation
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add((new mutationWrapper<RScandCont>(0.3,0.9))
								->Add(new gaussianNoiseMutation<RScandCont,probabilisticRounding<int> >(1))
              					->Add(new periodicPertubation<RScandCont>())) // we need to pertube only after mutation!
							);
							
	RSpop->AddExecution((new offsprRangedMasterMethod<RScandCont>())
							->Add(new sphericFunction<RScandCont>())//sudokuEvaluation<RScandCont>())
							);
							
				  
	RSpop->AddExecution(new replaceMerging<RScandCont,vType,int>());

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
	for(int i=0;i<1000;i++){
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
	cudaDeviceReset();

return 0;
}