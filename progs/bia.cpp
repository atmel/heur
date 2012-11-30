#include <iostream>

#include "heuristics.h"
#include <fstream>
#include <string>

//#ifndef _WIN32
//#include<cuda.h>
//#include<cuda_runtime_api.h>
//#endif

using namespace std;

//typedef inverseHilbMtrxNorm<1,int,1,int> testComb;
/*template<int dim, typename vectorType, typename evalType>
void SaveToFile(basicSingleObjPopulation<dim,vectorType,evalType> *pop,LARGE_INTEGER *start, LARGE_INTEGER *stop, string filename, int gen){
	LARGE_INTEGER frq;

	ofstream ofile(filename + "_avg.txt");
	ofile.precision(2);
	for(int i=0;i<gen;i++) ofile << fixed << pop->GetAvgFitness(i) << ", ";
	//time
	ofile << 1000*(stop->QuadPart-start->QuadPart)/frq.QuadPart << " ";
	ofile.close();

	ofile.open(filename + "_best.txt");
	ofile.precision(2);
	for(int i=0;i<gen;i++) ofile << fixed << pop->GetBestFitness(i) << ", ";
	ofile.close();
}*/

//specify sigma of normal distribution and mutation rate in percent
template<int dim, typename vectorType>						//20,30,5
class mygaussianDisplace: public gaussianDisplace<dim,vectorType,20,30,5>{};

//specify mating pool size and type of selection and parasitism rate in percent
template<int dim, typename vectorType, typename evalType>
class mybiaReproduction: public biaReproduction<dim,vectorType,evalType,3000,tournamentSelection,10,20>{};

//template<int dim, typename vectorType, typename evalType>
//class myPlainCopyReproduction: public plainCopyReproduction<dim,vectorType,evalType,0>{};

//specify number of elite candiates that will propagate to next population
template<int dim, typename vectorType, typename evalType>
class myelitismMerge : public elitismMerge<dim,vectorType,evalType,3> {};

int main(int argc, char* argv[]){

	cout << "Hello World!\n";
	//cout << "500 over 1:" << testComb::comb(500,1) << '\n';
	//cout << "500 over 499:" << testComb::comb(500,499) << '\n';
	//cout << "6 over 3:" << testComb::comb(6,3) << '\n';
	//cout << "6 over 2:" << testComb::comb(6,2) << '\n';
	//cout << "6 over 4:" << testComb::comb(6,4) << '\n';
	//cout << "lrnd:" << rnd::lrand() << "\n\n";


	//inverseHilbMtrxNorm<6,int,1,int>::Init(NULL);
	
#define G 1000
	testingClassicGOPopulation<98,signed,double,moleculePotentialEnergy,mygaussianDisplace,myelitismMerge,mybiaReproduction> 
		pop(60,1500,0);

	cout << "Creating pop\n";
	//LARGE_INTEGER frq,start,stop;
	//QueryPerformanceFrequency(&frq);

	cout << "Starting\n";
	char b = 'o';
	vector<int> converged;
	vector<long> time;

	//ofstream stats("22stats.txt");
	//stats.precision(2);
	for(int i=0;i<100;i++){
		pop.Create();
		//QueryPerformanceCounter(&start);
		while(1){
			pop.NextGeneration();
		}
		//QueryPerformanceCounter(&stop);
		//stats << 1000*(stop.QuadPart-start.QuadPart)/frq.QuadPart <<" "<< pop.GetGeneration() <<" "<< fixed << pop.GetBestFitness(pop.GetGeneration()-1)<<"\n";
		//cout << "time: "<< 1000*(stop.QuadPart-start.QuadPart)/frq.QuadPart;
		//cout << ", gen: "<< pop.GetGeneration() << "\n";
		cin>>b;
	}
	//stats.close();
	//SaveToFile<98,signed,double>(&pop,&start,&stop,"run21crossOnly",G);
	cin >> b;

	/*pop.Create();
	QueryPerformanceCounter(&start);
	for(int i=0;i<G;i++) pop.NextGeneration();
	QueryPerformanceCounter(&stop);
	SaveToFile<98,signed,double>(&pop,&start,&stop,"run22crossOnly",G);
	
	pop.Create();
	QueryPerformanceCounter(&start);
	for(int i=0;i<G;i++) pop.NextGeneration();
	QueryPerformanceCounter(&stop);
	SaveToFile<98,signed,double>(&pop,&start,&stop,"run23crossOnly",G);

	pop.Create();
	QueryPerformanceCounter(&start);
	for(int i=0;i<G;i++) pop.NextGeneration();
	QueryPerformanceCounter(&stop);
	SaveToFile<98,signed,double>(&pop,&start,&stop,"run24crossOnly",G);

	pop.Create();
	QueryPerformanceCounter(&start);
	for(int i=0;i<G;i++) pop.NextGeneration();
	QueryPerformanceCounter(&stop);
	SaveToFile<98,signed,double>(&pop,&start,&stop,"run25crossOnly",G);
*/
	

/*#ifndef _WIN32
	cudaDeviceProp dev;

	cout << "cuda code compiled\n";
	
	int cnt;
	 cudaGetDeviceCount(&cnt);

	for(int i=0; i < cnt; i++){
		cudaGetDeviceProperties(&dev,i);
		cout << "device "<< i << ": \n";
		cout << "totalGlobalMem: " << dev.totalGlobalMem << "\n";
        cout << "sharedMemPerBlock: " << dev.sharedMemPerBlock << "\n";
        cout << "regsPerBlock: " << dev.regsPerBlock << "\n";
        cout << "warpSize: " << dev.warpSize << "\n";
        cout << "memPitch: " << dev.memPitch << "\n";
        cout << "maxThreadsPerBlock: " << dev.maxThreadsPerBlock << "\n";
        cout << "clockRate: " << dev.clockRate << "\n";
        cout << "totalConstMem: " << dev.totalConstMem << "\n";
        cout << "compute capability: " << dev.major <<"."<< dev.minor <<'\n';
        cout << "deviceOverlap: " << dev.deviceOverlap << "\n";
        cout << "multiProcessorCount: " << dev.multiProcessorCount << "\n";
        cout << "kernelExecTimeoutEnabled: " << dev.kernelExecTimeoutEnabled << "\n";
        cout << "canMapHostMemory: " << dev.canMapHostMemory << "\n";
        cout << "computeMode: " << dev.computeMode << "\n";
        cout << "concurrentKernels: " << dev.concurrentKernels << "\n";
        //cout << "asyncEngineCount: " << dev.asyncEngineCount << "\n";
        //cout << "memoryClockRate: " << dev.memoryClockRate << "\n";
        //cout << "l2CacheSize: " << dev.l2CacheSize << "\n";
        //cout << "maxThreadsPerMultiProcessor: " << dev.maxThreadsPerMultiProcessor << "\n\n\n";

	}

#endif*/



	return 0;
}
