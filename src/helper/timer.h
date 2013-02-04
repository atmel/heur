#ifndef __HEURISTICS_TIMER__
#define __HEURISTICS_TIMER__

/*
	Hi-res measurement suggested at http://www.guyrutenberg.com/2007/09/22/profiling-code-using-clock_gettime/
*/

#include "heuristics.h" 
#include <time.h>
#include <string>
#include <cstdio>

class timer{
	timespec start, end;
	clockid_t id;

public:
	timer(clockid_t _id = CLOCK_PROCESS_CPUTIME_ID):id(_id){};
	
	void Start(){
		clock_gettime(id, &start);
	}

	void Stop(){
		clock_gettime(id, &end);
	}

	//return elapsed nanoseconds
	timespec GetElapsedTime(){
		timespec temp;
		//we will not use negative times!!
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		return temp;
	}
	std::string PrintElapsedTime(){
	  char s[400];
	  timespec t = GetElapsedTime();
	  sprintf(s,"%lds, %ldns", (long int)t.tv_sec, (long int)t.tv_nsec);
	  return std::string(s);
	}
	
	std::string PrintElapsedTimeRaw(){
	  char s[400];
	  timespec t = GetElapsedTime();
	  sprintf(s,"%ld, %ld", (long int)t.tv_sec, (long int)t.tv_nsec);
	  return std::string(s);
	}
};


#endif