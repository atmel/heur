#ifndef __HEURISTICS_RANGE__
#define __HEURISTICS_RANGE__

/* Range struct:
	when looping across the range, the first element is in range, the last is not (to be used in for cycle),
	therefore, valid elements lay in [lo,hi)
*/
typedef struct _range {
	int lo;
	int hi;
	int length;
} range;

__device__ __host__ inline range makeRange(int lo, int hi){
	range tmp;
	tmp.lo = lo;
	tmp.hi = hi;
	tmp.length = hi-lo;
	return tmp;
}

#endif