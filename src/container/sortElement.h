#ifndef __HEURISTICS_SORT_ELEMENT__
#define __HEURISTICS_SORT_ELEMENT__

//structure for sorting passed to std algorithm
template <typename evalType>
struct indexElement{
	evalType fit;
	int ind;

	bool operator<(const indexElement<evalType> &lhs, const indexElement<evalType> &rhs){
		return lhs.fit < rhs.fit;
	}
};

#endif