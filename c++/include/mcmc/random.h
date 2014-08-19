#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

#include <cstdlib>

#include <unordered_set>
#include <vector>

#include "mcmc/exception.h"

namespace mcmc {

class Random {
public:
	int randint(int from, int upto) {
#if 0
		throw UnimplementedException("Random.randint");
		return -1;
#else
		return (rand() % (upto - from)) + from;
#endif
	}


protected:
	void sample(std::unordered_set<int> *accu, int from, int upto, ::size_t count) {
		for (::size_t i = 0; i < count; i++) {
			int r = randint(from, upto);
			if (accu->find(r) == accu->end()) {
				accu->insert(r);
			} else {
				i--;
			}
		}
	}


public:
	template <class List>
	List *sample(const List &list, ::size_t count) {
		List *result = new List();

#if 1
		std::unordered_set<int> accu;
		sample(&accu, 0, list.size(), count);

		::size_t c = 0;
		for (typename List::const_iterator i = list.begin(); i != list.end(); i++) {
			if (accu.find(c) != accu.end()) {
				result->insert(*i);
			}
			c++;
		}

#else
		throw UnimplementedException("Random.sample");
#endif

		return result;
	}


	template <class List>
	List *sample(const List *list, ::size_t count) {
		return sample(*list, count);
	}


	template <class Element>
	std::vector<Element> *sample(const std::vector<Element> &list, ::size_t count) {
#if 1
		std::unordered_set<int> accu;
		sample(&accu, 0, list.size(), count);

		std::vector<Element> *result = new std::vector<Element>(accu.size());

		for (std::unordered_set<int>::const_iterator i = accu.begin(); i != accu.end(); i++) {
			result->push_back(list[*i]);
		}

#else
		throw UnimplementedException("Random.sample");
#endif

		return result;
	}


	void gamma(std::vector<std::vector<double> > *a,
			   double p1, double p2, ::size_t n1, ::size_t n2) {
#if 1
#else
		throw UnimplementedException("Random.gamma");
#endif
	}

	static Random random;
};

}	// namespace mcmc

#endif	// ndef MCMC_RANDOM_H__
