/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_DATA_H__
#define MCMC_DATA_H__

#include <unordered_set>
#include <utility>
#include <map>
#include <unordered_map>
#include <iostream>
#include <iomanip>

#include "mcmc/exception.h"

namespace mcmc {

class Edge {
public:
	Edge(int a, int b) : first(a), second(b) {
	}

	Edge(std::istream &s) {
		(void)get(s);
	}

	virtual ~Edge() {
	}

	virtual bool operator== (const Edge &a) const {
		return a.first == first && a.second == second;
	}

	virtual bool operator< (const Edge &a) const {
		return first < a.first || (first == a.first && second < a.second);
	}

	virtual std::ostream &put(std::ostream &s) const {
		s << "(" << first << ", " << second << ")";

		return s;
	}

protected:
	static char consume(std::istream &s, char expect) {
		char c;

		while (true) {
			// std::cerr << s.tellg() << " " << s.gcount() << std::endl;
			c = s.get();
			if (isspace(c)) {
				continue;
			}
			if (c != expect) {
				std::ostringstream os;
				os << "Expect " << expect << ", get '" << c << "'";
				throw MalformattedException(os.str());
			}

			return c;
		}
	}

public:
	virtual std::istream &get(std::istream &s) {
		// std::string line;
		// std::getline(s, line);
		// std::cerr << line << std::endl;

		consume(s, '(');
		s >> first;
		consume(s, ',');
		s >> second;
		consume(s, ')');

		return s;
	}

	int		first;
	int		second;
};


inline std::ostream &operator<< (std::ostream &s, const Edge &e) {
	return e.put(s);
}


inline std::istream &operator>> (std::istream &s, Edge &e) {
	return e.get(s);
}


typedef typename std::set<Edge> EdgeSet;
// typedef typename std::unordered_set<Edge> EdgeSet;

// typedef std::map<Edge, bool>			EdgeMap;
typedef std::unordered_map<Edge, bool>			EdgeMap;

}	// namespace mcmc


namespace std {
template<>
struct hash<mcmc::Edge> {
public:
	::size_t operator()(const mcmc::Edge &x) const;
};
}


namespace mcmc {

void dump(const EdgeSet &s) {
	for (EdgeSet::const_iterator e = s.begin(); e != s.end(); e++) {
		std::cout << *e << std::endl;
	}
}

bool present(const EdgeSet &s, const Edge &edge) {
	for (EdgeSet::const_iterator e = s.begin(); e != s.end(); e++) {
		if (*e == edge) {
			return true;
		}
		assert(e->first != edge.first || e->second != edge.second);
	}

	return false;
}

/**
 * Data class is an abstraction for the raw data, including vertices and edges.
 * It's possible that the class can contain some pre-processing functions to clean 
 * or re-structure the data.   
 *             
 * The data can be absorbed directly by sampler.
 */
class Data {
public:
	Data(const void *V, const EdgeSet *E, int N) {
		this->V = V;
		this->E = E;
		this->N = N;
	}

	virtual ~Data() {
		// delete const_cast<void *>(V);
		delete const_cast<EdgeSet *>(E);
	}

	void dump_data() const {
		std::cout << "Edge set size " << N << std::endl;
		for (EdgeSet::const_iterator edge = E->begin(); edge != E->end(); edge++) {
			std::cout << "    " << std::setw(10) << edge->first <<
				" " << std::setw(10) << edge->second << std::endl;
		}
	}

public:
	const void *V;	// mapping between vertices and attributes.
	const EdgeSet *E;	// all pair of "linked" edges.
	int N;				// number of vertices
};

}	// namespace mcmc

namespace std {
::size_t hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
	::size_t h = std::hash<int>()(x.first) ^ std::hash<int>()(x.second);
	return h;
}
}

#endif	// ndef MCMC_DATA_H__
