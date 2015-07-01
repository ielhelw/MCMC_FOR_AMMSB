/*
 * Copyright notice goes here
 */

/*
 * @author Wenzhe Li
 * @author Rutger Hofman, VU Amsterdam
 *
 * @date 2014-08-6
 */

#ifndef MCMC_DATA_H__
#define MCMC_DATA_H__

#include <utility>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <iostream>
#include <iomanip>

#include "mcmc/exception.h"

namespace mcmc {

#define EDGESET_IS_ADJACENCY_LIST

typedef std::vector<std::unordered_set<int>> AdjacencyList;

class Edge {
public:
	Edge(int a, int b) : first(a), second(b) {
	}

	Edge(std::istream &s) {
		(void)get(s);
	}

	template <typename SET>
	bool in(const SET &s) const {
		return s.find(*this) != s.end();
	}

	bool in(const AdjacencyList &s) const {
		if (static_cast<::size_t>(first) >= s.size() ||
				static_cast<::size_t>(second) >= s.size()) {
			return false;
		}

		if (s[first].size() > s[second].size()) {
			return s[second].find(first) != s[second].end();
		} else {
			return s[first].find(second) != s[first].end();
		}
	}

	template <typename SET>
	void insertMe(SET *s) const {
		s->insert(*this);
	}

	void insertMe(AdjacencyList *s) const {
		if (static_cast<::size_t>(first) >= s->size()) {
			s->resize(first + 1);
		}
		if (static_cast<::size_t>(second) >= s->size()) {
			s->resize(second + 1);
		}
		(*s)[first].insert(second);
		(*s)[second].insert(first);
	}

	bool operator== (const Edge &a) const {
		return a.first == first && a.second == second;
	}

	bool operator< (const Edge &a) const {
		return first < a.first || (first == a.first && second < a.second);
	}

	std::ostream &put(std::ostream &s) const {
		s << std::setw(1) << "(" << first << ", " << second << ")";

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
	std::istream &get(std::istream &s) {
		// std::string line;
		// std::getline(s, line);
		// std::cerr << "In get(): '" << line << "'" << std::endl;

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


#ifdef RANDOM_FOLLOWS_PYTHON

#ifdef RANDOM_FOLLOWS_CPP
#error "RANDOM_FOLLOWS_CPP is incompatible with RANDOM_FOLLOWS_PYTHON"
#endif

typedef std::unordered_set<int>			VertexSet;
typedef std::set<int>					OrderedVertexSet;

typedef std::unordered_set<Edge>		NetworkGraph;
typedef std::set<Edge>					MinibatchSet;
typedef std::list<Edge>					EdgeList;

typedef std::map<Edge, bool>			EdgeMap;

#else	// def RANDOM_FOLLOWS_PYTHON
typedef std::unordered_set<int>			VertexSet;
#ifdef RANDOM_FOLLOWS_CPP
typedef std::set<int>					OrderedVertexSet;
#else
typedef VertexSet						OrderedVertexSet;
#endif

#ifdef EDGESET_IS_ADJACENCY_LIST
typedef AdjacencyList					NetworkGraph;
#else
typedef std::unordered_set<Edge>		NetworkGraph;
#endif
typedef std::unordered_set<Edge>		MinibatchSet;
typedef std::list<Edge>					EdgeList;

typedef std::unordered_map<Edge, bool>	EdgeMap;
#endif	// def RANDOM_FOLLOWS_PYTHON

}	// namespace mcmc


namespace std {
template<>
struct hash<mcmc::Edge> {
public:
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
	int operator()(const mcmc::Edge &x) const;
#else
	::size_t operator()(const mcmc::Edge &x) const;
#endif
};
}


namespace mcmc {


std::ostream &dump_edgeset(std::ostream &out, ::size_t N,
						   const std::unordered_set<Edge> &E) {
	// out << "Edge set size " << N << std::endl;
	for (auto edge : E) {
		out << edge.first << "\t" << edge.second << std::endl;
	}

	return out;
}


std::ostream &dump_edgeset(std::ostream &out, ::size_t N, const AdjacencyList &E) {
	// out << "Edge set size " << N << std::endl;
	for (::size_t n = 0; n < E.size(); n++) {
		for (auto e : E[n]) {
			if (e > static_cast<int>(n)) {
				out << n << "\t" << e << std::endl;
			}
		}
	}

	return out;
}

bool present(const std::unordered_set<Edge> &s, const Edge &edge) {
	for (auto e : s) {
		if (e == edge) {
			return true;
		}
		assert(e.first != edge.first || e.second != edge.second);
	}

	return false;
}

bool present(const AdjacencyList &s, const Edge &edge) {
	for (auto e : s[edge.first]) {
		if (e == edge.second) {
			return true;
		}
	}

	return false;
}

void dump(const EdgeMap &s) {
	for (auto e = s.begin(); e != s.end(); e++) {
		std::cout << e->first << ": " << e->second << std::endl;
	}
}

template <typename EdgeContainer>
void dump(const EdgeContainer &s) {
	for (auto e = s.cbegin(); e != s.cend(); e++) {
		std::cout << *e << std::endl;
	}
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
	Data(const void *V, const NetworkGraph *E, int N, const std::string &header = "") :
		V(V), E(E), N(N), header_(header) {
	}

	~Data() {
		// delete const_cast<void *>(V); FIXME: somebody must delete V; the 'owner' of this dataset, I presume
		delete const_cast<NetworkGraph *>(E);
	}

	void dump_data() const {
		// std::cout << "Edge set size " << N << std::endl;
		std::cout << header_;
		(void)dump_edgeset(std::cout, N, *E);
	}

public:
	const void *V;	// mapping between vertices and attributes.
	const NetworkGraph *E;	// all pair of "linked" edges.
	int N;				// number of vertices
	std::string header_;
};

}	// namespace mcmc

namespace std {
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
int hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
	int h = std::hash<int>()(x.first) ^ std::hash<int>()(x.second);
	return h;
}
#else
::size_t hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
	::size_t h = ((size_t)x.first * (size_t)x.second) ^ ((size_t)x.first + (size_t)x.second);
	return h;
}
#endif
}

#endif	// ndef MCMC_DATA_H__
