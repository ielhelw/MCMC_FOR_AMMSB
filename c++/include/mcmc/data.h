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

#include <unistd.h>

#include <utility>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <iostream>
#include <iomanip>

// If <cinttypes> is not included before <google/sparse_hash_set>, compile errors
// because of missing defines of SCNd64 and friends
#include <cinttypes>
#include <google/sparse_hash_set>
#include <google/sparse_hash_map>

#include "mcmc/exception.h"
#include "mcmc/np.h"


#define EDGESET_IS_ADJACENCY_LIST

#define USE_GOOGLE_SPARSE_HASH


namespace mcmc {

typedef int32_t		Vertex;


static void print_mem_usage(std::ostream &s) {
	static const int64_t MEGA = 1 << 20;
	static int64_t pagesize = 0;
	static std::string proc_statm;
	if (pagesize == 0) {
		pagesize = sysconf(_SC_PAGESIZE);
		std::ostringstream ss;
		ss << "/proc/" << getpid() << "/statm";
		proc_statm = ss.str();
		s << "For memory query file " << proc_statm << std::endl;
	}

	std::ifstream statm(proc_statm);
	if (! statm) {
		std::cerr << "Cannot open input file \"" << proc_statm << "\"" << std::endl;
		return;
	}

	::size_t total;
	::size_t resident;
	::size_t shared;
	::size_t text;
	::size_t data;
	::size_t library;
	::size_t dirty;
	statm >> total >> resident >> shared >> text >> data >> library >> dirty;

	// s << "Memory pages: total " << total << " resident " << resident << std::endl;
	s << "Memory usage: total " << ((total * pagesize) / MEGA) << "MB " <<
	   	"resident " << ((resident * pagesize) / MEGA) << "MB " << std::endl;
}


#ifdef USE_GOOGLE_SPARSE_HASH
class GoogleHashSet : public google::sparse_hash_set<Vertex> {
public:
	GoogleHashSet() {
		// this->set_empty_key(-1);
		this->set_deleted_key(-2);
	}
};
typedef std::vector<GoogleHashSet> AdjacencyList;

#else
typedef std::vector<std::unordered_set<Vertex>> AdjacencyList;
#endif


class Edge {
public:
	// google::sparse_hash_map requires me to have a default constructor
	Edge() {
	}

	Edge(Vertex a, Vertex b) : first(a), second(b) {
	}

	Edge(std::istream &s) {
		(void)get(s);
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

#if 0
		if ((*s)[first].size() == 0) {
			(*s)[first].max_load_factor(2.0);
		}
		if ((*s)[second].size() == 0) {
			(*s)[second].max_load_factor(2.0);
		}
#endif

		(*s)[first].insert(second);
		(*s)[second].insert(first);

#if 0
		if ((*s)[first].size() <= 4) {
			(*s)[first].reserve(4);
		}
		if ((*s)[second].size() <= 4) {
			(*s)[second].reserve(4);
		}
#endif
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

	Vertex		first;
	Vertex		second;
};


inline std::ostream &operator<< (std::ostream &s, const Edge &e) {
	return e.put(s);
}


inline std::istream &operator>> (std::istream &s, Edge &e) {
	return e.get(s);
}


template <typename SET>
bool EdgeIn(const Edge &edge, const SET &s) {
	return s.find(edge) != s.end();
}

bool EdgeIn(const Edge &edge, const AdjacencyList &s) {
	if (static_cast<::size_t>(edge.first) >= s.size() ||
		static_cast<::size_t>(edge.second) >= s.size()) {
		return false;
	}

	if (s[edge.first].size() > s[edge.second].size()) {
		return s[edge.second].find(edge.first) != s[edge.second].end();
	} else {
		return s[edge.first].find(edge.second) != s[edge.first].end();
	}
}


#ifdef USE_GOOGLE_SPARSE_HASH
struct EdgeEquals {
	bool operator()(const Edge& e1, const Edge& e2) const {
		return e1 == e2;
	}
};
#endif

}	// namespace mcmc


namespace std {
template<>
struct hash<mcmc::Edge> {
public:
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
	int32_t operator()(const mcmc::Edge &x) const;
#else
	::size_t operator()(const mcmc::Edge &x) const;
#endif
};
}


namespace mcmc {

#ifdef RANDOM_FOLLOWS_PYTHON

#ifdef RANDOM_FOLLOWS_CPP
#error "RANDOM_FOLLOWS_CPP is incompatible with RANDOM_FOLLOWS_PYTHON"
#endif

typedef std::unordered_set<Vertex>		VertexSet;
typedef std::set<Vertex>				OrderedVertexSet;

typedef std::unordered_set<Edge>		NetworkGraph;
typedef std::set<Edge>					MinibatchSet;
typedef std::list<Edge>					EdgeList;

typedef std::map<Edge, bool>			EdgeMap;

#else	// def RANDOM_FOLLOWS_PYTHON
typedef std::unordered_set<Vertex>		VertexSet;
#ifdef RANDOM_FOLLOWS_CPP
typedef std::set<Vertext>				OrderedVertexSet;
#else
typedef VertexSet						OrderedVertexSet;
#endif

#ifdef EDGESET_IS_ADJACENCY_LIST
typedef AdjacencyList					NetworkGraph;
#else
typedef std::unordered_set<Edge>		NetworkGraph;
#endif

#ifdef USE_GOOGLE_SPARSE_HASH
class GoogleHashMap : public google::sparse_hash_map<Edge, bool, std::hash<Edge>, EdgeEquals> {
public:
	GoogleHashMap() {
		// this->set_empty_key(Edge(-1, -1));
		this->set_deleted_key(Edge(-2, -2));
	}
};
typedef GoogleHashMap					EdgeMap;
#else
typedef std::unordered_map<Edge, bool>	EdgeMap;
#endif

typedef std::unordered_set<Edge>		MinibatchSet;
typedef std::list<Edge>					EdgeList;

#endif	// def RANDOM_FOLLOWS_PYTHON


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
			if (e > static_cast<Vertex>(n)) {
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
	Data(const void *V, const NetworkGraph *E, Vertex N, const std::string &header = "") :
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

	void save(const std::string &filename, bool compressed = false) const {
#ifdef USE_GOOGLE_SPARSE_HASH
		FileHandle f(filename, compressed, "w");
		int32_t num_nodes = N;
		f.write_fully(&num_nodes, sizeof num_nodes);
		for (auto r : *E) {
			GoogleHashSet &rc = const_cast<GoogleHashSet &>(r);
			rc.write_metadata(f.handle());
			rc.write_nopointer_data(f.handle());
		}
#else
		throw MCMCException(__func__ + "() not implemented for this graph representation");
#endif
	}

public:
	const void *V;	// mapping between vertices and attributes.
	const NetworkGraph *E;	// all pair of "linked" edges.
	Vertex N;				// number of vertices
	std::string header_;
};

}	// namespace mcmc

namespace std {
#ifdef RANDOM_FOLLOWS_CPP_WENZHE
int32_t hash<mcmc::Edge>::operator()(const mcmc::Edge &x) const {
	int32_t h = std::hash<int32_t>()(x.first) ^ std::hash<int32_t>()(x.second);
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
