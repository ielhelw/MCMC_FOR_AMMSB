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

#include "mcmc/config.h"

#ifdef MCMC_USE_GOOGLE_SPARSE_HASH
// If <cinttypes> is not included before <google/sparse_hash_set>, compile
// errors because of missing defines of SCNd64 and friends
#include <cinttypes>
#include <google/sparse_hash_set>
#include <google/sparse_hash_map>
#endif

#include "mcmc/exception.h"
#include "mcmc/np.h"

namespace mcmc {

typedef int32_t Vertex;
class Edge;
#ifdef MCMC_USE_GOOGLE_SPARSE_HASH
class GoogleHashMap;
class GoogleHashSet;
typedef GoogleHashMap EdgeMap;
typedef std::vector<GoogleHashSet> AdjacencyList;
#else
typedef std::vector<std::unordered_set<Vertex>> AdjacencyList;
typedef std::unordered_map<Edge, bool> EdgeMap;
#endif

#ifdef RANDOM_FOLLOWS_PYTHON

#ifdef RANDOM_FOLLOWS_CPP
#error "RANDOM_FOLLOWS_CPP is incompatible with RANDOM_FOLLOWS_PYTHON"
#endif

typedef std::unordered_set<Vertex> VertexSet;
typedef std::set<Vertex> OrderedVertexSet;

typedef std::unordered_set<Edge> NetworkGraph;
typedef std::set<Edge> MinibatchSet;
typedef std::list<Edge> EdgeList;

typedef std::map<Edge, bool> EdgeMap;

#else  // def RANDOM_FOLLOWS_PYTHON
typedef std::unordered_set<Vertex> VertexSet;
#ifdef RANDOM_FOLLOWS_CPP
typedef std::set<Vertext> OrderedVertexSet;
#else
typedef VertexSet OrderedVertexSet;
#endif

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
typedef AdjacencyList NetworkGraph;
#else
typedef std::unordered_set<Edge> NetworkGraph;
#endif
typedef std::unordered_set<Edge> MinibatchSet;
typedef std::list<Edge> EdgeList;

#endif  // def RANDOM_FOLLOWS_PYTHON

class Edge {
 public:
  // google::sparse_hash_map requires me to have a default constructor
  Edge();

  Edge(Vertex a, Vertex b);

  Edge(std::istream &s);

  template <typename SET>
  bool in(const SET &s) const {
    return s.find(*this) != s.end();
  }

  template <typename SET>
  bool in(const std::vector<SET> &s) const {
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

  void insertMe(AdjacencyList *s) const;

  bool operator==(const Edge &a) const;

  bool operator<(const Edge &a) const;

  std::ostream &put(std::ostream &s) const;

 protected:
  static char consume(std::istream &s, char expect);

 public:
  std::istream &get(std::istream &s);

  Vertex first;
  Vertex second;
};

std::ostream &operator<<(std::ostream &s, const Edge &e);

std::istream &operator>>(std::istream &s, Edge &e);

std::ostream &dump_edgeset(std::ostream &out, ::size_t N,
                           const std::unordered_set<Edge> &E);

std::ostream &dump_edgeset(std::ostream &out, ::size_t N,
                           const AdjacencyList &E);

bool present(const std::unordered_set<Edge> &s, const Edge &edge);

bool present(const AdjacencyList &s, const Edge &edge);

void dump(const EdgeMap &s);

template <typename EdgeContainer>
void dump(const EdgeContainer &s) {
  for (auto e = s.cbegin(); e != s.cend(); e++) {
    std::cout << *e << std::endl;
  }
}

void print_mem_usage(std::ostream &s);

/**
 * Data class is an abstraction for the raw data, including vertices and edges.
 * It's possible that the class can contain some pre-processing functions to
 *clean
 * or re-structure the data.
 *
 * The data can be absorbed directly by sampler.
 */
class Data {
 public:
  Data(const void *V, const NetworkGraph *E, Vertex N,
       const std::string &header = "");

  ~Data();

  void dump_data() const;

  void save(const std::string &filename, bool compressed = false) const;

 public:
  const void *V;          // mapping between vertices and attributes.
  const NetworkGraph *E;  // all pair of "linked" edges.
  Vertex N;               // number of vertices
  std::string header_;
};

}  // namespace mcmc

namespace std {
template <>
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

#ifdef MCMC_USE_GOOGLE_SPARSE_HASH
class GoogleHashSet : public google::sparse_hash_set<Vertex> {
 public:
  GoogleHashSet() {
    // this->set_empty_key(-1);
    this->set_deleted_key(-2);
  }
};

struct EdgeEquals {
  bool operator()(const Edge &e1, const Edge &e2) const;
};

class GoogleHashMap
    : public google::sparse_hash_map<Edge, bool, std::hash<Edge>, EdgeEquals> {
 public:
  GoogleHashMap() { this->set_deleted_key(Edge(-2, -2)); }
};
#endif

}  // namespace mcmc

#endif  // ndef MCMC_DATA_H__
