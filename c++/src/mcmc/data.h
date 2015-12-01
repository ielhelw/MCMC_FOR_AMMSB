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

#include "mcmc/config.h"

/**
 * Offers 2 different implementations of the NetworkGraph class:
 * 2) -DMCMC_EDGESET_IS_ADJACENCY_LIST
 *    NetworkGraph is a std::vector<google sparsehash set>
 *      least memory pressure
 *      supports distribution of subgraphs
 * 3) otherwise
 *    NetworkGraph is a std::unordered_set<Edge>
 *      fastest
 *      no support for distributed
 */

#include <unistd.h>

#include <utility>
#include <map>
#if ! defined MCMC_EDGESET_IS_ADJACENCY_LIST
#  include <unordered_map>
#endif
#include <unordered_set>
#include <list>
#include <iostream>
#include <iomanip>

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
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

class Edge {
 public:
  // google::sparse_hash_map requires me to have a default constructor
  // inline for performance
  Edge() { }

  // inline for performance
  Edge(Vertex a, Vertex b) : first(a), second(b) {}

  Edge(std::istream &s);

  // Shorthand for the eternal repetition of set.find(*this) != set.end()
  template <typename SET>
  bool in(const SET &s) const {
    return s.find(*this) != s.end();
  }

  // inline for performance
  bool operator==(const Edge &a) const {
    return a.first == first && a.second == second;
  }

  // inline for performance
  bool operator<(const Edge &a) const {
    return first < a.first || (first == a.first && second < a.second);
  }

  std::ostream &put(std::ostream &s) const;

 protected:
  static char consume(std::istream &s, char expect);

 public:
  std::istream &get(std::istream &s);

  Vertex first;
  Vertex second;
};


struct EdgeHash {
 public:
  ::size_t operator()(const mcmc::Edge &x) const {
    // inline for performance
    ::size_t h = ((size_t)x.first * (size_t)x.second) ^
      ((size_t)x.first + (size_t)x.second);
    return h;
  }
};


typedef std::unordered_set<Vertex> VertexSet;
typedef std::unordered_set<Vertex> NodeSet;
typedef std::unordered_set<Edge, EdgeHash> MinibatchSet;

#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST

typedef std::unordered_map<Edge, bool, EdgeHash> EdgeMap;
typedef std::unordered_set<Edge, EdgeHash> NetworkGraph;

#else // ndef MCMC_EDGESET_IS_ADJACENCY_LIST

class GoogleHashSet : public google::sparse_hash_set<Vertex> {
 public:
  GoogleHashSet() {
    // this->set_empty_key(-1);
    this->set_deleted_key(-2);
  }
};

struct EdgeEquals {
  // inline for performance
  bool operator()(const Edge &e1, const Edge &e2) const {
    return e1 == e2;
  }
};

class GoogleHashMap
    : public google::sparse_hash_map<Edge, bool, EdgeHash, EdgeEquals> {
 public:
  GoogleHashMap() { this->set_deleted_key(Edge(-2, -2)); }
};


typedef GoogleHashMap EdgeMap;

// Implements so much (so little) of the set/unordered set interface that it
// can be used virtually without specialization
class NetworkGraph {
 public:
  typedef Edge key_type;
  typedef key_type value_type;

  NetworkGraph() { }
  NetworkGraph(const std::string &filename, ::size_t progress = 0);

  template <typename SubListIterator>
  class Iterator {
   public:
    Iterator(const NetworkGraph& outer, Vertex v,
             SubListIterator edge_iterator)
        : outer_(outer), v_(v), edge_iterator_(edge_iterator) {
    }

    Iterator& operator++() {
      if (static_cast<::size_t>(v_) == outer_.edges_at_.size()) return *this;
      ++edge_iterator_;
      if (edge_iterator_ == outer_.edges_at_[v_].end()) {
        ++v_;
        for (; static_cast< ::size_t>(v_) < outer_.edges_at_.size(); ++v_) {
          if (!outer_.edges_at_[v_].empty()) {
            edge_iterator_ = outer_.edges_at_[v_].begin();
            break;
          }
        }
        if (static_cast<::size_t>(v_) == outer_.edges_at_.size()) {
          edge_iterator_ = outer_.edges_at_.back().end();
        }
      }
      return *this;
    }

    bool operator!=(const Iterator& other) {
      if (v_ != other.v_) {
        return true;
      }
      if (static_cast< ::size_t>(v_) == outer_.edges_at_.size()) {
        return false;
      }
      return edge_iterator_ != other.edge_iterator_;
    }

    Edge operator*() {
      if (v_ < 0 || static_cast<::size_t>(v_) >= outer_.edges_at_.size()) {
        std::cout << "ISSUE 1" << std::endl;
        abort();
      }
      if (edge_iterator_ == outer_.edges_at_[v_].end()) {
        std::cout << "ISSUE 2: v=" << v_ << ", edges_at_.size=" << outer_.edges_at_.size() << std::endl;
        abort();
      }
      return Edge(v_, *edge_iterator_);
    }

    static Iterator end(const NetworkGraph& outer_) {
      ::size_t n = outer_.edges_at_.size();
      if (n == 0) {
        return Iterator(outer_);
      }
      return Iterator(outer_, n, outer_.edges_at_[n - 1].end());
    }

   private:
    // generate end() via a special constructor:
    Iterator(const NetworkGraph& outer)
        : outer_(outer), v_(outer.edges_at_.size()) {
    }

    const NetworkGraph& outer_;
    Vertex      v_;
    SubListIterator edge_iterator_;
  };


  typedef Iterator<GoogleHashSet::iterator> iterator;
  typedef Iterator<GoogleHashSet::const_iterator> const_iterator;

  iterator begin() noexcept {
    if (edges_at_.size() == 0) {
      return end();
    }
    Vertex i = 0;
    for (; static_cast<::size_t>(i) < edges_at_.size(); ++i) {
      if (!edges_at_[i].empty()) {
        return iterator(*this, i, edges_at_[i].begin());
      }
    }
    return end();
  }

  iterator end() noexcept {
    return iterator::end(*this);
  }

  const_iterator begin() const noexcept {
    if (edges_at_.size() == 0) {
      return end();
    }
    Vertex i = 0;
    for (; static_cast<::size_t>(i) < edges_at_.size(); ++i) {
      if (!edges_at_[i].empty()) {
        return const_iterator(*this, i, edges_at_[i].begin());
      }
    }
    return end();
  }

  const_iterator end() const noexcept {
    return const_iterator::end(*this);
  }

#ifdef WHAT_IS_THIS
  // google_sparse_hash has no cbegin/cend either...
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;
#endif

  iterator find(const key_type& k) {
    if (static_cast< ::size_t>(k.first) >= edges_at_.size()) {
      return end();
    }

    auto r = edges_at_[k.first].find(k.second);
    if (r == edges_at_[k.first].end()) {
      return end();
    }
    return iterator(*this, k.first, r);
  }

  const_iterator find(const key_type& k) const {
    if (static_cast< ::size_t>(k.first) >= edges_at_.size()) {
      return end();
    }

    auto r = edges_at_[k.first].find(k.second);
    if (r == edges_at_[k.first].end()) {
      return end();
    }
    return const_iterator(*this, k.first, r);
  }

  std::pair<iterator, bool> insert(const value_type& val) {
    ::size_t max_v = static_cast< ::size_t>(std::max(val.first, val.second));
    if (max_v >= edges_at_.size()) {
      edges_at_.resize(max_v + 1);
    }

    auto r1 = edges_at_[val.first].insert(val.second);
    auto r2 = edges_at_[val.second].insert(val.first);
    assert(r1.second == r2.second);
    if (r1.second) {
      ++size_;
    }
    if (r2.second) {
      ++size_;
    }

    return std::pair<iterator, bool>(iterator(*this, val.first, r1.first),
                                     r1.second);
  }

  ::size_t edges_at_size() const {
    return edges_at_.size();
  }

  const GoogleHashSet& edges_at(Vertex v) const {
    return edges_at_[v];
  }

  ::size_t size() const {
    return size_;
  }

 private:
  std::vector<GoogleHashSet> edges_at_;
  ::size_t size_ = 0;
};

#endif  // ndef MCMC_EDGESET_IS_ADJACENCY_LIST

typedef NodeSet MinibatchNodeSet;
typedef NodeSet NeighborSet;
typedef std::list<Edge> EdgeList;


std::ostream &operator<<(std::ostream &s, const Edge &e);

std::istream &operator>>(std::istream &s, Edge &e);

std::ostream& dump(std::ostream& out, const NetworkGraph& graph);

std::ostream& dump(std::ostream& out, const EdgeMap& s);

template <typename EdgeContainer>
std::ostream& dump(std::ostream& out, const EdgeContainer& s) {
  for (auto e = s.cbegin(); e != s.cend(); e++) {
    out << *e << std::endl;
  }

  return out;
}

// FIXME: why is that in data.h? Prefer it in np or some other misc thingy?
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

#endif  // ndef MCMC_DATA_H__
