#ifndef MR_NET_BROADCAST_H__
#define MR_NET_BROADCAST_H__

#include <tuple>

#include "mr/net/netio.h"

namespace mr {
namespace net {

/**
 * Broadcast and reduce using a spanning tree, embedded in a hypercube.
 * The tree is rooted at the master. It would be straightforward
 * to generalize this to trees rooted anywhere: calculate
 * modulo the root in stead of modulo the master. But then
 * the tree construction must be dynamic, whereas for this fixed
 * case the tree, including all its connections, is static.
 */
class Broadcast {

#ifndef NDEBUG
#  define DEBUG(x)    do { x } while (0)
#else
#  define DEBUG(x)    do { } while (0)
#endif

  typedef typename std::tuple<WriterInterface *, ReaderInterface *, ::size_t> EndPoint;

 public:
  Broadcast() {
  }

  ~Broadcast() {
    Finish();
  }

  /**
   * Reentrant cleanup. Allow Broadcast to be finished before the
   * Network itself is shut down.
   */
  void Finish() {
    if (rank != master) {
      if (std::get<0>(parent) != NULL) {
        delete std::get<0>(parent);
        std::get<0>(parent) = NULL;
      }
      if (std::get<1>(parent) != NULL) {
        delete std::get<1>(parent);
        std::get<1>(parent) = NULL;
      }
    }
    for (auto & c : children) {
      if (std::get<0>(c) != NULL) {
        delete std::get<0>(c);
        std::get<0>(c) = NULL;
      }
      if (std::get<1>(c) != NULL) {
        delete std::get<1>(c);
        std::get<1>(c) = NULL;
      }
    }
  }

  void Init(Network *network) {
    if (network == NULL) {
      size = 2;
      return;
    }

    const Network::PeerVector &peers = network->getPeers();
    ::size_t i = 0;
    for (auto p : peers) {
      if (p == network->getMe()) {
        rank = i;
      }
      if (p == network->getMaster()) {
        master = i;
      }
      i++;
    }
    size = peers.size();

    ::size_t rel_rank;
    if (rank < master) {
      rel_rank = rank + size - master;
    } else {
      rel_rank = rank - master;
    }

    network_type::Type type = network_type::Type::BROADCAST;
    // Construct the spanning tree
    ::size_t bitmask = 1;
    while (bitmask < size) {
      ::size_t next_bitmask = bitmask << 1;
      ::size_t rel_peer = rel_rank ^ bitmask;
      ::size_t peer = (rel_peer + master) % size;
      if (rel_peer < size) {
        if (rel_rank < bitmask) {
          DEBUG(std::cerr << rank << "(=" << rel_rank << ") connect w/r to peer " << peer << "(=" << rel_peer << ") " << *peers[peer] << std::endl;);
          WriterInterface *w = network->createWriter(*peers[peer], type);
          ReaderInterface *r = network->createReader(*peers[peer], type);
          children.push_back(EndPoint(w, r, peer));
        } else if (rel_rank < next_bitmask) {
          DEBUG(std::cerr << rank << "(=" << rel_rank << ") connect r/w to peer " << peer << "(=" << rel_peer << ") " << *peers[peer] << std::endl;);
          ReaderInterface *r = network->createReader(*peers[peer], type);
          WriterInterface *w = network->createWriter(*peers[peer], type);
          parent = EndPoint(w, r, peer);
        }
      }
      bitmask = next_bitmask;
    }
  }

  template <typename T>
  void bcast_send(const T &data) {
    assert(rank == master);
    for (auto c : children) {
      std::get<0>(c)->write(&data, sizeof data);
    }
  }

  void bcast_send(const void *data, ::size_t size) {
    assert(rank == master);
    for (auto c : children) {
      std::get<0>(c)->write(data, size);
    }
  }

  template <typename T>
  void bcast_rcve(T *data) {
    assert(rank != master);
    std::get<1>(parent)->readFully(data, sizeof *data);
    for (auto c : children) {
      std::get<0>(c)->write(data, sizeof *data);
    }
  }

  void bcast_rcve(void *data, ::size_t size) {
    assert(rank != master);
    std::get<1>(parent)->readFully(data, size);
    for (auto c : children) {
      std::get<0>(c)->write(data, size);
    }
  }

  template <typename T>
  void reduce_send(const T &data) {
    assert(rank != master);
    T accu = data;
    for (auto c : children) {
      T contrib;
      std::get<1>(c)->readFully(&contrib, sizeof contrib);
      accu = accu + contrib;
    }
    std::get<0>(parent)->write(&accu, sizeof accu);
  }

  template <typename T>
  void reduce_send(const T &data, T reducer(const T &a, const T &b)) {
    assert(rank != master);
    T accu = data;
    for (auto c : children) {
      T contrib;
      std::get<1>(c)->readFully(&contrib, sizeof contrib);
      accu = reducer(accu, contrib);
    }
    std::get<0>(parent)->write(&accu, sizeof accu);
  }

  template <typename T>
  T reduce_rcve() {
    assert(rank == master);
    T accu = 0;
    for (auto c : children) {
      T contrib;
      std::get<1>(c)->readFully(&contrib, sizeof contrib);
      accu = accu + contrib;
    }

    return accu;
  }

  template <typename T>
  T reduce_rcve(T reducer(const T &a, const T &b)) {
    assert(rank == master);
    T accu = 0;
    for (auto c : children) {
      T contrib;
      std::get<1>(c)->readFully(&contrib, sizeof contrib);
      accu = reducer(accu, contrib);
    }

    return accu;
  }

 private:
  ::size_t rank;
  ::size_t size;
  ::size_t master;
  EndPoint parent;
  std::vector<EndPoint> children;
};

}   // namespace net
}   // namespace mr

#endif  // ndef MR_NET_BROADCAST_H__
