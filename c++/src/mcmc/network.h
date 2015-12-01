#ifndef MCMC_MSB_NETWORK_H__
#define MCMC_MSB_NETWORK_H__

#include <algorithm>
#include <set>
#include <unordered_set>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include "mcmc/config.h"
#include "mcmc/types.h"
#include "mcmc/data.h"
#include "mcmc/random.h"
#include "mcmc/preprocess/dataset.h"
#include "mcmc/options.h"
#include "mcmc/fileio.h"
#include "mcmc/timer.h"

namespace mcmc {

using ::mcmc::timer::Timer;

typedef std::pair<MinibatchSet*, Float> EdgeSample;

/**
 * Network class represents the whole graph that we read from the
 * data file. Since we store all the edges ONLY, the size of this
 * information is much smaller due to the graph sparsity (in general,
 * around 0.1% of links are connected)
 *
 * We use the term "linked edges" to denote the edges that two nodes
 * are connected, "non linked edges", otherwise. If we just say edge,
 * it means either linked or non-link edge.
 *
 * The class also contains lots of sampling methods that sampler can utilize.
 * This is great separation between different learners and data layer. By
 *calling
 * the function within this class, each learner can get different types of
 * data.
 */

struct NetworkInfo {
  int32_t N;
  int64_t E;
  int32_t max_fan_out;
  double held_out_ratio;
  int64_t held_out_size;
};

struct EdgeMapItem {
  EdgeMapItem() { }
  EdgeMapItem(Edge &edge, bool is_edge) : edge(edge), is_edge(is_edge) { }

  Edge edge;
  bool is_edge;
};

class Network {
 public:
  Network();

  // Stub for the distributed implementation that does not replicate the graph
  Network(const NetworkInfo& info);
  virtual ~Network();

  /**
   * In this initialization step, we separate the whole data set
   * into training, validation and testing sets. Basically,
   * Training ->  used for tuning the parameters.
   * Held-out/Validation -> used for evaluating the current model, avoid
   *over-fitting
   *               , the accuracy for validation set used as stopping criteria
   * Testing -> used for calculating final model accuracy.
   *
   * Arguments:
   *     data:   representation of the while graph.
   *     held_out_ratio:  the percentage of data used for validation and
   *testing.
   *     create_held_out: create held-out set and test set; otherwise, caller
   *must
   *     					read them from file
   */
  void Init(const Options& args, double held_out_ratio,
            std::vector<Random::Random*>* rng_threaded);

  const Data* get_data() const;

  void ReadSet(FileHandle& f, EdgeMap* set);

  void ReadHeldOutSet(const std::string& filename, bool compressed);

  void ReadTestSet(const std::string& filename, bool compressed);

  void ReadAuxData(const std::string& filename, bool compressed);

  template <typename T>
  void WriteSet(FileHandle& f, const T* set) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    T* mset = const_cast<T*>(set);
    // Read held_out set
    mset->write_metadata(f.handle());
    mset->write_nopointer_data(f.handle());
#else
    throw MCMCException("Cannot write set for this representation");
#endif
  }

  void WriteHeldOutSet(const std::string& filename, bool compressed);

  void WriteTestSet(const std::string& filename, bool compressed);

  void WriteAuxData(const std::string& filename, bool compressed);

  void save(const std::string& dirname);

  // Stub info for the distributed implementation that does not replicate the
  // graph
  void FillInfo(NetworkInfo* info);

  /**
   * Sample a mini-batch of edges from the training data.
   * There are four different sampling strategies for edge sampling
   * 1.random-pair sampling
   *   sample node pairs uniformly at random.This method is an instance of
   *independent
   *   pair sampling, with h(x) equal to 1/(N(N-1)/2) * mini_batch_size
   *
   * 2.random-node sampling
   *    A set consists of all the pairs that involve one of the N nodes: we
   *first sample one of
   *    the node from N nodes, and sample all the edges for that node. h(x) =
   *1/N
   *
   * 3.stratified-random-pair sampling
   *   We divide the edges into linked and non-linked edges, and each time
   *either sample
   *   mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for
   *non-link and
   *   1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked
   *edges.
   *
   * 4.stratified-random-node sampling
   *   For each node, we define a link set consisting of all its linkes, and m
   *non-link sets
   *   that partition its non-links. We first selct a random node, and either
   *select its link
   *   set or sample one of its m non-link sets. h(x) = 1/N if linked set, 1/Nm
   *otherwise
   *
   *  Returns (sampled_edges, scale)
   *  scale equals to 1/h(x), insuring the sampling gives the unbiased
   *gradients.
   */
  EdgeSample sample_mini_batch(::size_t mini_batch_size,
                               strategy::strategy strategy);
  ::size_t num_pieces_for_minibatch(::size_t mini_batch_size) const;
  ::size_t real_minibatch_size(::size_t mini_batch_size) const;

  ::size_t max_minibatch_nodes_for_strategy(::size_t mini_batch_size,
                                            strategy::strategy strategy) const;

  ::size_t max_minibatch_edges_for_strategy(::size_t mini_batch_size,
                                            strategy::strategy strategy) const;

  ::size_t get_num_linked_edges() const;

  ::size_t get_held_out_size() const;

  int get_num_nodes() const;

  const NetworkGraph& get_linked_edges() const;

  const EdgeMap& get_held_out_set() const;

  const EdgeMap& get_test_set() const;

#ifdef UNUSED
  void set_num_pieces(::size_t num_pieces);
#endif

  EdgeSample sample_full_training_set() const;
  EdgeSample random_edge_sampling(::size_t mini_batch_size);

  /**
   * stratified sampling approach gives more attention to link edges (the edge
   * is connected by two
   * nodes). The sampling process works like this:
   * a) randomly choose one node $i$ from all nodes (1,....N)
   * b) decide to choose link edges or non-link edges with (50%, 50%)
   * probability.
   * c) if we decide to sample link edge:
   *         return all the link edges for the chosen node $i$
   *    else
   *         sample edges from all non-links edges for node $i$. The number of
   * edges
   *         we sample equals to  number of all non-link edges / num_pieces
   */
  EdgeSample stratified_random_node_sampling(::size_t num_pieces);

 protected:
  /**
   * create a set for each node, which contains list of
   * nodes. i.e {0: Set[2,3,4], 1: Set[3,5,6]...}
   * is used for sub-sampling
   * the later.
   */
  void init_train_link_map();

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST

  // FIXME: move into np/
  template <typename T>
  static void prefix_sum(std::vector<T>* a) {
#ifndef NDEBUG
    std::vector<T> orig(a->size());
#pragma omp parallel for schedule(static, 1)
    for (::size_t i = 0; i < a->size(); ++i) {
      orig[i] = (*a)[i];
    }
#endif
    ::size_t chunk =
        (a->size() + omp_get_max_threads() - 1) / omp_get_max_threads();
    std::vector< ::size_t> chunk_sum(omp_get_max_threads());
#pragma omp parallel for
    for (::size_t t = 0; t < static_cast< ::size_t>(omp_get_max_threads());
         ++t) {
      for (::size_t i = chunk * t + 1; i < std::min(a->size(), chunk * (t + 1));
           ++i) {
        (*a)[i] += (*a)[i - 1];
      }
    }
    chunk_sum[0] = 0;
    for (::size_t t = 1; t < static_cast< ::size_t>(omp_get_max_threads());
         ++t) {
      chunk_sum[t] = chunk_sum[t - 1] + (*a)[t * chunk - 1];
    }
#pragma omp parallel for
    for (::size_t t = 0; t < static_cast< ::size_t>(omp_get_max_threads());
         ++t) {
      for (::size_t i = chunk * t; i < std::min(a->size(), chunk * (t + 1));
           ++i) {
        (*a)[i] += chunk_sum[t];
      }
    }
#ifndef NDEBUG
    assert((*a)[0] == orig[0]);
#pragma omp parallel for schedule(static, 1)
    for (::size_t i = 1; i < a->size(); ++i) {
      assert((*a)[i] == (*a)[i - 1] + orig[i]);
    }
#endif
  }

  void thread_random_init(int random_seed, int world_rank);
  void thread_random_end();

  void adjacency_list_init();

  void sample_random_edges(const NetworkGraph* linked_edges, ::size_t p,
                           std::vector<Edge>* edges) const;

#endif  // def MCMC_EDGESET_IS_ADJACENCY_LIST

  /**
   * Sample held out set. we draw equal number of
   * links and non-links from the whole graph.
   */
  void init_held_out_set();

  /**
   * sample test set. we draw equal number of samples for
   * linked and non-linked edges
   */
  void init_test_set();

  template <class T>
  static bool descending(T i, T j) {
    return (i > j);
  }

  void calc_max_fan_out();

 public:
  ::size_t get_max_fan_out() const;

  ::size_t get_max_fan_out(::size_t batch_size) const;

  ::size_t get_fan_out(Vertex i);

  ::size_t marshall_edges_from(Vertex node, Vertex* marshall_area);

 protected:
  /**
   * sample one non-link edge for held out set from the network. We should make
   * sure the edge is not
   * been used already, so we need to check the condition before we add it into
   * held out sets
   * TODO: add condition for checking the infinit-loop
   */
  Edge sample_non_link_edge_for_held_out();

  /**
   * Sample one non-link edge for test set from the network. We first randomly
   * generate one
   * edge, then check conditions. If that edge passes all the conditions, return
   * that edge.
   * TODO prevent the infinit loop
   */
  Edge sample_non_link_edge_for_test();

 protected:
  const Data* data_ = NULL;
  int32_t N;                         // number of nodes in the graph
  const NetworkGraph* linked_edges;  // all pair of linked edges.
  ::size_t num_total_edges;          // number of total edges.
  double held_out_ratio_;            // percentage of held-out data size
  ::size_t held_out_size_;

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  std::vector< ::size_t> cumulative_edges;
#endif

  std::vector<Random::Random*>* rng_;

// The map stores all the neighboring nodes for each node, within the training
// set. The purpose of keeping this object is to make the stratified sampling
// process easier, in which case we need to sample all the neighboring nodes
// given the current one. The object looks like this:
// {
//     0: [1,3,1000,4000]
//     1: [0,4,999]
//   .............
// 10000: [0,441,9000]
//                         }
#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST
  std::vector<VertexSet> train_link_map;  //
#endif
  EdgeMap held_out_map;  // store all held out edges
  EdgeMap test_map;      // store all test edges

  std::vector< ::size_t> fan_out_cumul_distro;
  ::size_t progress = 0;

  Timer t_sample_sample_;
  Timer t_sample_merge_tail_;
  Timer t_sample_merge_;
};

};  // namespace mcmc

#endif  // ndef MCMC_MSB_NETWORK_H__
