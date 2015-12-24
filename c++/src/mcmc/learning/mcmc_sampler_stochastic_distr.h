#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__

#include <unordered_set>
#include <iostream>
#include <thread>
#include <boost/thread.hpp>

#include "mcmc/config.h"

#include "dkvstore/DKVStore.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"

#include "mcmc/learning/mcmc_sampler_stochastic.h"

namespace mcmc {
namespace learning {

using ::mcmc::timer::Timer;


// Mirror of the global Network Graph that contains only a small slice of
// the edges, i.c. the edges whose first element is in the minibatch
class LocalNetwork {
 public:
  typedef typename std::unordered_set<Vertex> EndpointSet;

  void unmarshall_local_graph(::size_t index, const Vertex* linked,
                              ::size_t size);

  void reset();

  bool find(const Edge& edge) const;

  const EndpointSet &linked_edges(::size_t i) const;

  void swap(::size_t from, ::size_t to);

 protected:
  std::vector<EndpointSet> linked_edges_;
};


struct perp_counter {
  perp_counter() : count(0), likelihood(0.0) {
  }

  void reset() {
    count = 0;
    likelihood = 0.0;
  }

  ::size_t count;
  Float	likelihood;
};


struct perp_accu {
  perp_counter link;
  perp_counter non_link;
};


class PerpData {
 public:
  void Init(::size_t max_perplexity_chunk);

  std::vector<Vertex> nodes_;
  std::vector<Float*> pi_;
  // OpenMP parallelism requires a vector
  std::vector<EdgeMapItem> data_;
  std::vector<perp_accu> accu_;
};


// **************************************************************************
//
// class PiChunk
//
// **************************************************************************

class PiChunk {
 public:
  ::size_t      buffer_index_;  // index into the multi-buffer
  ::size_t      start_;         // start index of my chunk nodes into the graph
  std::vector<Vertex> chunk_nodes_;
  std::vector<Float *> pi_node_;
  std::vector<Vertex> flat_neighbors_;
  std::vector<Float *> pi_neighbor_;
  void swap(PiChunk& x);
};


// **************************************************************************
//
// class ChunkPipeline
//
// **************************************************************************

namespace PIPELINE_STATE {
  enum PipelineState {
    FILL_REQUESTED,
    FILLING,
    FILLED,
    FREE,
    STOP,
  };
}

class ChunkPipeline;

class PipelineBuffer {
 public:
  PipelineBuffer() : state_(PIPELINE_STATE::FREE) { }

  PiChunk* chunk_;
  PIPELINE_STATE::PipelineState state_;
  boost::condition_variable fill_requested_;
  boost::condition_variable fill_completed_;

  friend class ChunkPipeline;
};


class ChunkPipeline {
 public:
  ChunkPipeline(::size_t num_buffers, DKV::DKVStoreInterface& d_kv_store)
      : buffer_(num_buffers), d_kv_store_(d_kv_store),
        client_enq_(0), client_deq_(0), client_clear_(0),
        server_deq_(0), server_complete_(0) {
  }

  // Queue interface
  void EnqueueChunk(PiChunk* chunk);
  PiChunk* AwaitChunkFilled();
  PiChunk* DequeueChunk();
  void NotifyChunkFilled();
  void ReleaseChunk();
  void Stop();
  ::size_t num_buffers() const;

  void operator() ();
  std::ostream& report(std::ostream& s) const;

  // Nonqueue interface
  ::size_t GrabFreeBufferIndex() const;

private:
  boost::mutex lock_;
  std::vector<PipelineBuffer> buffer_;
  DKV::DKVStoreInterface& d_kv_store_;
  ::size_t client_enq_;
  ::size_t client_deq_;
  ::size_t client_clear_;
  ::size_t server_deq_;
  ::size_t server_complete_;

  Timer         t_load_pi_minibatch_;
  Timer         t_load_pi_neighbor_;
};


// **************************************************************************
//
// class MinibatchSlice
//
// **************************************************************************

class MinibatchSlice {
 public:
  MinibatchSlice() : processed_(0) {
  }

  void SwapNodeAndGraph(::size_t i, ::size_t with, PiChunk* c, PiChunk* with_c);

  std::vector<PiChunk> pi_chunks_;
  std::vector<Vertex> nodes_;       // my subsample of the minibatch
  VertexSet full_minibatch_nodes_;
  LocalNetwork  local_network_;
  EdgeSample edge_sample_;

 private:
  ::size_t processed_;

  friend class MinibatchPipeline;
};


// **************************************************************************
//
// class MinibatchPipeline
//
// **************************************************************************

class MCMCSamplerStochasticDistributed;

class MinibatchPipeline {
 public:
  MinibatchPipeline(MCMCSamplerStochasticDistributed& sampler,
                    ::size_t max_minibatch_chunk,
                    ChunkPipeline& chunk_pipeline,
                    std::vector<Random::Random*>& rng);
  void StageNextChunk();
  void StageNextMinibatch();
  void AdvanceMinibatch();
  const MinibatchSlice& CurrentChunk() {
    return minibatch_slice_[current_];
  }
  // Returns whether any node is in <code>one</code> and the previous
  // minibatch set
  bool PreviousMinibatchOverlap(Vertex one) const;
  void ReorderMinibatchOverlap(MinibatchSlice* mb_chunk);
  void CreateMinibatchSliceChunks(MinibatchSlice* mb_chunk);
  void SampleNeighbors(MinibatchSlice* mb_chunk);
  /** @return false if there is overlap with previous minibatch sample */
  bool SampleNeighborSet(PiChunk* pi_chunk, ::size_t i);

  std::ostream& report(std::ostream& s) const;

 private:

  MCMCSamplerStochasticDistributed& sampler_;
  ChunkPipeline& chunk_pipeline_;

  std::vector<MinibatchSlice> minibatch_slice_;
  ::size_t      max_minibatch_chunk_;

  std::vector<Random::Random*>& rng_;

  ::size_t      prev_;
  ::size_t      current_;
  ::size_t      next_;

  Timer         t_chunk_minibatch_slice_;
  Timer         t_reorder_minibatch_overlap_;
  Timer         t_sample_neighbor_nodes_;
  Timer         t_resample_neighbor_nodes_;
};


/**
 * The distributed version differs in these aspects from the parallel version:
 *  - the minibatch is distributed
 *  - pi/phi is distributed. It is not necessary to store both phi and pi since
 *    the two are equivalent. We store pi + sum(phi), and we can restore phi
 *    as phi[i] = pi[i] * phi_sum.
 *
 * Algorithm:
 * LOOP:
 *   One host (the Master) creates the minibatch and distributes its nodes over
 *   the workers. Each worker samples a neighbor set for each of its minibatch
 *   nodes.
 *
 *   Then, each worker:
 *    - fetches pi/phi for its minibatch nodes
 *    - fetches pi/phi for the neighbors of its minibatch nodes
 *    - calulates an updated phi/pi for its minibatch nodes
 *    - stores the updated phi/pi
 *
 *   Then, all synchronise. The master calculates an updated value for beta.
 *   This could plausibly be done in a distributed way, but it is so quick that
 *   we guess there is no point to do that.
 *   If needs, calculate the perplexity in parallel. If termination is met,
 *   let the workers know. Else, the master broadcasts its updated value for
 *   beta.
 * END LOOP
 */

/**
 * For the sequential part of the algorithm, see class MCMCSamplerStochastic
 */
class MCMCSamplerStochasticDistributed : public MCMCSamplerStochastic {

 public:
  MCMCSamplerStochasticDistributed(const Options &args);

  virtual ~MCMCSamplerStochasticDistributed();

  void init() override;

  void run() override;

  void deploy_mini_batch(MinibatchSlice* mb_chunk);

  ::size_t real_num_node_sample() const;

 protected:
  template <typename T>
  std::vector<const T*>& constify(std::vector<T*>& v) {
    // Compiler doesn't know how to automatically convert
    // std::vector<T*> to std::vector<T const*> because the way
    // the template system works means that in theory the two may
    // be specialised differently.  This is an explicit conversion.
    return reinterpret_cast<std::vector<const T*>&>(v);
  }

  void BroadcastNetworkInfo();

  void BroadcastHeldOut();

  void MasterAwareLoadNetwork();

  void InitSlaveState(const NetworkInfo &info, ::size_t world_rank);

  void InitDKVStore();

  void init_theta();
  void beta_from_theta();

  void init_pi();
  // Calculate pi[0..K> ++ phi_sum from phi[0..K>
  void pi_from_phi(Float* pi, const std::vector<Float> &phi);

  void ScatterSubGraph(const std::vector<std::vector<Vertex> >& subminibatch,
                       MinibatchSlice *mb_chunk);

  std::ostream& PrintStats(std::ostream& out) const;

  EdgeSample deploy_mini_batch();
  void update_phi(std::vector<std::vector<Float> >* phi_node);
  void update_phi_node(const MinibatchSlice& mb_chunk, const PiChunk& pi_chunk,
                       ::size_t index, Vertex i, const Float* pi_node,
                       Float eps_t, Random::Random* rnd,
                       std::vector<Float>* phi_node	// out parameter
                      );
  void update_pi(const MinibatchSlice& mb_chunk,
                 const std::vector<std::vector<Float> >& phi_node);

  void broadcast_theta_beta();
  void scatter_minibatch_for_theta(const MinibatchSet &mini_batch,
                                   std::vector<EdgeMapItem>* mini_batch_slice);
  void beta_calc_grads(const std::vector<EdgeMapItem>& mini_batch_slice);
  void beta_sum_grads();
  void beta_update_theta(Float scale);
  void update_beta(const MinibatchSet &mini_batch, Float scale);

  void reduce_plus(const perp_accu &in, perp_accu* accu);
  void check_perplexity(bool force);
  /**
   * calculate the perplexity for data.
   * perplexity defines as exponential of negative average log likelihood. 
   * formally:
   *     ppx = exp(-1/N * \sum){i}^{N}log p(y))
   * 
   * we calculate average log likelihood for link and non-link separately, with the 
   * purpose of weighting each part proportionally. (the reason is that we sample 
   * the equal number of link edges and non-link edges for held out data and test data,
   * which is not true representation of actual data set, which is extremely sparse.
   */
  Float cal_perplexity_held_out();

  int node_owner(Vertex node) const;

  static void mpi_error_test(int r, const std::string &message);

 protected:
  ::size_t	max_minibatch_nodes_;
  ::size_t	max_minibatch_chunk_;
  ::size_t	max_perplexity_chunk_;
  ::size_t  max_dkv_write_entries_;

  std::vector<Float*> pi_update_;
  // gradients K*2 dimension
  std::vector<std::vector<std::vector<Float> > > grads_beta_;

  const int     mpi_master_;
  int		    mpi_size_;
  int		    mpi_rank_;

  bool          master_is_worker_;
  bool          master_hosts_pi_;

  ::size_t      num_buffers_;
  std::unique_ptr<ChunkPipeline> chunk_pipeline_;
  boost::thread dkv_server_thread_;
  std::unique_ptr<MinibatchPipeline> minibatch_pipeline_;

  std::unique_ptr<DKV::DKVStoreInterface> d_kv_store_;

  PerpData      perp_;

  ::size_t      stats_print_interval_;
  Timer         t_load_network_;
  Timer         t_init_dkv_;
  Timer         t_outer_;
  Timer         t_populate_pi_;
  Timer         t_perplexity_;
  Timer         t_cal_edge_likelihood_;
  Timer         t_mini_batch_;
  Timer         t_deploy_minibatch_;
  Timer         t_scatter_subgraph_;
  Timer         t_scatter_subgraph_marshall_edge_count_;
  Timer         t_scatter_subgraph_scatterv_edge_count_;
  Timer         t_scatter_subgraph_marshall_edges_;
  Timer         t_scatter_subgraph_scatterv_edges_;
  Timer         t_scatter_subgraph_unmarshall_;
  Timer         t_nodes_in_mini_batch_;
  Timer         t_update_phi_pi_;
  Timer         t_update_phi_;
  Timer         t_barrier_phi_;
  Timer         t_update_pi_;
  Timer         t_store_pi_minibatch_;
  Timer         t_barrier_pi_;
  Timer         t_update_beta_;
  Timer         t_beta_zero_;
  Timer         t_beta_rank_;
  Timer         t_load_pi_beta_;
  Timer         t_beta_calc_grads_;
  Timer         t_beta_sum_grads_;
  Timer         t_beta_reduce_grads_;
  Timer         t_beta_update_theta_;
  Timer         t_load_pi_perp_;
  Timer         t_purge_pi_perp_;
  Timer         t_reduce_perp_;
  Timer         t_broadcast_theta_beta_;

  std::vector<double> timings_;
};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
