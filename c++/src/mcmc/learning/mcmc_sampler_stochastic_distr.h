#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__

#include <unordered_set>
#include <iostream>

#include "mcmc/config.h"

#include "dkvstore/DKVStore.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"

#include "mcmc/learning/mcmc_sampler_stochastic.h"

#ifdef MAYBE_UNUSED
#include <cinttypes>
#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>	// min, max
#include <chrono>

#include "mcmc/exception.h"

#include "mcmc/np.h"

#endif

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
  double	likelihood;
};


struct perp_accu {
  perp_counter link;
  perp_counter non_link;
};


class PerpData {
 public:
  void Init(::size_t max_perplexity_chunk);

  std::vector<Vertex> nodes_;
  std::vector<double*> pi_;
  // OpenMP parallelism requires a vector
  std::vector<EdgeMapItem> data_;
  std::vector<perp_accu> accu_;
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

  void BroadcastNetworkInfo();

  void BroadcastHeldOut();

  void MasterAwareLoadNetwork();

  virtual void init();

  virtual void run();

 protected:
  template <typename T>
    std::vector<const T*>& constify(std::vector<T*>& v) {
      // Compiler doesn't know how to automatically convert
      // std::vector<T*> to std::vector<T const*> because the way
      // the template system works means that in theory the two may
      // be specialised differently.  This is an explicit conversion.
      return reinterpret_cast<std::vector<const T*>&>(v);
    }


  ::size_t real_num_node_sample() const;

  void init_beta();

  // Calculate pi[0..K> ++ phi_sum from phi[0..K>
  void pi_from_phi(double* pi, const std::vector<double> &phi);

  void init_pi();

  void check_perplexity();

  void ScatterSubGraph(const std::vector<std::vector<int32_t>> &subminibatch);

  EdgeSample deploy_mini_batch();


  void update_phi(std::vector<std::vector<double>>* phi_node);


  void update_phi_node(::size_t index, Vertex i, const double* pi_node,
                       const std::vector<int32_t>::iterator &neighbors,
                       const std::vector<double*>::iterator &pi,
                       double eps_t, Random::Random* rnd,
                       std::vector<double>* phi_node	// out parameter
                      );

  void update_beta(const MinibatchSet &mini_batch, double scale);

  void reduce_plus(const perp_accu &in, perp_accu* accu);


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
  double cal_perplexity_held_out();

  int node_owner(Vertex node) const;

  static void mpi_error_test(int r, const std::string &message);

 protected:
  ::size_t	max_minibatch_nodes_;
  ::size_t	max_pi_cache_entries_;
  ::size_t	max_minibatch_chunk_;
  ::size_t	max_perplexity_chunk_;

  // Lift to class member to avoid (de)allocation in each iteration
  std::vector<int32_t> nodes_;		// my minibatch nodes
  std::vector<double*> pi_update_;
  // gradients K*2 dimension
  std::vector<std::vector<std::vector<double> > > grads_beta_;

  const int     mpi_master_;
  int		mpi_size_;
  int		mpi_rank_;

  bool          master_is_worker_;
  bool          master_hosts_pi_;

  DKV::DKVStoreInterface* d_kv_store_;

  Random::Random* phi_init_rng_;
  std::vector<Random::Random*> neighbor_sample_rng_;
  std::vector<Random::Random*> phi_update_rng_;

  bool REPLICATED_NETWORK = false;
  LocalNetwork  local_network_;

  PerpData      perp_;

  Timer         t_outer_;
  Timer         t_populate_pi_;
  Timer         t_perplexity_;
  Timer         t_cal_edge_likelihood_;
  Timer         t_mini_batch_;
  Timer         t_nodes_in_mini_batch_;
  Timer         t_sample_neighbor_nodes_;
  Timer         t_update_phi_;
  Timer         t_update_phi_in_;
  Timer         t_load_pi_minibatch_;
  Timer         t_load_pi_neighbor_;
  Timer         t_barrier_phi_;
  Timer         t_update_pi_;
  Timer         t_barrier_pi_;
  Timer         t_update_beta_;
  Timer         t_beta_zero_;
  Timer         t_beta_rank_;
  Timer         t_load_pi_beta_;
  Timer         t_beta_calc_grads_;
  Timer         t_beta_sum_grads_;
  Timer         t_beta_update_theta_;
  Timer         t_load_pi_perp_;
  Timer         t_store_pi_minibatch_;
  Timer         t_purge_pi_perp_;
  Timer         t_reduce_perp_;
  Timer         t_broadcast_beta_;
  Timer         t_deploy_minibatch_;

  clock_t	t_start_;
  std::vector<double> timings_;
};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_DISTR_H__