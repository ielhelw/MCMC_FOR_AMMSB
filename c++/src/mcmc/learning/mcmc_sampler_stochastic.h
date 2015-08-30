#ifndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
#define MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__

#include <cmath>

#include <utility>
#include <numeric>
#include <algorithm>  // min, max
#include <chrono>

#include "mcmc/np.h"
#include "mcmc/random.h"
#include "mcmc/timer.h"

#include "mcmc/learning/learner.h"

namespace mcmc {
namespace learning {

#ifdef UNUSED

// EDGEMAP_IS_VECTOR is a more efficient implementation anyway
#ifdef EDGEMAP_IS_VECTOR
typedef std::vector<int> EdgeMapZ;
#else
// typedef std::map<Edge, int>	EdgeMapZ;
typedef std::unordered_map<Edge, int> EdgeMapZ;
#endif
#endif

class MCMCSamplerStochastic : public Learner {
 public:
  /**
  Mini-batch based MCMC sampler for community overlapping problems. Basically,
  given a
  connected graph where each node connects to other nodes, we try to find out
  the
  community information for each node.

  Formally, each node can be belong to multiple communities which we can
  represent it by
  distribution of communities. For instance, if we assume there are total K
  communities
  in the graph, then each node a, is attached to community distribution \pi_{a},
  where
  \pi{a} is K dimensional vector, and \pi_{ai} represents the probability that
  node a
  belongs to community i, and \pi_{a0} + \pi_{a1} +... +\pi_{aK} = 1

  Also, there is another parameters called \beta representing the community
  strength, where
  \beta_{k} is scalar.

  In summary, the model has the parameters:
  Prior: \alpha, \eta
  Parameters: \pi, \beta
  Latent variables: z_ab, z_ba
  Observations: y_ab for every link.

  And our goal is to estimate the posterior given observations and priors:
  p(\pi,\beta | \alpha,\eta, y).

  Because of the intractability, we use MCMC(unbiased) to do approximate
  inference. But
  different from classical MCMC approach, where we use ALL the examples to
  update the
  parameters for each iteration, here we only use mini-batch (subset) of the
  examples.
  This method is great marriage between MCMC and stochastic methods.
  */
  MCMCSamplerStochastic(const Options &args);

  virtual void init();

  virtual ~MCMCSamplerStochastic();

  virtual void run();

 protected:
  void update_beta(const MinibatchSet &mini_batch, double scale);

  void update_phi(Vertex i, const NeighborSet &neighbors
#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
                  ,
                  double eps_t
#endif
                  );

  NeighborSet sample_neighbor_nodes(::size_t sample_size, Vertex nodeId,
                                    Random::Random *rnd);

  MinibatchNodeSet nodes_in_batch(const MinibatchSet &mini_batch) const;

  // replicated in both mcmc_sampler_
  double a;
  double b;
  double c;

  ::size_t num_node_sample;
  ::size_t interval;

  std::vector<std::vector<double> > theta;  // parameterization for \beta
  std::vector<std::vector<double> > phi;    // parameterization for \pi
};

}  // namespace learning
}  // namespace mcmc

#endif  // ndef MCMC_LEARNING_MCMC_SAMPLER_STOCHASTIC_H__
