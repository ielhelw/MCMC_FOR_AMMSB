#ifndef MCMC_LEARNING_LEARNER_H__
#define MCMC_LEARNING_LEARNER_H__

#include <cmath>

#include <boost/circular_buffer.hpp>

#include "mcmc/config.h"

#include "mcmc/types.h"
#include "mcmc/options.h"
#include "mcmc/network.h"
#include "mcmc/preprocess/data_factory.h"


namespace mcmc {
namespace learning {

/**
 * This is base class for all concrete learners, including MCMC sampler,
 * variational
 * inference,etc.
 */
class Learner {
 public:
  const Float MCMC_NONZERO_GUARD = FLOAT(1.0e-24);

  Learner(const Options &args);

  void LoadNetwork(int world_rank = 0, bool allocate_pi = true);

  virtual ~Learner();

  /**
   * Each concrete learner should implement this. It basically
   * iterate the data sets, then update the model parameters, until
   * convergence. The convergence can be measured by perplexity score.
   *
   * We currently support four different learners:
   * 1. MCMC for batch learning
   * 2. MCMC for mini-batch training
   * 3. Variational inference for batch learning
   * 4. Stochastic variational inference
   */
  virtual void run() = 0;

 protected:
  void info(std::ostream &s);

  void set_max_iteration(::size_t max_iteration);

  Float cal_perplexity_held_out();

  bool is_converged() const;

  /**
   * calculate the perplexity for data.
   * perplexity defines as exponential of negative average log likelihood.
   * formally:
   *     ppx = exp(-1/N * \sum){i}^{N}log p(y))
   *
   * we calculate average log likelihood for link and non-link separately, with
   *the
   * purpose of weighting each part proportionally. (the reason is that we
   *sample
   * the equal number of link edges and non-link edges for held out data and
   *test data,
   * which is not true representation of actual data set, which is extremely
   *sparse.
   */
  Float cal_perplexity(const EdgeMap &data);

  template <typename T>
  static void dump(const std::vector<T> &a, ::size_t n,
                   const std::string &name = "") {
    n = std::min(n, a.size());
    std::cerr << name;
    if (n != a.size()) {
      std::cerr << "[0:" << n << "]";
    }
    std::cerr << " ";
    for (auto i = a.begin(); i < a.begin() + n; i++) {
      std::cerr << std::fixed << std::setprecision(12) << *i << " ";
    }
    std::cerr << std::endl;
  }

  /**
   * calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
   * in order to calculate this, we need to sum over all the possible (z_ab,
   * z_ba)
   * such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
   * but this calculation can be done in O(K), by using some trick.
   */
  template <typename T>
  Float cal_edge_likelihood(const T &pi_a, const T &pi_b, bool y,
                             const std::vector<Float> &beta) const {
    Float s = FLOAT(0.0);
#define DONT_FOLD_Y
#ifdef DONT_FOLD_Y
    if (y) {
      for (::size_t k = 0; k < K; k++) {
        s += pi_a[k] * pi_b[k] * beta[k];
      }
    } else {
      Float sum = FLOAT(0.0);
      for (::size_t k = 0; k < K; k++) {
        Float f = pi_a[k] * pi_b[k];
        s += f * (FLOAT(1.0) - beta[k]);
        sum += f;
      }
      s += (FLOAT(1.0) - sum) * (FLOAT(1.0) - epsilon);
    }
#else   // def DONT_FOLD_Y
    int iy = y ? 1 : 0;
    int y2_1 = 2 * iy - 1;
    int y_1 = iy - 1;
    Float sum = FLOAT(0.0);
    for (::size_t k = 0; k < K; k++) {
      Float f = pi_a[k] * pi_b[k];
      sum += f;
      s += f * (beta[k] * y2_1 - y_1);
    }
    if (!y) {
      s += (FLOAT(1.0) - sum) * (FLOAT(1.0) - epsilon);
    }
#endif  // def DONT_FOLD_Y

    if (s < FLOAT(1.0e-30)) {
      s = FLOAT(1.0e-30);
    }

    return s;
  }

  const Options args_;
  Network network;

  Float alpha;
  std::vector<Float> eta;
  ::size_t K;
  Float epsilon;
  ::size_t N;

  std::vector<Float> beta;
  std::vector<std::vector<Float> > pi;

  ::size_t mini_batch_size;
  Float link_ratio;

  ::size_t step_count;

  boost::circular_buffer<Float> ppxs_heldout_cb_;
  // Used to calculate perplexity per edge in the held-out set.
  std::vector<Float> ppx_per_heldout_edge_;

  ::size_t max_iteration;

  Float CONVERGENCE_THRESHOLD;

  bool stepsize_switch;
  ::size_t average_count;

  strategy::strategy strategy;

  std::vector<Random::Random*> rng_;
};

}  // namespace learning
}  // namespace mcmc

#endif  // ndef MCMC_LEARNING_LEARNER_H__
