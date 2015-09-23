#ifndef MCMC_LEARNING_LEARNER_H__
#define MCMC_LEARNING_LEARNER_H__

#include <cmath>

#include "mcmc/config.h"

#include "mcmc/types.h"
#include "mcmc/options.h"
#include "mcmc/network.h"
#include "mcmc/source-aware-random.h"
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

  double cal_perplexity_held_out();

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
  double cal_perplexity(const EdgeMap &data);

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
  double cal_edge_likelihood(const T &pi_a, const T &pi_b, bool y,
                             const std::vector<double> &beta) const {
    double s = 0.0;
#define DONT_FOLD_Y
#ifdef DONT_FOLD_Y
    if (y) {
      for (::size_t k = 0; k < K; k++) {
        s += pi_a[k] * pi_b[k] * beta[k];
      }
    } else {
      double sum = 0.0;
      for (::size_t k = 0; k < K; k++) {
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
        s += pi_a[k] * pi_b[k] * (1.0 - beta[k]);
        sum += pi_a[k] * pi_b[k];
#else
        double f = pi_a[k] * pi_b[k];
        assert(!std::isnan(f));
        s += f * (1.0 - beta[k]);
        sum += f;
        assert(!std::isnan(s));
#endif
        assert(!std::isnan(sum));
      }
      s += (1.0 - sum) * (1.0 - epsilon);
    }
#else
    int iy = y ? 1 : 0;
    int y2_1 = 2 * iy - 1;
    int y_1 = iy - 1;
    double sum = 0.0;
    for (::size_t k = 0; k < K; k++) {
      double f = pi_a[k] * pi_b[k];
      sum += f;
      s += f * (beta[k] * y2_1 - y_1);
    }
    if (!y) {
      s += (1.0 - sum) * (1.0 - epsilon);
    }
#endif

    if (s < 1.0e-30) {
      s = 1.0e-30;
    }

    return s;
  }

  const Options args_;
  Network network;

  double alpha;
  std::vector<double> eta;
  ::size_t K;
  double epsilon;
  ::size_t N;

  std::vector<double> beta;
  std::vector<std::vector<double> > pi;

  ::size_t mini_batch_size;
  double link_ratio;

  ::size_t step_count;

  std::vector<double> ppxs_held_out;

  ::size_t max_iteration;

  double CONVERGENCE_THRESHOLD;

  bool stepsize_switch;
  ::size_t average_count;

  strategy::strategy strategy;

  SourceAwareRandom rng_;

#ifdef MCMC_RANDOM_COMPATIBILITY_MODE
  const bool RANDOM_PRESERVE_RANGE_ORDER = true;
#else
  const bool RANDOM_PRESERVE_RANGE_ORDER = false;
#endif

 private:
  // Used to calculate perplexity per edge in the held-out set.
  std::vector<double> ppx_per_heldout_edge_;
};

}  // namespace learning
}  // namespace mcmc

#endif  // ndef MCMC_LEARNING_LEARNER_H__
