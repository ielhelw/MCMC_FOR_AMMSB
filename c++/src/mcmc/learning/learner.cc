#include "mcmc/learning/learner.h"

#include "mcmc/np.h"            // Only for omp_get_max_threads() report :-(

namespace mcmc {
namespace learning {

Learner::Learner(const Options &args) : args_(args) {
  // model priors
  alpha = args_.alpha;
  eta.resize(2);
  eta[0] = args_.eta0;
  eta[1] = args_.eta1;
  average_count = 1;

  // parameters related to control model
  K = args_.K;
  epsilon = args_.epsilon;

  // check the number of iterations.
  step_count = 1;

  max_iteration = args_.max_iteration;
  CONVERGENCE_THRESHOLD = 0.000000000001;

  stepsize_switch = false;

  rng_.Init(args_.random_seed);

  if (args_.strategy == "unspecified") {
    strategy = strategy::STRATIFIED_RANDOM_NODE;
  } else if (args_.strategy == "stratified-random-node") {
    strategy = strategy::STRATIFIED_RANDOM_NODE;
  } else {
    throw MCMCException("Unknown strategy type: " + args_.strategy);
  }
}

void Learner::LoadNetwork(int world_rank) {
  double held_out_ratio = args_.held_out_ratio;
  if (args_.held_out_ratio == 0.0) {
    held_out_ratio = 0.01;
    std::cerr << "Set held_out_ratio to default " << held_out_ratio
              << std::endl;
  }
  network.Init(args_, held_out_ratio, &rng_, world_rank);

  // parameters related to network
  N = network.get_num_nodes();

  // model parameters to learn
  beta = std::vector<double>(K, 0.0);
  pi = std::vector<std::vector<double> >(N, std::vector<double>(K, 0.0));

  // parameters related to sampling
  mini_batch_size = args_.mini_batch_size;
  if (mini_batch_size < 1) {
    mini_batch_size = N / 2;  // default option.
  }

  // ration between link edges and non-link edges
  link_ratio = network.get_num_linked_edges() / ((N * (N - 1)) / 2.0);
  // store perplexity for all the iterations
  // ppxs_held_out = [];
  // ppxs_test = [];

  ppx_for_heldout = std::vector<double>(network.get_held_out_size(), 0.0);

  info(std::cerr);
}

Learner::~Learner() {}

void Learner::info(std::ostream &s) {
  s << "N " << N;
  s << " E " << network.get_num_linked_edges();
  s << " K " << K;
  s << " iterations " << max_iteration;
  s << " minibatch size " << mini_batch_size;
  s << " link ratio " << link_ratio;
  s << " convergence " << CONVERGENCE_THRESHOLD;
  s << std::endl;
  s << "omp max threads " << omp_get_max_threads() << std::endl;
}

const std::vector<double> &Learner::get_ppxs_held_out() const {
  return ppxs_held_out;
}

const std::vector<double> &Learner::get_ppxs_test() const { return ppxs_test; }

void Learner::set_max_iteration(::size_t max_iteration) {
  this->max_iteration = max_iteration;
}

double Learner::cal_perplexity_held_out() {
  return cal_perplexity(network.get_held_out_set());
}

double Learner::cal_perplexity_test() {
  return cal_perplexity(network.get_test_set());
}

bool Learner::is_converged() const {
  ::size_t n = ppxs_held_out.size();
  if (n < 2) {
    return false;
  }
  if (std::abs(ppxs_held_out[n - 1] - ppxs_held_out[n - 2]) /
          ppxs_held_out[n - 2] >
      CONVERGENCE_THRESHOLD) {
    return false;
  }

  return true;
}

double Learner::cal_perplexity(const EdgeMap &data) {
  double link_likelihood = 0.0;
  double non_link_likelihood = 0.0;
  ::size_t link_count = 0;
  ::size_t non_link_count = 0;

  ::size_t i = 0;
  for (EdgeMap::const_iterator edge = data.begin(); edge != data.end();
       edge++) {
    const Edge &e = edge->first;
    double edge_likelihood =
        cal_edge_likelihood(pi[e.first], pi[e.second], edge->second, beta);
    if (std::isnan(edge_likelihood)) {
      std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
    }

    ppx_for_heldout[i] =
        (ppx_for_heldout[i] * (average_count - 1) + edge_likelihood) /
        (average_count);
    if (edge->second) {
      link_count++;
      link_likelihood += std::log(ppx_for_heldout[i]);

      if (std::isnan(link_likelihood)) {
        std::cerr << "link_likelihood is NaN; potential bug" << std::endl;
      }
    } else {
      assert(!present(network.get_linked_edges(), e));
      non_link_count++;
      non_link_likelihood += std::log(ppx_for_heldout[i]);
      if (std::isnan(non_link_likelihood)) {
        std::cerr << "non_link_likelihood is NaN; potential bug" << std::endl;
      }
    }
    i++;
  }
  double avg_likelihood = 0.0;
  if (link_count + non_link_count != 0) {
    avg_likelihood =
        (link_likelihood + non_link_likelihood) / (link_count + non_link_count);
  }
  if (true) {
    double avg_likelihood1 =
        link_ratio * (link_likelihood / link_count) +
        (1.0 - link_ratio) * (non_link_likelihood / non_link_count);
    std::cerr << std::fixed << std::setprecision(12) << avg_likelihood << " "
              << (link_likelihood / link_count) << " " << link_count << " "
              << (non_link_likelihood / non_link_count) << " " << non_link_count
              << " " << avg_likelihood1 << std::endl;
  }

  average_count = average_count + 1;
  std::cout << "average_count is: " << average_count << " ";
  return (-avg_likelihood);
}

}  // namespace learning
}  // namespace mcmc
