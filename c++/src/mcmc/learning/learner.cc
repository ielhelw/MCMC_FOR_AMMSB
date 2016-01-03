#include "mcmc/learning/learner.h"

#include <boost/asio.hpp>

#include "mcmc/np.h"            // Only for omp_get_max_threads() report :-(

namespace mcmc {
namespace learning {

Learner::Learner(const Options &args) : args_(args), ppxs_heldout_cb_(10) {
  std::cerr << "Floating point precision: " << (sizeof(Float) * CHAR_BIT) <<
    "bit" << std::endl;

  std::string hostname = boost::asio::ip::host_name();
  std::cerr << "Host " << hostname << " PID " << getpid() << std::endl;

  std::cerr << "Build type " <<
#ifdef NDEBUG
    "Release"
#else
    "Debug"
#endif
    << std::endl;
  std::cerr << "Graph implementation: " <<
    "adjacency list (google sparseset)"
    << std::endl;

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
  CONVERGENCE_THRESHOLD = args_.convergence_threshold;

  stepsize_switch = false;

  strategy = args_.strategy;
}

void Learner::InitRandom(::size_t world_rank) {
  std::cerr << "Create per-thread randoms" << std::endl;
  rng_.resize(omp_get_max_threads());
  int seed;
  seed = args_.random_seed;
  for (::size_t i = 0; i < rng_.size(); ++i) {
    int my_seed = seed + 1 + i + world_rank * rng_.size();
    rng_[i] = new Random::Random(my_seed, seed, false);
  }
  std::cerr << "Random seed[0] " << std::hex << "0x" << rng_[0]->seed(0) <<
    ",0x" << rng_[0]->seed(1) << std::endl;
  std::cerr << std::dec;
}

void Learner::Init(bool allocate_pi) {
  // parameters related to network
  N = network.get_num_nodes();

  // model parameters to learn
  beta = std::vector<Float>(K, FLOAT(0.0));
  if (allocate_pi) {
    pi = std::vector<std::vector<Float> >(N, std::vector<Float>(K, FLOAT(0.0)));
  }

  // parameters related to sampling
  mini_batch_size = args_.mini_batch_size;
  if (mini_batch_size < 1) {
    mini_batch_size = N / 2;  // default option.
  }

  // ratio between link edges and non-link edges
  link_ratio = network.get_num_linked_edges() / ((N * (double)(N - 1)) / 2.0);

  ppx_per_heldout_edge_ = std::vector<Float>(network.get_held_out_size(),
                                             FLOAT(0.0));

  info(std::cerr);
}

void Learner::LoadNetwork(::size_t world_rank, bool allocate_pi) {
  Float held_out_ratio = args_.held_out_ratio;
  if (args_.held_out_ratio == 0.0) {
    held_out_ratio = 0.1;
    std::cerr << "Set held_out_ratio to default " << held_out_ratio
              << std::endl;
  }

  InitRandom(world_rank);
  network.Init(args_, held_out_ratio, &rng_);
  Init(allocate_pi);
}

Learner::~Learner() {
  for (auto r : rng_) {
    delete r;
  }
}

void Learner::info(std::ostream &s) {
  s.unsetf(std::ios_base::floatfield);
  s << std::setprecision(6);
  s << "N " << N;
  s << " E " << network.get_num_linked_edges();
  s << " link ratio " << link_ratio;
  s << " K " << K << std::endl;
  s << "minibatch size " << mini_batch_size;
  s << " epsilon " << epsilon;
  s << " alpha " << alpha;
  s << " iterations " << max_iteration;
  s << " convergence " << CONVERGENCE_THRESHOLD;
  s << std::endl;
  s << "sampling strategy " << strategy << std::endl;
  s << "omp max threads " << omp_get_max_threads() << std::endl;
}

void Learner::set_max_iteration(::size_t max_iteration) {
  this->max_iteration = max_iteration;
}

Float Learner::cal_perplexity_held_out() {
  return cal_perplexity(network.get_held_out_set());
}

bool Learner::is_converged() const {
  ::size_t n = ppxs_heldout_cb_.size();
  if (n < 2) return false;
  return std::abs(ppxs_heldout_cb_[n - 1] - ppxs_heldout_cb_[n - 2]) /
             ppxs_heldout_cb_[n - 2] <=
         CONVERGENCE_THRESHOLD;
}

Float Learner::cal_perplexity(const EdgeMap &data) {
  Float link_likelihood = FLOAT(0.0);
  Float non_link_likelihood = FLOAT(0.0);
  ::size_t link_count = 0;
  ::size_t non_link_count = 0;

  ::size_t i = 0;
  for (EdgeMap::const_iterator edge = data.begin(); edge != data.end();
       edge++) {
    const Edge &e = edge->first;
    Float edge_likelihood =
        cal_edge_likelihood(pi[e.first], pi[e.second], edge->second, beta);
    if (std::isnan(edge_likelihood)) {
      std::cerr << "edge_likelihood is NaN; potential bug" << std::endl;
    }

    ppx_per_heldout_edge_[i] =
        (ppx_per_heldout_edge_[i] * (average_count - 1) + edge_likelihood) /
        (average_count);
    if (edge->second) {
      link_count++;
      link_likelihood += std::log(ppx_per_heldout_edge_[i]);

      if (std::isnan(link_likelihood)) {
        std::cerr << "link_likelihood is NaN; potential bug" << std::endl;
      }
    } else {
      assert(! e.in(network.get_linked_edges()));
      non_link_count++;
      non_link_likelihood += std::log(ppx_per_heldout_edge_[i]);
      if (std::isnan(non_link_likelihood)) {
        std::cerr << "non_link_likelihood is NaN; potential bug" << std::endl;
      }
    }
    i++;
  }
  Float avg_likelihood = FLOAT(0.0);
  if (link_count + non_link_count != 0) {
    avg_likelihood =
        (link_likelihood + non_link_likelihood) / (link_count + non_link_count);
  }

  average_count = average_count + 1;

  return (-avg_likelihood);
}

}  // namespace learning
}  // namespace mcmc
