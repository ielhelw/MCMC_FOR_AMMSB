#include "mcmc/learning/mcmc_sampler_stochastic.h"

#include <cmath>
#include <algorithm>  // min, max

namespace mcmc {
namespace learning {

MCMCSamplerStochastic::MCMCSamplerStochastic(const Options &args)
    : Learner(args) {
  // step size parameters.
  this->a = args_.a;
  this->b = args_.b;
  this->c = args_.c;

  // control parameters for learning
  if (args_.interval == 0) {
    interval = 50;
  } else {
    interval = args_.interval;
  }
}

void MCMCSamplerStochastic::init() {
  LoadNetwork();

  // control parameters for learning
  // num_node_sample = static_cast<
  // ::size_t>(std::sqrt(network.get_num_nodes()));
  if (args_.num_node_sample == 0) {
    // TODO: automative update.....
    num_node_sample = N / 50;
  } else {
    num_node_sample = args_.num_node_sample;
  }
  if (args_.mini_batch_size == 0) {
    // old default for STRATIFIED_RANDOM_NODE_SAMPLING
    mini_batch_size = N / 10;
  }
  sampler_stochastic_info(std::cerr);

  // model parameters and re-parameterization
  // since the model parameter - \pi and \beta should stay in the simplex,
  // we need to restrict the sum of probability equals to 1.  The way we
  // restrict this is using re-reparameterization techniques, where we
  // introduce another set of variables, and update them first followed by
  // updating \pi and \beta.
  // parameterization for \beta
  theta = rng_[0]->gamma(eta[0], eta[1], K, 2);
  // std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" <<
  // std::endl;
  // theta = rng_[0]->gamma(100.0, 0.01, K, 2);		//

  std::vector<std::vector<Float> > temp(theta.size(),
                                         std::vector<Float>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<Float>(1));

  // parameterization for \pi
  phi = rng_[0]->gamma(1, 1, N, K);
  std::cerr << "Done host random for phi" << std::endl;
#ifndef NDEBUG
  for (auto pph : phi) {
    for (auto ph : pph) {
      assert(ph >= 0.0);
    }
  }
#endif
  pi.resize(phi.size(), std::vector<Float>(phi[0].size()));
  np::row_normalize(&pi, phi);

  std::cerr << "Done " << __func__ << "()" << std::endl;
}

MCMCSamplerStochastic::~MCMCSamplerStochastic() {
}

void MCMCSamplerStochastic::sampler_stochastic_info(std::ostream &s) {
  s.unsetf(std::ios_base::floatfield);
  s << std::setprecision(6);
  s << "num_node_sample " << num_node_sample << std::endl;
  s << "a " << a << " b " << b << " c " << c;
  s << " eta (" << eta[0] << "," << eta[1] << ")" << std::endl;
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      s << "minibatch size: specified " << mini_batch_size <<
          " from num_pieces (" <<
          network.num_pieces_for_minibatch(mini_batch_size) <<
          ") is " << network.real_minibatch_size(mini_batch_size) << std::endl;
      break;
    case strategy::RANDOM_EDGE:
      s << "minibatch size: " << mini_batch_size << std::endl;
  }
#ifdef MCMC_NO_NOISE
  s << "remove noise to do SungJin's experiment" << std::endl;
#endif
}

void MCMCSamplerStochastic::run() {
  /** run mini-batch based MCMC sampler, based on the sungjin's note */
  timer::Timer t_outer("  outer");
  timer::Timer t_perplexity("  perplexity");
  timer::Timer t_mini_batch("  sample_mini_batch");
  timer::Timer t_nodes_in_mini_batch("  nodes_in_mini_batch");
  timer::Timer t_sample_neighbor_nodes("  sample_neighbor_nodes");
  timer::Timer t_update_phi("  update_phi");
  timer::Timer t_update_pi("  update_pi");
  timer::Timer t_update_beta("  update_beta");
  timer::Timer::setTabular(true);

  using namespace std::chrono;
  t_start_ = system_clock::now();

  clock_t t1, t2;
  std::vector<double> timings;
  t1 = clock();
  while (step_count < max_iteration && !is_converged()) {
    t_outer.start();
    if ((step_count - 1) % interval == 0) {
      t_perplexity.start();
      Float ppx_score = cal_perplexity_held_out();
      t_perplexity.stop();
      auto t_now = system_clock::now();
      auto t_ms = duration_cast<milliseconds>(t_now - t_start_).count();
      std::cout << "average_count is: " << average_count << " ";
      std::cout << std::fixed
                << "step count: " << step_count
                << " time: " << std::setprecision(3) << (t_ms / 1000.0)
                << " perplexity for hold out set: " << std::setprecision(12) <<
                ppx_score << std::endl;
      ppxs_heldout_cb_.push_back(ppx_score);

      t2 = clock();
      double diff = (double)t2 - (double)t1;
      double seconds = diff / CLOCKS_PER_SEC;
      timings.push_back(seconds);
    }
    t_mini_batch.start();
    EdgeSample edgeSample = network.sample_mini_batch(mini_batch_size,
                                                      strategy);
    t_mini_batch.stop();
    const MinibatchSet &mini_batch = *edgeSample.first;
    Float scale = edgeSample.second;

    // iterate through each node in the mini batch.
    t_nodes_in_mini_batch.start();
    MinibatchNodeSet nodes = nodes_in_batch(mini_batch);
    t_nodes_in_mini_batch.stop();

    Float eps_t = get_eps_t();

    // ************ do in parallel at each host
    // std::cerr << "Sample neighbor nodes" << std::endl;
    std::vector<Vertex> node_vector(nodes.begin(), nodes.end());
    for (::size_t n = 0; n < node_vector.size(); ++n) {
      Vertex node = node_vector[n];
      t_sample_neighbor_nodes.start();
      // sample a mini-batch of neighbors
      NeighborSet neighbors = sample_neighbor_nodes(num_node_sample, node,
                                                    rng_[0]);
      t_sample_neighbor_nodes.stop();

      t_update_phi.start();
      update_phi(node, neighbors, eps_t);
      t_update_phi.stop();
    }

    // ************ do in parallel at each host
    t_update_pi.start();
    for (auto i : nodes) {
      np::normalize(&pi[i], phi[i]);
    }
    t_update_pi.stop();

    t_update_beta.start();
    update_beta(mini_batch, scale);
    t_update_beta.stop();

    delete edgeSample.first;

    step_count++;
    t_outer.stop();
  }

  timer::Timer::printHeader(std::cout);
  std::cout << t_outer << std::endl;
  std::cout << t_perplexity << std::endl;
  std::cout << t_mini_batch << std::endl;
  std::cout << t_nodes_in_mini_batch << std::endl;
  std::cout << t_sample_neighbor_nodes << std::endl;
  std::cout << t_update_phi << std::endl;
  std::cout << t_update_pi << std::endl;
  std::cout << t_update_beta << std::endl;
}

void MCMCSamplerStochastic::update_beta(const MinibatchSet &mini_batch,
                                        Float scale) {
  std::vector<std::vector<Float> > grads(
      K, std::vector<Float>(2, FLOAT(0.0)));  // gradients K*2 dimension
  std::vector<Float> probs(K);
  // sums = np.sum(self.__theta,1)
  std::vector<Float> theta_sum(theta.size());
  std::transform(theta.begin(), theta.end(), theta_sum.begin(),
                 np::sum<Float>);

  // update gamma, only update node in the grad
  Float eps_t = get_eps_t();
  for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
    int y = 0;
    if (edge->in(network.get_linked_edges())) {
      y = 1;
    }
    int i = edge->first;
    int j = edge->second;

    Float pi_sum = FLOAT(0.0);
    for (::size_t k = 0; k < K; k++) {
      pi_sum += pi[i][k] * pi[j][k];
      Float f = pi[i][k] * pi[j][k];
      if (y == 1) {
        probs[k] = beta[k] * f;
      } else {
        probs[k] = (FLOAT(1.0) - beta[k]) * f;
      }
    }

    Float prob_0 = ((y == 1) ? epsilon : (FLOAT(1.0) - epsilon)) *
                                          (FLOAT(1.0) - pi_sum);
    Float prob_sum = np::sum(probs) + prob_0;
    for (::size_t k = 0; k < K; k++) {
      Float f = probs[k] / prob_sum;
      Float one_over_theta_sum = FLOAT(1.0) / theta_sum[k];
      grads[k][0] += f * ((FLOAT(1.0) - y) / theta[k][0] - one_over_theta_sum);
      grads[k][1] += f * (y / theta[k][1] - one_over_theta_sum);
    }
  }

  // update theta

  // random noise.
  std::vector<std::vector<Float> > noise = rng_[0]->randn(K, 2);
  // std::vector<std::vector<Float> > theta_star(theta);
  for (::size_t k = 0; k < K; k++) {
    for (::size_t i = 0; i < 2; i++) {
      Float f = std::sqrt(eps_t * theta[k][i]);
      theta[k][i] =
          std::abs(theta[k][i] +
                   eps_t / FLOAT(2.0) * (eta[i] - theta[k][i] +
                                         scale * grads[k][i])
                   + f * noise[k][i]
                   );
      if (theta[k][i] < MCMC_NONZERO_GUARD) {
        theta[k][i] = MCMC_NONZERO_GUARD;
      }
    }
  }

  std::vector<std::vector<Float> > temp(theta.size(),
                                         std::vector<Float>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<Float>(1));
}

void MCMCSamplerStochastic::update_phi(Vertex i, const NeighborSet &neighbors,
                                       Float eps_t) {
  Float phi_i_sum = np::sum(phi[i]);
  std::vector<Float> grads(K, FLOAT(0.0));  // gradient for K classes

  for (auto neighbor : neighbors) {
    if (i == neighbor) {
      continue;
    }

    int y_ab = 0;  // observation
    Edge edge(std::min(i, neighbor), std::max(i, neighbor));
    if (edge.in(network.get_linked_edges())) {
      y_ab = 1;
    }

    std::vector<Float> probs(K);
    Float e = (y_ab == 1) ? epsilon : (FLOAT(1.0) - epsilon);
    for (::size_t k = 0; k < K; k++) {
      Float f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
      probs[k] = pi[i][k] * (pi[neighbor][k] * f + e);
    }

    Float prob_sum = np::sum(probs);
    for (::size_t k = 0; k < K; k++) {
      grads[k] += (probs[k] / prob_sum) / phi[i][k] - FLOAT(1.0) / phi_i_sum;
    }
  }

  // random gaussian noise.
  std::vector<Float> noise = rng_[0]->randn(K);
  Float Nn = (FLOAT(1.0) * N) / num_node_sample;
  // update phi for node i
  for (::size_t k = 0; k < K; k++) {
    phi[i][k] =
        std::abs((phi[i][k]
                 + eps_t / FLOAT(2.0) * (alpha - phi[i][k] + Nn * grads[k]))
                 + std::sqrt(eps_t * phi[i][k]) * noise[k]
                 );
    if (phi[i][k] < MCMC_NONZERO_GUARD) {
      phi[i][k] = MCMC_NONZERO_GUARD;
    }
  }

}

NeighborSet MCMCSamplerStochastic::sample_neighbor_nodes(::size_t sample_size,
                                                         Vertex nodeId,
                                                         Random::Random* rnd) {
  /**
  Sample subset of neighborhood nodes.
   */
  int p = (int)sample_size;
  NeighborSet neighbor_nodes;
  const EdgeMap &held_out_set = network.get_held_out_set();
  const EdgeMap &test_set = network.get_test_set();

  for (int i = 0; i <= p; ++i) {
    Vertex neighborId;
    Edge edge(0, 0);
    do {
      neighborId = rnd->randint(0, N - 1);
      edge = Edge(std::min(nodeId, neighborId), std::max(nodeId, neighborId));
    } while (neighborId == nodeId || edge.in(held_out_set) || edge.in(test_set)
             || neighbor_nodes.find(neighborId) != neighbor_nodes.end()
             );
    neighbor_nodes.insert(neighborId);
  }

  return neighbor_nodes;
}

MinibatchNodeSet MCMCSamplerStochastic::nodes_in_batch(
    const MinibatchSet &mini_batch) const {
  /**
  Get all the unique nodes in the mini_batch.
   */
  MinibatchNodeSet node_set;
  for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
    node_set.insert(edge->first);
    node_set.insert(edge->second);
  }

  return node_set;
}

}  // namespace learning
}  // namesapce mcmc
