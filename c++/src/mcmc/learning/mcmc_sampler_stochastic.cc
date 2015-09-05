#include "mcmc/learning/mcmc_sampler_stochastic.h"

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
    mini_batch_size =
        N / 10;  // old default for STRATIFIED_RANDOM_NODE_SAMPLING
  }
  std::cerr << "num_node_sample " << num_node_sample << " a " << a << " b " << b
            << " c " << c << " alpha " << alpha << " eta (" << eta[0] << ","
            << eta[1] << ")" << std::endl;

  // model parameters and re-parameterization
  // since the model parameter - \pi and \beta should stay in the simplex,
  // we need to restrict the sum of probability equals to 1.  The way we
  // restrict this is using re-reparameterization techniques, where we
  // introduce another set of variables, and update them first followed by
  // updating \pi and \beta.
  // parameterization for \beta
  theta = rng_.random(SourceAwareRandom::THETA_INIT)->gamma(eta[0], eta[1], K,
                                                            2);
  // std::cerr << "Ignore eta[] in random.gamma: use 100.0 and 0.01" <<
  // std::endl;
  // theta = rng_.random(SourceAwareRandom::THETA_INIT)->gamma(100.0, 0.01, K, 2);		//

  std::vector<std::vector<double> > temp(theta.size(),
                                         std::vector<double>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<double>(1));

  // parameterization for \pi
  phi = rng_.random(SourceAwareRandom::PHI_INIT)->gamma(1, 1, N, K);
  std::cerr << "Done host random for phi" << std::endl;
#ifndef NDEBUG
  for (auto pph : phi) {
    for (auto ph : pph) {
      assert(ph >= 0.0);
    }
  }
#endif
  pi.resize(phi.size(), std::vector<double>(phi[0].size()));
  np::row_normalize(&pi, phi);

  if (true) {
    std::cout << std::fixed << std::setprecision(12) << "beta[0] " << beta[0]
              << std::endl;
  } else {
    std::cerr << "beta ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << beta[k] << " ";
    }
    std::cerr << std::endl;
  }

  if (false) {
    std::cout << "theta[*][0]: ";
    for (::size_t k = 0; k < K; k++) {
      std::cout << std::fixed << std::setprecision(12) << theta[k][0] << " ";
    }
    std::cout << std::endl;
    std::cout << "theta[*][1]: ";
    for (::size_t k = 0; k < K; k++) {
      std::cout << std::fixed << std::setprecision(12) << theta[k][1] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "phi[0][0] " << phi[0][0] << std::endl;
  if (false) {
    std::cout << "pi[0] ";
    for (::size_t k = 0; k < K; k++) {
      std::cout << pi[0][k] << " ";
    }
    std::cout << std::endl;
  }

  if (true) {
    for (::size_t i = 0; i < 10; i++) {
      std::cerr << "phi[" << i << "]: ";
      for (::size_t k = 0; k < 10; k++) {
        std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
      }
      std::cerr << std::endl;
      std::cerr << "pi[" << i << "]: ";
      for (::size_t k = 0; k < 10; k++) {
        std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
      }
      std::cerr << std::endl;
    }
  }

  std::cerr << "Random seed " << std::hex << "0x" <<
    rng_.random(SourceAwareRandom::GRAPH_INIT)->seed(0) << ",0x" <<
    rng_.random(SourceAwareRandom::GRAPH_INIT)->seed(1) << std::endl;
  std::cerr << std::dec;
  std::cerr << "Done " << __func__ << "()" << std::endl;
}

MCMCSamplerStochastic::~MCMCSamplerStochastic() {
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

  clock_t t1, t2;
  std::vector<double> timings;
  t1 = clock();
  while (step_count < max_iteration && !is_converged()) {
    t_outer.start();
    auto l1 = std::chrono::system_clock::now();
    // if (step_count > 200000){
    // interval = 2;
    //}
    if (step_count % interval == 0) {
      t_perplexity.start();
      double ppx_score = cal_perplexity_held_out();
      t_perplexity.stop();
      std::cout << std::fixed << std::setprecision(12)
                << "step count: " << step_count
                << " perplexity for hold out set: " << ppx_score << std::endl;
      ppxs_held_out.push_back(ppx_score);

      t2 = clock();
      double diff = (double)t2 - (double)t1;
      double seconds = diff / CLOCKS_PER_SEC;
      timings.push_back(seconds);
      iterations.push_back(step_count);
#if 0
      if (ppx_score < 5.0) {
        stepsize_switch = true;
        //print "switching to smaller step size mode!"
      }
#endif
      print_mem_usage(std::cout);
    }

    // write into file
    if (step_count % 2000 == 1) {
      if (false) {
        std::ofstream myfile;
        std::string file_name = "mcmc_stochastic_" + to_string(K) +
                                "_num_nodes_" + to_string(num_node_sample) +
                                "_us_air.txt";
        myfile.open(file_name);
        int size = ppxs_held_out.size();
        for (int i = 0; i < size; i++) {
          // int iteration = i * 100 + 1;
          myfile << iterations[i] << "    " << timings[i] << "    "
                 << ppxs_held_out[i] << "\n";
        }

        myfile.close();
      }
    }

    // print "step: " + str(self._step_count)
    /**
    pr = cProfile.Profile()
    pr.enable()
     */

    // (mini_batch, scale) =
    // self._network.sample_mini_batch(self._mini_batch_size,
    // "stratified-random-node")
    // std::cerr << "Invoke sample_mini_batch" << std::endl;
    t_mini_batch.start();
    EdgeSample edgeSample = network.sample_mini_batch(
        mini_batch_size, strategy::STRATIFIED_RANDOM_NODE);
    t_mini_batch.stop();
    if (false) {
      std::cerr << "Minibatch: ";
      for (auto e : *edgeSample.first) {
        std::cerr << e << " ";
      }
      std::cerr << std::endl;
    }
    // std::cerr << "Done sample_mini_batch" << std::endl;
    const MinibatchSet &mini_batch = *edgeSample.first;
    double scale = edgeSample.second;

    // std::unordered_map<Vertex, std::vector<int> > latent_vars;
    // std::unordered_map<Vertex, ::size_t> size;

    // iterate through each node in the mini batch.
    t_nodes_in_mini_batch.start();
    MinibatchNodeSet nodes = nodes_in_batch(mini_batch);
    t_nodes_in_mini_batch.stop();
// std::cerr << "mini_batch size " << mini_batch.size() << " num_node_sample "
// << num_node_sample << std::endl;

#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
    double eps_t = a * std::pow(1 + step_count / b, -c);  // step size
// double eps_t = std::pow(1024+step_count, -0.5);
#endif

    // ************ do in parallel at each host
    // std::cerr << "Sample neighbor nodes" << std::endl;
    std::vector<Vertex> node_vector(nodes.begin(), nodes.end());
    for (::size_t n = 0; n < node_vector.size(); ++n) {
      Vertex node = node_vector[n];
      t_sample_neighbor_nodes.start();
      // sample a mini-batch of neighbors
      NeighborSet neighbors =
        sample_neighbor_nodes(num_node_sample, node,
                              rng_.random(
                                SourceAwareRandom::NEIGHBOR_SAMPLER));
      t_sample_neighbor_nodes.stop();

      t_update_phi.start();
      update_phi(node, neighbors
#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
                 ,
                 eps_t
#endif
                 );
      t_update_phi.stop();
    }

    // ************ do in parallel at each host
    t_update_pi.start();
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
    np::row_normalize(&pi, phi);  // update pi from phi.
#else
    // No need to update pi where phi is unchanged
    for (auto i : nodes) {
      np::normalize(&pi[i], phi[i]);
    }
#endif
    t_update_pi.stop();

    t_update_beta.start();
    update_beta(mini_batch, scale);
    t_update_beta.stop();

    delete edgeSample.first;

    step_count++;
    t_outer.stop();
    auto l2 = std::chrono::system_clock::now();
    if (false) {
      std::cout << "LOOP  = " << (l2 - l1).count() << std::endl;
    }
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
                                        double scale) {
  std::vector<std::vector<double> > grads(
      K, std::vector<double>(2, 0.0));  // gradients K*2 dimension
  std::vector<double> probs(K);
  // sums = np.sum(self.__theta,1)
  std::vector<double> theta_sum(theta.size());
  std::transform(theta.begin(), theta.end(), theta_sum.begin(),
                 np::sum<double>);

  // update gamma, only update node in the grad
  double eps_t = get_eps_t();
  for (auto edge = mini_batch.begin(); edge != mini_batch.end(); edge++) {
    int y = 0;
    if (edge->in(network.get_linked_edges())) {
      y = 1;
    }
    int i = edge->first;
    int j = edge->second;

    double pi_sum = 0.0;
    for (::size_t k = 0; k < K; k++) {
      pi_sum += pi[i][k] * pi[j][k];
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
      probs[k] = std::pow(beta[k], y) * std::pow(1 - beta[k], 1 - y) *
                 pi[i][k] * pi[j][k];
#else
      double f = pi[i][k] * pi[j][k];
      if (y == 1) {
        probs[k] = beta[k] * f;
      } else {
        probs[k] = (1.0 - beta[k]) * f;
      }
#endif
    }

#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
    double prob_0 =
        std::pow(epsilon, y) * std::pow(1 - epsilon, 1 - y) * (1 - pi_sum);
    double prob_sum = np::sum(probs) + prob_0;
    for (::size_t k = 0; k < K; k++) {
      grads[k][0] += (probs[k] / prob_sum) *
                     (std::abs(1 - y) / theta[k][0] - 1 / theta_sum[k]);
      grads[k][1] += (probs[k] / prob_sum) *
                     (std::abs(-y) / theta[k][1] - 1 / theta_sum[k]);
    }
#else
    double prob_0 = ((y == 1) ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
    double prob_sum = np::sum(probs) + prob_0;
    for (::size_t k = 0; k < K; k++) {
      double f = probs[k] / prob_sum;
      double one_over_theta_sum = 1.0 / theta_sum[k];
      grads[k][0] += f * ((1 - y) / theta[k][0] - one_over_theta_sum);
      grads[k][1] += f * (y / theta[k][1] - one_over_theta_sum);
    }
#endif
  }

  // update theta

  // random noise.
  std::vector<std::vector<double> > noise =
      rng_.random(SourceAwareRandom::BETA_UPDATE)->randn(K, 2);
  // std::vector<std::vector<double> > theta_star(theta);
  for (::size_t k = 0; k < K; k++) {
    for (::size_t i = 0; i < 2; i++) {
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
      theta[k][i] = std::abs(
          theta[k][i] +
          eps_t / 2 * (eta[i] - theta[k][i] + scale * grads[k][i]) +
          std::pow(eps_t, .5) * std::pow(theta[k][i], .5) * noise[k][i]);
#else
      double f = std::sqrt(eps_t * theta[k][i]);
      theta[k][i] =
          std::abs(theta[k][i] +
                   eps_t / 2.0 * (eta[i] - theta[k][i] + scale * grads[k][i]) +
                   f * noise[k][i]);
#endif
    }
  }

  std::vector<std::vector<double> > temp(theta.size(),
                                         std::vector<double>(theta[0].size()));
  np::row_normalize(&temp, theta);
  std::transform(temp.begin(), temp.end(), beta.begin(),
                 np::SelectColumn<double>(1));

  if (false) {
    for (auto n : noise) {
      std::cerr << "noise ";
      for (auto b : n) {
        std::cerr << std::fixed << std::setprecision(12) << b << " ";
      }
      std::cerr << std::endl;
    }
    std::cerr << "beta ";
    for (auto b : beta) {
      std::cerr << std::fixed << std::setprecision(12) << b << " ";
    }
    std::cerr << std::endl;
  }
}

void MCMCSamplerStochastic::update_phi(Vertex i, const NeighborSet &neighbors
#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
                                       ,
                                       double eps_t
#endif
                                       ) {
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
  double eps_t = get_eps_t();
#endif

  if (false) {
    std::cerr << "update_phi pre phi[" << i << "] ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "pi[" << i << "] ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
    }
    std::cerr << std::endl;
    for (auto n : neighbors) {
      std::cerr << "pi[" << n << "] ";
      for (::size_t k = 0; k < K; k++) {
        std::cerr << std::fixed << std::setprecision(12) << pi[n][k] << " ";
      }
      std::cerr << std::endl;
    }
  }

  double phi_i_sum = np::sum(phi[i]);
  std::vector<double> grads(K, 0.0);  // gradient for K classes

  for (auto neighbor : neighbors) {
    if (i == neighbor) {
      continue;
    }

    int y_ab = 0;  // observation
    Edge edge(std::min(i, neighbor), std::max(i, neighbor));
    if (edge.in(network.get_linked_edges())) {
      y_ab = 1;
    }

    std::vector<double> probs(K);
#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
    double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
#endif
    for (::size_t k = 0; k < K; k++) {
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
      probs[k] = std::pow(beta[k], y_ab) * std::pow(1 - beta[k], 1 - y_ab) *
                 pi[i][k] * pi[neighbor][k];
      probs[k] += std::pow(epsilon, y_ab) * std::pow(1 - epsilon, 1 - y_ab) *
                  pi[i][k] * (1 - pi[neighbor][k]);
#else
      double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
      probs[k] = pi[i][k] * (pi[neighbor][k] * f + e);
#endif
    }

    double prob_sum = np::sum(probs);
    for (::size_t k = 0; k < K; k++) {
      grads[k] += (probs[k] / prob_sum) / phi[i][k] - 1.0 / phi_i_sum;
    }
  }

  // random gaussian noise.
  std::vector<double> noise =
    rng_.random(SourceAwareRandom::PHI_UPDATE)->randn(K);
  if (false) {
    for (::size_t k = 0; k < K; ++k) {
      std::cerr << "randn " << std::fixed << std::setprecision(12) << noise[k]
                << std::endl;
    }
  }
#ifndef MCMC_EFFICIENCY_COMPATIBILITY_MODE
  double Nn = (1.0 * N) / num_node_sample;
#endif
  // update phi for node i
  for (::size_t k = 0; k < K; k++) {
#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
    phi[i][k] =
        std::abs(phi[i][k] +
                 eps_t / 2 * (alpha - phi[i][k] +
                              (N * 1.0 / num_node_sample) * grads[k]) +
                 std::pow(eps_t, 0.5) * std::pow(phi[i][k], 0.5) * noise[k]);
#else
    phi[i][k] =
        std::abs(phi[i][k] + eps_t / 2 * (alpha - phi[i][k] + Nn * grads[k]) +
                 sqrt(eps_t * phi[i][k]) * noise[k]);
#endif
  }

  if (false) {
    std::cerr << std::fixed << std::setprecision(12) << "update_phi post Nn "
              << Nn << " phi[" << i << "] ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << phi[i][k] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "pi[" << i << "] ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << pi[i][k] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "grads ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << grads[k] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "noise ";
    for (::size_t k = 0; k < K; k++) {
      std::cerr << std::fixed << std::setprecision(12) << noise[k] << " ";
    }
    std::cerr << std::endl;
  }
}

NeighborSet MCMCSamplerStochastic::sample_neighbor_nodes(::size_t sample_size,
                                                         Vertex nodeId,
                                                         Random::Random *rnd) {
  /**
  Sample subset of neighborhood nodes.
   */
  int p = (int)sample_size;
  NeighborSet neighbor_nodes;
  const EdgeMap &held_out_set = network.get_held_out_set();
  const EdgeMap &test_set = network.get_test_set();

#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE

  while (p > 0) {
#if 1
    auto nodeList = rnd->sample(np::xrange(0, N), sample_size * 2);
#else
    // this optimization is superseeded by re-implementation below
    auto nodeList = rnd->sampleRange(N, sample_size * 2);
#endif

    for (std::vector<Vertex>::const_iterator neighborId = nodeList->begin();
         neighborId != nodeList->end(); neighborId++) {
      if (p < 0) {
        if (p != 0) {
          // std::cerr << __func__ << ": Are you sure p < 0 is a good idea?"
          // << std::endl;
        }
        break;
      }
      if (*neighborId == nodeId) {
        continue;
      }
      // check condition, and insert into mini_batch_set if it is valid.
      Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
      if (edge.in(held_out_set) || edge.in(test_set) ||
          neighbor_nodes.find(*neighborId) != neighbor_nodes.end()
          ) {
        continue;
      } else {
// add it into mini_batch_set
        neighbor_nodes.insert(*neighborId);
        p -= 1;
      }
    }

    delete nodeList;
  }

#else   // def MCMC_EFFICIENCY_COMPATIBILITY_MODE

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

#endif  // def MCMC_EFFICIENCY_COMPATIBILITY_MODE

  if (false) {
    std::cerr << "Node " << nodeId << ": neighbors ";
    for (auto n : neighbor_nodes) {
      std::cerr << n << " ";
    }
    std::cerr << std::endl;
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
