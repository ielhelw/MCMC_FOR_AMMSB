#include "mcmc/network.h"

namespace mcmc {

Network::Network() {}

void Network::Init(const Data *data, double held_out_ratio) {
  progress = 1 << 20;  // FIXME: make this a parameter

  N = data->N;             // number of nodes in the graph
  linked_edges = data->E;  // all pair of linked edges.
#ifdef EDGESET_IS_ADJACENCY_LIST
  adjacency_list_init();
  num_total_edges = cumulative_edges[N - 1] / 2;  // number of undirected edges.
#else
  num_total_edges = get_num_linked_edges();  // number of total edges.
#endif
  this->held_out_ratio = held_out_ratio;  // percentage of held-out data size

  // Based on the a-MMSB paper, it samples equal number of
  // linked edges and non-linked edges.
  held_out_size = held_out_ratio * get_num_linked_edges();

  // initialize train_link_map
  init_train_link_map();
  // randomly sample hold-out and test sets.
  init_held_out_set();
  init_test_set();

  calc_max_fan_out();
}

Network::~Network() {
#ifdef EDGESET_IS_ADJACENCY_LIST
  adjacency_list_end();
#endif
}

EdgeSample Network::sample_mini_batch(::size_t mini_batch_size,
                                      strategy::strategy strategy) const {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE: {
      ::size_t num_pieces = (N + mini_batch_size - 1) / mini_batch_size;
      // std::cerr << "Set stratified random node sampling divisor to " <<
      // num_pieces << std::endl;
      return stratified_random_node_sampling(num_pieces);
    }
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::minibatch_nodes_for_strategy(
    ::size_t mini_batch_size, strategy::strategy strategy) const {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      return minibatch_edges_for_strategy(mini_batch_size, strategy) + 1;
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::minibatch_edges_for_strategy(
    ::size_t mini_batch_size, strategy::strategy strategy) const {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      return std::max(mini_batch_size + 1, get_max_fan_out(1));
    // return fan_out_cumul_distro[mini_batch_size];
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::get_num_linked_edges() const {
#ifdef EDGESET_IS_ADJACENCY_LIST
  return cumulative_edges[N - 1] / 2;  // number of undirected edges.
#else
  return linked_edges->size();
#endif
}

::size_t Network::get_num_total_edges() const { return num_total_edges; }

::size_t Network::get_held_out_size() const { return held_out_size; }

int Network::get_num_nodes() const { return N; }

const NetworkGraph &Network::get_linked_edges() const { return *linked_edges; }

const EdgeMap &Network::get_held_out_set() const { return held_out_map; }

const EdgeMap &Network::get_test_set() const { return test_map; }

void Network::set_num_pieces(::size_t num_pieces) {
  this->num_pieces = num_pieces;
}

EdgeSample Network::stratified_random_node_sampling(::size_t num_pieces) const {
  while (true) {
    // randomly select the node ID
    int nodeId = Random::random->randint(0, N - 1);
    // decide to sample links or non-links
    int flag = Random::random->randint(
        0, 1);  // flag=0: non-link edges  flag=1: link edges
    // std::cerr << "num_pieces " << num_pieces << " flag " << flag <<
    // std::endl;

    MinibatchSet *mini_batch_set = new MinibatchSet();

    if (flag == 0) {
      /* sample non-link edges */
      // this is approximation, since the size of self.train_link_map[nodeId]
      // greatly smaller than N.
      // ::size_t mini_batch_size = (int)((N - train_link_map[nodeId].size())
      // / num_pieces);
      ::size_t mini_batch_size = (int)(N / num_pieces);
      int p = (int)mini_batch_size;

      while (p > 0) {
// because of the sparsity, when we sample $mini_batch_size*2$ nodes, the list
// likely
// contains at least mini_batch_size valid nodes.
#ifdef EFFICIENCY_FOLLOWS_PYTHON
        auto nodeList =
            Random::random->sample(np::xrange(0, N), mini_batch_size * 2);
#else
        auto nodeList = Random::random->sampleRange(N, mini_batch_size * 2);
#endif
        for (std::vector<int>::iterator neighborId = nodeList->begin();
             neighborId != nodeList->end(); neighborId++) {
          // std::cerr << "random neighbor " << *neighborId << std::endl;
          if (p < 0) {
            // std::cerr << __func__ << ": Are you sure p < 0 is a good idea?"
            // << std::endl;
            break;
          }
          if (*neighborId == nodeId) {
            continue;
          }

          // check condition, and insert into mini_batch_set if it is valid.
          Edge edge(std::min(nodeId, *neighborId),
                    std::max(nodeId, *neighborId));
          if (edge.in(*linked_edges) || edge.in(held_out_map) ||
              edge.in(test_map) || edge.in(*mini_batch_set)) {
            continue;
          }

          edge.insertMe(mini_batch_set);
          p--;
        }

        delete nodeList;
      }
      return EdgeSample(mini_batch_set, N * num_pieces);

    } else {
/* sample linked edges */
// return all linked edges
#ifdef EDGESET_IS_ADJACENCY_LIST
      for (auto neighborId : (*linked_edges)[nodeId]) {
        Edge e(std::min(nodeId, neighborId), std::max(nodeId, neighborId));
        if (!e.in(test_map) && !e.in(held_out_map)) {
          e.insertMe(mini_batch_set);
        }
      }
#else
      for (VertexSet::const_iterator neighborId =
               train_link_map[nodeId].begin();
           neighborId != train_link_map[nodeId].end(); neighborId++) {
        Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
        edge.insertMe(mini_batch_set);
      }
#endif

      return EdgeSample(mini_batch_set, N);
    }
  }
}

void Network::init_train_link_map() {
#ifdef EDGESET_IS_ADJACENCY_LIST
  std::cerr << "train_link_map is gone; calculate membership for each edge "
               "as linked_edges - held_out_map - test_map" << std::endl;
#else
  train_link_map = std::vector<VertexSet>(N);
  for (auto edge = linked_edges->begin(); edge != linked_edges->end(); edge++) {
    train_link_map[edge->first].insert(edge->second);
    train_link_map[edge->second].insert(edge->first);
  }
#endif
}

#ifdef EDGESET_IS_ADJACENCY_LIST

void Network::adjacency_list_init() {
  cumulative_edges.resize(N);

  if (linked_edges->size() != static_cast<::size_t>(N)) {
    throw MCMCException("AdjList size and/or cumul size corrupt");
  }

#pragma omp parallel for
  for (int32_t i = 0; i < N; ++i) {
    cumulative_edges[i] = (*linked_edges)[i].size();
  }
  np::prefix_sum(&cumulative_edges);
  std::cerr
      << "Found prefix sum for edges in AdjacencyList graph: top bucket edge "
      << cumulative_edges[cumulative_edges.size() - 1] << " max edge "
      << (cumulative_edges[cumulative_edges.size() - 1] +
          (*linked_edges)[cumulative_edges.size() - 1].size()) << std::endl;

  std::cerr << "Initializing the held-out set on multiple machines will "
               "create randomness bugs" << std::endl;
  thread_random.resize(omp_get_max_threads());
  for (::size_t i = 0; i < thread_random.size(); ++i) {
    thread_random[i] = new Random::Random(i, 47);
  }
}

void Network::adjacency_list_end() {
  for (auto r : thread_random) {
    delete r;
  }
}

void Network::sample_random_edges(const NetworkGraph *linked_edges, ::size_t p,
                                  std::vector<Edge> *edges) {
  std::unordered_set<Edge> collector;

  // The graph has a hit for (a,b) as well as (b,a). Therefore it contains
  // duplicates. Trust on unordered_set to filter out the duplicates. We need
  // to
  // iterate until the requested number of edges has been inserted though.
  while (collector.size() < p) {
    std::vector<std::unordered_set<Edge>> thread_edges(omp_get_max_threads());

#pragma omp parallel for
    for (::size_t i = 0; i < p - collector.size(); ++i) {
      // Draw from 2 |Edges| since each edge is represented twice
      ::size_t edge_index =
          thread_random[omp_get_thread_num()]->randint(0, 2 * num_total_edges);
      // locate the vertex where this edge lives
      // Actually search for find_ge, so do edge_index + 1
      int v1 = np::find_le(cumulative_edges, edge_index + 1);
      if (v1 == -1) {
        throw MCMCException(
            "Cannot find edge " + std::to_string(edge_index) + " max is " +
            std::to_string(cumulative_edges[cumulative_edges.size() - 1]));
      }
      if (v1 != 0) {
        edge_index -= cumulative_edges[v1 - 1];
      }
      int v2 = -1;
      // draw edge_index'th neighbor within this edge list
      if ((::size_t)v1 >= linked_edges->size()) {
        std::cerr << "OOOOOOOPPPPPPPPPPPPPSSSSSSSS outside vector" << std::endl;
      }
      if ((*linked_edges)[v1].size() <= 0) {
        std::cerr << "OOOOPPPPPPPPPPSSSSSSSSSSSS empty elt" << std::endl;
      }
      for (auto n : (*linked_edges)[v1]) {
        if (edge_index == 0) {
          v2 = n;
          break;
        }
        edge_index--;
      }
      assert(v2 >= 0);
      Edge e(std::min(v1, v2), std::max(v1, v2));
      thread_edges[omp_get_thread_num()].insert(e);
    }

    for (auto e : thread_edges) {
      collector.insert(e.begin(), e.end());
    }
  }

  edges->assign(collector.begin(), collector.end());
}

#endif  // def EDGESET_IS_ADJACENCY_LIST

void Network::init_held_out_set() {
  ::size_t p = held_out_size / 2;

  // Sample p linked-edges from the network.
  if (get_num_linked_edges() < p) {
    throw MCMCException(
        "There are not enough linked edges that can sample from. "
        "please use smaller held out ratio.");
  }

  // FIXME make sampled_linked_edges an out param
  print_mem_usage(std::cerr);
#if defined RANDOM_FOLLOWS_CPP_WENZHE || defined RANDOM_FOLLOWS_PYTHON
  std::cerr << __func__
            << ": FIXME: replace EdgeList w/ (unordered) EdgeSet again"
            << std::endl;
  auto sampled_linked_edges = Random::random->sampleList(linked_edges, p);
#elif defined EDGESET_IS_ADJACENCY_LIST
  std::vector<Edge> *sampled_linked_edges = new std::vector<Edge>();
  sample_random_edges(linked_edges, p, sampled_linked_edges);
#else
  auto sampled_linked_edges = Random::random->sample(linked_edges, p);
#endif
  ::size_t count = 0;
  for (auto edge = sampled_linked_edges->begin();
       edge != sampled_linked_edges->end(); edge++) {
    held_out_map[*edge] = true;
#ifndef EDGESET_IS_ADJACENCY_LIST
    train_link_map[edge->first].erase(edge->second);
    train_link_map[edge->second].erase(edge->first);
#endif
    if (progress != 0 && count % progress == 0) {
      std::cerr << "Edges/in in held-out set " << count << std::endl;
      print_mem_usage(std::cerr);
    }
    count++;
  }

  delete sampled_linked_edges;

  // sample p non-linked edges from the network
  while (p > 0) {
    Edge edge = sample_non_link_edge_for_held_out();
    held_out_map[edge] = false;
    p--;
    if (progress != 0 && count % progress == 0) {
      std::cerr << "Edges/out in held-out set " << count << std::endl;
      print_mem_usage(std::cerr);
    }
    count++;
  }
}

void Network::init_test_set() {
  int p = (int)(held_out_size / 2);
  // sample p linked edges from the network
  ::size_t count = 0;
  while (p > 0) {
// Because we already used some of the linked edges for held_out sets,
// here we sample twice as much as links, and select among them, which
// is likely to contain valid p linked edges.
#if defined RANDOM_FOLLOWS_CPP_WENZHE || defined RANDOM_FOLLOWS_PYTHON
    std::cerr << __func__
              << ": FIXME: replace EdgeList w/ (unordered) EdgeSet again"
              << std::endl;
    // FIXME make sampled_linked_edges an out param
    auto sampled_linked_edges = Random::random->sampleList(linked_edges, 2 * p);
#elif defined EDGESET_IS_ADJACENCY_LIST
    std::vector<Edge> *sampled_linked_edges = new std::vector<Edge>();
    sample_random_edges(linked_edges, 2 * p, sampled_linked_edges);
#else
    // FIXME make sampled_linked_edges an out param
    auto sampled_linked_edges = Random::random->sample(linked_edges, 2 * p);
#endif
    for (auto edge = sampled_linked_edges->cbegin();
         edge != sampled_linked_edges->cend(); edge++) {
      if (p < 0) {
        // std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" <<
        // std::endl;
        break;
      }

      // check whether it is already used in hold_out set
      if (edge->in(held_out_map) || edge->in(test_map)) {
        continue;
      }

      test_map[*edge] = true;
#ifndef EDGESET_IS_ADJACENCY_LIST
      train_link_map[edge->first].erase(edge->second);
      train_link_map[edge->second].erase(edge->first);
#endif
      p--;
      if (progress != 0 && count % progress == 0) {
        std::cerr << "Edges/in in test set " << count << std::endl;
        print_mem_usage(std::cerr);
      }
      count++;
    }

    delete sampled_linked_edges;
  }

  // sample p non-linked edges from the network
  p = held_out_size / 2;
  while (p > 0) {
    Edge edge = sample_non_link_edge_for_test();
    test_map[edge] = false;
    p--;
    if (progress != 0 && count % progress == 0) {
      std::cerr << "Edges/out in test set " << count << std::endl;
      print_mem_usage(std::cerr);
    }
    count++;
  }
}

void Network::calc_max_fan_out() {
#ifdef EDGESET_IS_ADJACENCY_LIST
  // The AdjacencyList is a directed link representation; an edge
  // <me, other> in adj_list[me] is matched by an edge <other, me> in
  // adj_list[other]
  // Need to count edges only for
  //    train_link_map = linked_edges - held_out_map - test_map.
  std::vector<int> fan_out(N, 0);
  for (::size_t i = 0; i < linked_edges->size(); ++i) {
    for (auto n : (*linked_edges)[i]) {
      if (n >= static_cast<int>(i)) {  // don't count (a,b) as well as (b,a)
        Edge e(i, n);
        if (!e.in(held_out_map) && !e.in(test_map)) {
          fan_out[i]++;
          fan_out[n]++;
        }
      }
    }
  }

  fan_out_cumul_distro.resize(fan_out.size());
#pragma omp parallel for schedule(static, 1)
  for (::size_t i = 0; i < fan_out.size(); ++i) {
    fan_out_cumul_distro[i] = fan_out[i];
  }

#else  // ifdef EDGESET_IS_ADJACENCY_LIST
  std::unordered_map<int, ::size_t> fan_out;

  ::size_t i = 0;
  for (auto e : train_link_map) {
    fan_out[i] = e.size();
    i++;
  }

  std::transform(
      fan_out.begin(), fan_out.end(), std::back_inserter(fan_out_cumul_distro),
      boost::bind(&std::unordered_map<int, ::size_t>::value_type::second, _1));
#endif

  std::sort(fan_out_cumul_distro.begin(), fan_out_cumul_distro.end(),
            std::greater<::size_t>());
  std::partial_sum(fan_out_cumul_distro.begin(), fan_out_cumul_distro.end(),
                   fan_out_cumul_distro.begin());

  std::cerr << "max_fan_out " << get_max_fan_out() << std::endl;
}

::size_t Network::get_max_fan_out() const { return get_max_fan_out(1); }

::size_t Network::get_max_fan_out(::size_t batch_size) const {
  if (batch_size == 0) {
    return 0;
  }

  return fan_out_cumul_distro[batch_size - 1];
}

Edge Network::sample_non_link_edge_for_held_out() {
  while (true) {
    int firstIdx = Random::random->randint(0, N - 1);
    int secondIdx = Random::random->randint(0, N - 1);

    if (firstIdx == secondIdx) {
      continue;
    }

    // ensure the first index is smaller than the second one.
    Edge edge(std::min(firstIdx, secondIdx), std::max(firstIdx, secondIdx));

    // check conditions.
    if (edge.in(*linked_edges) || edge.in(held_out_map)) {
      continue;
    }

    return edge;
  }
}

Edge Network::sample_non_link_edge_for_test() {
  while (true) {
    int firstIdx = Random::random->randint(0, N - 1);
    int secondIdx = Random::random->randint(0, N - 1);

    if (firstIdx == secondIdx) {
      continue;
    }

    // ensure the first index is smaller than the second one.
    Edge edge(std::min(firstIdx, secondIdx), std::max(firstIdx, secondIdx));

    // check conditions.
    if (edge.in(*linked_edges) || edge.in(held_out_map) || edge.in(test_map)) {
      continue;
    }

    return edge;
  }
}

};  // namespace mcmc
