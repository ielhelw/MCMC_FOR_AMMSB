#include "mcmc/network.h"
#include "mcmc/preprocess/data_factory.h"

namespace mcmc {

Network::Network() {}

Network::Network(const NetworkInfo& info)
    : N(info.N),
      linked_edges(NULL),
      num_total_edges(info.E),
      held_out_ratio_(info.held_out_ratio),
      held_out_size_(info.held_out_size) {
  fan_out_cumul_distro = std::vector< ::size_t>(1, info.max_fan_out);
  assert(N != 0);
  print_mem_usage(std::cerr);
}

Network::~Network() {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  thread_random_end();
#endif
  delete const_cast<Data*>(data_);

  std::cout << t_sample_sample_ << std::endl;
  std::cout << t_sample_merge_ << std::endl;
  std::cout << t_sample_merge_tail_ << std::endl;
}

::size_t Network::num_pieces_for_minibatch(::size_t mini_batch_size) const {
  return (N + mini_batch_size - 1) / mini_batch_size;
}

::size_t Network::real_minibatch_size(::size_t mini_batch_size) const {
  return N / num_pieces_for_minibatch(mini_batch_size);
}

void Network::Init(const Options& args, double held_out_ratio,
                   SourceAwareRandom* rng, int world_rank) {
  rng_ = rng;

  held_out_ratio_ = held_out_ratio;
  if (held_out_ratio_ == 0) {
    throw MCMCException("Need to specify held-out ratio");
  }

  progress = 1 << 20;  // FIXME: make this a parameter

  preprocess::DataFactory df(args);
  df.setProgress(progress);
  data_ = df.get_data();

  N = data_->N;             // number of nodes in the graph
  linked_edges = data_->E;  // all pair of linked edges.

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  thread_random_init(args.random_seed, world_rank);
#else
  sample_random.push_back(rng_->random(SourceAwareRandom::MINIBATCH_SAMPLER));
#endif

  if (args.input_class_ == "preprocessed") {
    ReadAuxData(args.input_filename_ + "/aux.gz", true);
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    num_total_edges =
        cumulative_edges[N - 1] / 2;  // number of undirected edges.
#else
    num_total_edges = linked_edges->size();  // number of total edges.
#endif

    ::size_t my_held_out_size = held_out_ratio_ * get_num_linked_edges();
    std::string held_out = args.input_filename_ + "/held-out.gz";
    ReadHeldOutSet(args.input_filename_ + "/held-out.gz", true);
    ReadTestSet(args.input_filename_ + "/test.gz", true);

    if (held_out_size_ != my_held_out_size) {
      std::cerr << "WARNING: Expect held-out size " +
                          to_string(my_held_out_size) + ", get " +
                          to_string(held_out_size_);
    }

  } else {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    adjacency_list_init();
    // number of undirected edges
    num_total_edges = cumulative_edges[N - 1] / 2;
#else
    num_total_edges = linked_edges->size();  // number of total edges.
#endif

    // Based on the a-MMSB paper, it samples equal number of
    // linked edges and non-linked edges.
    held_out_size_ = held_out_ratio_ * get_num_linked_edges();

    // initialize train_link_map
    init_train_link_map();
    // randomly sample hold-out and test sets.
    init_held_out_set();
    init_test_set();

    calc_max_fan_out();
  }

  t_sample_sample_       = Timer("      sample sample");
  t_sample_merge_        = Timer("      sample merge");
  t_sample_merge_tail_   = Timer("      sample merge/tail");
}

const Data* Network::get_data() const { return data_; }

void Network::ReadSet(FileHandle& f, EdgeMap* set) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  // Read held_out set
  set->read_metadata(f.handle());
  set->read_nopointer_data(f.handle());
#else
  throw MCMCException("Cannot read set for this representation");
#endif
}

void Network::ReadHeldOutSet(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "r");
  f.read_fully(&held_out_ratio_, sizeof held_out_ratio_);
  f.read_fully(&held_out_size_, sizeof held_out_size_);
  ReadSet(f, &held_out_map);
}

void Network::ReadTestSet(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "r");
  ReadSet(f, &test_map);
}

void Network::ReadAuxData(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "r");
  fan_out_cumul_distro.resize(N);
  f.read_fully(fan_out_cumul_distro.data(),
               fan_out_cumul_distro.size() * sizeof fan_out_cumul_distro[0]);
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  cumulative_edges.resize(N);
  f.read_fully(cumulative_edges.data(),
               cumulative_edges.size() * sizeof cumulative_edges[0]);
#endif
}

void Network::WriteHeldOutSet(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "w");
  f.write_fully(&held_out_ratio_, sizeof held_out_ratio_);
  f.write_fully(&held_out_size_, sizeof held_out_size_);
  WriteSet(f, &held_out_map);
}

void Network::WriteTestSet(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "w");
  WriteSet(f, &test_map);
}

void Network::WriteAuxData(const std::string& filename, bool compressed) {
  FileHandle f(filename, compressed, "w");
  f.write_fully(fan_out_cumul_distro.data(),
                fan_out_cumul_distro.size() * sizeof fan_out_cumul_distro[0]);
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  f.write_fully(cumulative_edges.data(),
                cumulative_edges.size() * sizeof cumulative_edges[0]);
#endif
}

void Network::save(const std::string& dirname) {
  if (!boost::filesystem::exists(dirname)) {
    boost::filesystem::create_directories(dirname);
  }

  data_->save(dirname + "/graph.gz", true);
  WriteHeldOutSet(dirname + "/held-out.gz", true);
  WriteTestSet(dirname + "/test.gz", true);
  WriteAuxData(dirname + "/aux.gz", true);
}

void Network::FillInfo(NetworkInfo* info) {
  assert(N != 0);
  info->N = N;
  info->E = num_total_edges;
  info->held_out_ratio = held_out_ratio_;
  info->held_out_size = held_out_size_;
  info->max_fan_out = fan_out_cumul_distro[0];
}

EdgeSample Network::sample_mini_batch(::size_t mini_batch_size,
                                      strategy::strategy strategy) {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      return stratified_random_node_sampling(
               num_pieces_for_minibatch(mini_batch_size));
    case strategy::RANDOM_EDGE:
      return random_edge_sampling(mini_batch_size);
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::max_minibatch_nodes_for_strategy(
    ::size_t mini_batch_size, strategy::strategy strategy) const {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      return max_minibatch_edges_for_strategy(mini_batch_size, strategy) + 1;
    case strategy::RANDOM_EDGE:
      return 2 * mini_batch_size;
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::max_minibatch_edges_for_strategy(
    ::size_t mini_batch_size, strategy::strategy strategy) const {
  switch (strategy) {
    case strategy::STRATIFIED_RANDOM_NODE:
      return std::max(mini_batch_size + 1, get_max_fan_out(1));
    // return fan_out_cumul_distro[mini_batch_size];
    case strategy::RANDOM_EDGE:
      return mini_batch_size;
    default:
      throw MCMCException("Invalid sampling strategy");
  }
}

::size_t Network::get_num_linked_edges() const { return num_total_edges; }

::size_t Network::get_held_out_size() const { return held_out_size_; }

int Network::get_num_nodes() const { return N; }

const NetworkGraph& Network::get_linked_edges() const { return *linked_edges; }

const EdgeMap& Network::get_held_out_set() const { return held_out_map; }

const EdgeMap& Network::get_test_set() const { return test_map; }

EdgeSample Network::sample_full_training_set() const {
  MinibatchSet* mini_batch_set = new MinibatchSet();

  for (auto edge : *linked_edges) {
    if (edge.first < edge.second) {
      if (edge.in(held_out_map) || edge.in(test_map) ||
          edge.in(*mini_batch_set)) {
        continue;
      }
      mini_batch_set->insert(edge);
    }
  }

  Float weight = (N - 1) * (double)N / 2.0 / mini_batch_set->size();
  // Float weight = 1.0;

  // std::cerr << "Minibatch size " << mini_batch_set->size() << " weight " << weight << std::endl;

  return EdgeSample(mini_batch_set, weight);
}

EdgeSample Network::random_edge_sampling(::size_t mini_batch_size) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  ::size_t undirected_edges = linked_edges->size() / 2;
#else
  ::size_t undirected_edges = linked_edges->size();
#endif
  if (mini_batch_size >= undirected_edges - held_out_map.size() - test_map.size()) {
    return sample_full_training_set();
  }

  MinibatchSet* mini_batch_set = new MinibatchSet();

  while (mini_batch_set->size() < mini_batch_size) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    std::vector<Edge>* sampled_linked_edges = new std::vector<Edge>();
    sample_random_edges(linked_edges, mini_batch_size, sampled_linked_edges);
#else
    Random::Random* rng = rng_->random(SourceAwareRandom::MINIBATCH_SAMPLER);
    auto sampled_linked_edges = rng->sample(*linked_edges, mini_batch_size);
#endif
    for (auto edge : *sampled_linked_edges) {
      if (mini_batch_set->size() == mini_batch_size) {
        break;
      }
      if (edge.first < edge.second) {
        if (edge.in(held_out_map) || edge.in(test_map) ||
            edge.in(*mini_batch_set)) {
          continue;
        }
        mini_batch_set->insert(edge);
      }
    }
    delete sampled_linked_edges;
  }

  Float weight = (N - 1) * (double)N / 2.0 / mini_batch_set->size();

  // std::cerr << "Minibatch size " << mini_batch_set->size() << std::endl;

  return EdgeSample(mini_batch_set, weight);
}

EdgeSample Network::stratified_random_node_sampling(::size_t num_pieces) {
  Random::Random* rng = rng_->random(SourceAwareRandom::MINIBATCH_SAMPLER);
  while (true) {
    // randomly select the node ID
    int nodeId = rng->randint(0, N - 1);
    // decide to sample links or non-links
    // flag=0: non-link edges  flag=1: link edges
    int flag = rng->randint(0, 1);
    // std::cerr << "num_pieces " << num_pieces << " flag " << flag <<
    // std::endl;

    MinibatchSet* mini_batch_set = new MinibatchSet();

    if (flag == 0) {
      /* sample non-link edges */
      // this is approximation, since the size of self.train_link_map[nodeId]
      // greatly smaller than N.
      // ::size_t mini_batch_size = (int)((N - train_link_map[nodeId].size()) /
      // num_pieces);
      ::size_t mini_batch_size = (::size_t)(N / num_pieces);

      std::vector<MinibatchSet> local_minibatch(sample_random.size());

      while (mini_batch_set->size() < mini_batch_size) {
#pragma omp parallel for
        for (::size_t t = 0; t < local_minibatch.size(); ++t) {
          local_minibatch[t].clear();
        }

        t_sample_sample_.start();
        ::size_t sample_size = (mini_batch_size - mini_batch_set->size() + local_minibatch.size() - 1) / local_minibatch.size();
#pragma omp parallel for
        for (::size_t t = 0; t < local_minibatch.size(); ++t) {
          Random::Random *rng = sample_random[t];
          auto nodeList = rng->sampleRange(N, sample_size);
          for (std::vector<int>::iterator neighborId = nodeList->begin();
               neighborId != nodeList->end(); neighborId++) {
            // std::cerr << "random neighbor " << *neighborId << std::endl;
            if (*neighborId != nodeId) {
              // check condition, and insert into mini_batch_set if it is valid.
              Edge edge(std::min(nodeId, *neighborId),
                        std::max(nodeId, *neighborId));
              if (! edge.in(*linked_edges) && ! edge.in(held_out_map) &&
                  ! edge.in(test_map) && ! edge.in(*mini_batch_set)) {

                local_minibatch[t].insert(edge);
              }
            }
          }
          delete nodeList;
        }
        t_sample_sample_.stop();

        for (::size_t t = 0; t < local_minibatch.size(); ++t) {
          ::size_t chunk = std::min(mini_batch_size - mini_batch_set->size(),
                                    local_minibatch[t].size());
          if (chunk < local_minibatch[t].size()) {
            t_sample_merge_tail_.start();
            for (auto x : local_minibatch[t]) {
              mini_batch_set->insert(x);
              chunk--;
              if (chunk == 0) {
                t_sample_merge_tail_.stop();
                break;
              }
            }
            break;
          } else {
            t_sample_merge_.start();
            mini_batch_set->insert(local_minibatch[t].begin(),
                                   local_minibatch[t].end());
            t_sample_merge_.stop();
          }
        }
      }


      Float scale = (Float)N * num_pieces;
      scale *= (double)num_total_edges / ((double)N * (N - 1.0) / 2.0 - num_total_edges);
      if (false) {
        std::cerr << "A Create mini batch size " << mini_batch_set->size()
                  << " scale " << scale << std::endl;
      }

      return EdgeSample(mini_batch_set, scale);

    } else {
/* sample linked edges */
// return all linked edges
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
      for (auto neighborId : linked_edges->edges_at(nodeId)) {
        Edge e(std::min(nodeId, neighborId), std::max(nodeId, neighborId));
        if (!e.in(test_map) && !e.in(held_out_map)) {
          mini_batch_set->insert(e);
        }
      }
#else
      if (false) {
        std::cerr << "train_link_map[" << nodeId << "] size "
                  << train_link_map[nodeId].size() << std::endl;
      }
      for (VertexSet::const_iterator neighborId =
               train_link_map[nodeId].begin();
           neighborId != train_link_map[nodeId].end(); neighborId++) {
        Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
        mini_batch_set->insert(edge);
      }
#endif

      if (false) {
        std::cerr << "B Create mini batch size " << mini_batch_set->size()
                  << " scale " << N << std::endl;
      }

      if (mini_batch_set->size() != 0) {
        return EdgeSample(mini_batch_set, N);
      }
    }

    std::cerr << "Empty minibatch, try again..." << std::endl;
  }
}

void Network::init_train_link_map() {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  std::cerr << "train_link_map is gone; calculate membership for each edge as "
               "linked_edges - held_out_map - test_map" << std::endl;
#else
  train_link_map = std::vector<VertexSet>(N);
  for (auto edge = linked_edges->begin(); edge != linked_edges->end(); edge++) {
    train_link_map[edge->first].insert(edge->second);
    train_link_map[edge->second].insert(edge->first);
  }
#endif
}

#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
void Network::adjacency_list_init() {
  cumulative_edges.resize(N);

  if (linked_edges->edges_at_size() != static_cast< ::size_t>(N)) {
    throw MCMCException("AdjList size and/or cumul size corrupt");
  }

#pragma omp parallel for
  for (int32_t i = 0; i < N; ++i) {
    cumulative_edges[i] = linked_edges->edges_at(i).size();
  }
  prefix_sum(&cumulative_edges);
  std::cerr
      << "Found prefix sum for edges in AdjacencyList graph: top bucket edge "
      << cumulative_edges[cumulative_edges.size() - 1] << " max edge "
      << (cumulative_edges[cumulative_edges.size() - 1] +
          linked_edges->edges_at(cumulative_edges.size() - 1).size()) << std::endl;
}

void Network::thread_random_init(int random_seed, int world_rank) {
  std::cerr << "Create per-thread GRAPH_INIT randoms" << std::endl;
  thread_random.resize(omp_get_max_threads());
  for (::size_t i = 0; i < thread_random.size(); ++i) {
    int seed = random_seed + SourceAwareRandom::GRAPH_INIT;
    thread_random[i] = new Random::Random(seed + 1 + i +
                                           world_rank * thread_random.size(),
                                          seed, false);
  }
  std::cerr << "Create per-thread MINIBATCH_SAMPLER randoms" << std::endl;
  sample_random.resize(omp_get_max_threads());
  for (::size_t i = 0; i < sample_random.size(); ++i) {
    int seed = random_seed + SourceAwareRandom::MINIBATCH_SAMPLER;
    sample_random[i] = new Random::Random(seed + 1 + i +
                                           world_rank * thread_random.size(),
                                          seed, false);
  }
}

void Network::thread_random_end() {
  for (auto r : thread_random) {
    delete r;
  }
  for (auto r : sample_random) {
    delete r;
  }
}

void Network::sample_random_edges(const NetworkGraph* linked_edges, ::size_t p,
                                  std::vector<Edge>* edges) const {
  std::unordered_set<Edge, EdgeHash> collector;

  // The graph has a hit for (a,b) as well as (b,a). Therefore it contains
  // duplicates. Trust on unordered_set to filter out the duplicates. We need to
  // iterate until the requested number of edges has been inserted though.
  while (collector.size() < p) {
    std::vector<std::unordered_set<Edge, EdgeHash> > thread_edges(omp_get_max_threads());

#pragma omp parallel for
    for (::size_t i = 0; i < p - collector.size(); ++i) {
      // Draw from 2 |Edges| since each edge is represented twice
      ::size_t edge_index =
          thread_random[omp_get_thread_num()]->randint(0,
                                                       2LL * num_total_edges - 1);
      // locate the vertex where this edge lives
      // Actually search for find_ge, so do edge_index + 1
      int v1 = np::find_le(cumulative_edges, edge_index + 1);
      if (v1 == -1) {
        throw MCMCException(
            "Cannot find edge " + to_string(edge_index) + " max is " +
            to_string(cumulative_edges[cumulative_edges.size() - 1]));
      }
      if (v1 != 0) {
        edge_index -= cumulative_edges[v1 - 1];
      }
      int v2 = -1;
      // draw edge_index'th neighbor within this edge list
      if ((::size_t)v1 >= linked_edges->edges_at_size()) {
        std::cerr << "OOOOOOOPPPPPPPPPPPPPSSSSSSSS outside vector" << std::endl;
      }
      if (linked_edges->edges_at(v1).size() <= 0) {
        std::cerr << "OOOOPPPPPPPPPPSSSSSSSSSSSS empty elt" << std::endl;
      }
      for (auto n : linked_edges->edges_at(v1)) {
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

#endif  // def MCMC_EDGESET_IS_ADJACENCY_LIST

void Network::init_held_out_set() {
  ::size_t p = held_out_size_ / 2;

  // Sample p linked-edges from the network.
  if (get_num_linked_edges() < p) {
    throw MCMCException(
        "There are not enough linked edges that can sample from. "
        "please use smaller held out ratio.");
  }

  ::size_t count = 0;
  print_mem_usage(std::cerr);
#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST
  auto* rng = rng_->random(SourceAwareRandom::GRAPH_INIT);
#endif
  while (count < p) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    std::vector<Edge>* sampled_linked_edges = new std::vector<Edge>();
    sample_random_edges(linked_edges, p, sampled_linked_edges);
#else
    auto sampled_linked_edges = rng->sample(*linked_edges, p);
#endif
    for (auto edge : *sampled_linked_edges) {
      // EdgeMap is an undirected graph, unfit for partitioning
      assert(edge.first < edge.second);
      held_out_map[edge] = true;
#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST
      train_link_map[edge.first].erase(edge.second);
      train_link_map[edge.second].erase(edge.first);
#endif
      if (progress != 0 && count % progress == 0) {
        std::cerr << "Edges/in in held-out set " << count << std::endl;
        print_mem_usage(std::cerr);
      }
      count++;
    }

    if (false) {
      std::cout << "sampled_linked_edges:" << std::endl;
      dump(std::cout, *sampled_linked_edges);
    }

    delete sampled_linked_edges;
  }

  // sample p non-linked edges from the network
  while (count < 2 * p) {
    Edge edge = sample_non_link_edge_for_held_out();
    // EdgeMap is an undirected graph, unfit for partitioning
    assert(edge.first < edge.second);
    held_out_map[edge] = false;
    if (progress != 0 && count % progress == 0) {
      std::cerr << "Edges/out in held-out set " << count << std::endl;
      print_mem_usage(std::cerr);
    }
    count++;
  }

  if (progress != 0) {
    std::cerr << "Edges in held-out set " << held_out_map.size() << std::endl;
    print_mem_usage(std::cerr);
  }

  if (false) {
    std::cout << "held_out_set:" << std::endl;
    dump(std::cout, held_out_map);
  }
}

void Network::init_test_set() {
  int p = (int)(held_out_size_ / 2);
  // sample p linked edges from the network
  ::size_t count = 0;
#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST
  auto* rng = rng_->random(SourceAwareRandom::GRAPH_INIT);
#endif
  while (p > 0) {
// Because we already used some of the linked edges for held_out sets,
// here we sample twice as much as links, and select among them, which
// is likely to contain valid p linked edges.
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
    std::vector<Edge>* sampled_linked_edges = new std::vector<Edge>();
    sample_random_edges(linked_edges, 2 * p, sampled_linked_edges);
#else
    auto sampled_linked_edges = rng->sample(*linked_edges, 2 * p);
#endif
    for (auto edge : *sampled_linked_edges) {
      if (p < 0) {
        // std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" <<
        // std::endl;
        break;
      }

      // check whether it is already used in hold_out set
      if (edge.in(held_out_map) || edge.in(test_map)) {
        continue;
      }

      test_map[edge] = true;
#ifndef MCMC_EDGESET_IS_ADJACENCY_LIST
      train_link_map[edge.first].erase(edge.second);
      train_link_map[edge.second].erase(edge.first);
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
  p = held_out_size_ / 2;
  while (p > 0) {
    Edge edge = sample_non_link_edge_for_test();
    assert(!edge.in(test_map));
    assert(!edge.in(held_out_map));
    test_map[edge] = false;
    --p;
    if (progress != 0 && count % progress == 0) {
      std::cerr << "Edges/out in test set " << count << std::endl;
      print_mem_usage(std::cerr);
    }
    ++count;
  }

  if (progress != 0) {
    std::cerr << "Edges in test set " << count << std::endl;
    print_mem_usage(std::cerr);
  }
}

void Network::calc_max_fan_out() {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  // The AdjacencyList is a directed link representation; an edge
  // <me, other> in adj_list[me] is matched by an edge <other, me> in
  // adj_list[other]
  // Need to count edges only for
  //    train_link_map = linked_edges - held_out_map - test_map.
  std::vector<int> fan_out(N, 0);
  std::cerr << "FIXME: can't I parallelize by not doing the order check?"
            << std::endl;
  // abort();

  for (::size_t i = 0; i < linked_edges->edges_at_size(); ++i) {
    for (auto n : linked_edges->edges_at(i)) {
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

#else  // ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
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
            descending< ::size_t>);
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

::size_t Network::get_fan_out(Vertex i) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  return linked_edges->edges_at(i).size();
#else
  throw MCMCException(std::string(__func__) +
                      "() not implemented for this graph representation");
#endif
}

::size_t Network::marshall_edges_from(Vertex node, Vertex* marshall_area) {
#ifdef MCMC_EDGESET_IS_ADJACENCY_LIST
  ::size_t i = 0;
  for (auto n : linked_edges->edges_at(node)) {
    marshall_area[i] = n;
    i++;
  }

  return i;
#else
  throw MCMCException(std::string(__func__) +
                      "() not implemented for this graph representation");
#endif
}

Edge Network::sample_non_link_edge_for_held_out() {
  while (true) {
    auto* rng = rng_->random(SourceAwareRandom::GRAPH_INIT);
    int firstIdx = rng->randint(0, N - 1);
    int secondIdx = rng->randint(0, N - 1);

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
    auto* rng = rng_->random(SourceAwareRandom::GRAPH_INIT);
    int firstIdx = rng->randint(0, N - 1);
    int secondIdx = rng->randint(0, N - 1);

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

}  //  namespace mcmc
