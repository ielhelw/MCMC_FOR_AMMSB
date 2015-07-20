#ifndef MCMC_MSB_NETWORK_H__
#define MCMC_MSB_NETWORK_H__

#include <algorithm>
#include <set>
#include <unordered_set>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include "mcmc/types.h"
#include "mcmc/data.h"
#include "mcmc/random.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {

typedef std::pair<MinibatchSet *, double>		EdgeSample;

/**
 * Network class represents the whole graph that we read from the
 * data file. Since we store all the edges ONLY, the size of this
 * information is much smaller due to the graph sparsity (in general,
 * around 0.1% of links are connected)
 *
 * We use the term "linked edges" to denote the edges that two nodes
 * are connected, "non linked edges", otherwise. If we just say edge,
 * it means either linked or non-link edge.
 *
 * The class also contains lots of sampling methods that sampler can utilize.
 * This is great separation between different learners and data layer. By calling
 * the function within this class, each learner can get different types of
 * data.
 */

struct NetworkInfo {
	int32_t		N;
	int64_t		E;
	int32_t		max_fan_out;
	double		held_out_ratio;
	int64_t		held_out_size;
};


struct EdgeMapItem {
	Edge		edge;
	bool		is_edge;
};


class Network {

public:
	Network() {
	}


	// Stub for the distributed implementation that does not replicate the graph
	Network(const NetworkInfo& info)
		: N(info.N), linked_edges(NULL), num_total_edges(info.E),
		  held_out_ratio_(info.held_out_ratio), held_out_size(info.held_out_size) {
		fan_out_cumul_distro = std::vector<::size_t>(1, info.max_fan_out);
		assert(N != 0);
	}


	virtual ~Network() {
#ifdef EDGESET_IS_ADJACENCY_LIST
		adjacency_list_end();
#endif
		delete const_cast<Data *>(data_);
	}


	/**
	 * In this initialization step, we separate the whole data set
	 * into training, validation and testing sets. Basically,
	 * Training ->  used for tuning the parameters.
	 * Held-out/Validation -> used for evaluating the current model, avoid over-fitting
	 *               , the accuracy for validation set used as stopping criteria
	 * Testing -> used for calculating final model accuracy.
	 *
	 * Arguments:
	 *     data:   representation of the while graph.
	 *     held_out_ratio:  the percentage of data used for validation and testing.
	 *     create_held_out: create held-out set and test set; otherwise, caller must
	 *     					read them from file
	 */
	void Init(const Options& args, double held_out_ratio) {
		held_out_ratio_ = held_out_ratio;
		if (held_out_ratio_ == 0) {
			held_out_ratio_ = 0.1;
			std::cerr << "Choose default held_out_ratio: " << held_out_ratio << std::endl;
		}

		progress = 1 << 20;		// FIXME: make this a parameter

		preprocess::DataFactory df(args);
		df.setProgress(progress);
		data_ = df.get_data();

		N = data_->N;							// number of nodes in the graph
		linked_edges = data_->E;					// all pair of linked edges.

		if (args.dataset_class == "preprocessed") {
			ReadAuxData(args.filename + "/aux.gz", true);
#ifdef EDGESET_IS_ADJACENCY_LIST
			num_total_edges = cumulative_edges[N - 1] / 2; // number of undirected edges.
#else
			num_total_edges = linked_edges->size();		// number of total edges.
#endif

			::size_t my_held_out_size = held_out_ratio_ * get_num_linked_edges();
			std::string held_out = args.filename + "/held-out.gz";
			ReadHeldOutSet(args.filename + "/held-out.gz", true);
			ReadTestSet(args.filename + "/test.gz", true);

			if (held_out_size != my_held_out_size) {
				throw MCMCException("Expect held-out size " +
								   		to_string(my_held_out_size) +
										", get " + to_string(held_out_size));
			}

		} else {
#ifdef EDGESET_IS_ADJACENCY_LIST
			adjacency_list_init();
			num_total_edges = cumulative_edges[N - 1] / 2; // number of undirected edges.
#else
			num_total_edges = linked_edges->size();		// number of total edges.
#endif

			// Based on the a-MMSB paper, it samples equal number of
			// linked edges and non-linked edges.
			held_out_size = held_out_ratio_ * get_num_linked_edges();

			// initialize train_link_map
			init_train_link_map();
			// randomly sample hold-out and test sets.
			init_held_out_set();
			init_test_set();

			calc_max_fan_out();
		}
	}


	const Data* get_data() const {
		return data_;
	}


	void ReadSet(FileHandle& f, EdgeMap* set) {
#ifdef USE_GOOGLE_SPARSE_HASH
		// Read held_out set
		set->read_metadata(f.handle());
		set->read_nopointer_data(f.handle());
#else
		throw MCMCException("Cannot read set for this representation");
#endif
	}

	void ReadHeldOutSet(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "r");
		f.read_fully(&held_out_ratio_, sizeof held_out_ratio_);
		f.read_fully(&held_out_size, sizeof held_out_size);
		ReadSet(f, &held_out_map);
	}

	void ReadTestSet(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "r");
		ReadSet(f, &test_map);
	}

	void ReadAuxData(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "r");
		fan_out_cumul_distro.resize(N);
		f.read_fully(fan_out_cumul_distro.data(),
					 fan_out_cumul_distro.size() * sizeof fan_out_cumul_distro[0]);
#ifdef EDGESET_IS_ADJACENCY_LIST
		cumulative_edges.resize(N);
		f.read_fully(cumulative_edges.data(),
					 cumulative_edges.size() * sizeof cumulative_edges[0]);
#endif
	}


	template <typename T>
	void WriteSet(FileHandle& f, const T* set) {
#ifdef USE_GOOGLE_SPARSE_HASH
		T* mset = const_cast<T*>(set);
		// Read held_out set
		mset->write_metadata(f.handle());
		mset->write_nopointer_data(f.handle());
#else
		throw MCMCException("Cannot write set for this representation");
#endif
	}

	void WriteHeldOutSet(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "w");
		f.write_fully(&held_out_ratio_, sizeof held_out_ratio_);
		f.write_fully(&held_out_size, sizeof held_out_size);
		WriteSet(f, &held_out_map);
	}

	void WriteTestSet(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "w");
		WriteSet(f, &test_map);
	}

	void WriteAuxData(const std::string& filename, bool compressed) {
		FileHandle f(filename, compressed, "w");
		f.write_fully(fan_out_cumul_distro.data(),
					  fan_out_cumul_distro.size() * sizeof fan_out_cumul_distro[0]);
#ifdef EDGESET_IS_ADJACENCY_LIST
		f.write_fully(cumulative_edges.data(),
					  cumulative_edges.size() * sizeof cumulative_edges[0]);
#endif
	}


	void save(const std::string& dirname) {
		if (! boost::filesystem::exists(dirname)) {
			boost::filesystem::create_directories(dirname);
		}

		data_->save(dirname + "/graph.gz", true);
		WriteHeldOutSet(dirname + "/held-out.gz", true);
		WriteTestSet(dirname + "/test.gz", true);
		WriteAuxData(dirname + "/aux.gz", true);
	}


	// Stub info for the distributed implementation that does not replicate the graph
	void FillInfo(NetworkInfo* info) {
		assert(N != 0);
		info->N = N;
		info->E = num_total_edges;
		info->held_out_ratio = held_out_ratio_;
		info->held_out_size = held_out_size;
		info->max_fan_out = fan_out_cumul_distro[0];
	}


	/**
	 * Sample a mini-batch of edges from the training data.
	 * There are four different sampling strategies for edge sampling
	 * 1.random-pair sampling
	 *   sample node pairs uniformly at random.This method is an instance of independent
	 *   pair sampling, with h(x) equal to 1/(N(N-1)/2) * mini_batch_size
	 *
	 * 2.random-node sampling
	 *    A set consists of all the pairs that involve one of the N nodes: we first sample one of
	 *    the node from N nodes, and sample all the edges for that node. h(x) = 1/N
	 *
	 * 3.stratified-random-pair sampling
	 *   We divide the edges into linked and non-linked edges, and each time either sample
	 *   mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
	 *   1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
	 *
	 * 4.stratified-random-node sampling
	 *   For each node, we define a link set consisting of all its linkes, and m non-link sets
	 *   that partition its non-links. We first selct a random node, and either select its link
	 *   set or sample one of its m non-link sets. h(x) = 1/N if linked set, 1/Nm otherwise
	 *
	 *  Returns (sampled_edges, scale)
	 *  scale equals to 1/h(x), insuring the sampling gives the unbiased gradients.
	 */
	EdgeSample sample_mini_batch(::size_t mini_batch_size, strategy::strategy strategy) const {
		switch (strategy) {
		case strategy::RANDOM_PAIR:
			return random_pair_sampling(mini_batch_size);
		case strategy::RANDOM_NODE:
			return random_node_sampling();
		case strategy::STRATIFIED_RANDOM_PAIR:
			return stratified_random_pair_sampling(mini_batch_size);
		case strategy::STRATIFIED_RANDOM_NODE:
			{
				::size_t num_pieces = (N + mini_batch_size - 1) / mini_batch_size;
				// std::cerr << "Set stratified random node sampling divisor to " << num_pieces << std::endl;
				return stratified_random_node_sampling(num_pieces);
			}
		default:
			throw MCMCException("Invalid sampling strategy");
		}
	}

	::size_t minibatch_nodes_for_strategy(::size_t mini_batch_size, strategy::strategy strategy) const {
		switch (strategy) {
		case strategy::RANDOM_PAIR:
			return 2 * minibatch_edges_for_strategy(mini_batch_size, strategy);
		case strategy::RANDOM_NODE:
			return minibatch_edges_for_strategy(mini_batch_size, strategy) + 1;
		case strategy::STRATIFIED_RANDOM_PAIR:
			return minibatch_edges_for_strategy(mini_batch_size, strategy) + 1;
		case strategy::STRATIFIED_RANDOM_NODE:
			return minibatch_edges_for_strategy(mini_batch_size, strategy) + 1;
		default:
			throw MCMCException("Invalid sampling strategy");
		}
	}

	::size_t minibatch_edges_for_strategy(::size_t mini_batch_size, strategy::strategy strategy) const {
		switch (strategy) {
		case strategy::RANDOM_PAIR:
			return mini_batch_size + 1;
		case strategy::RANDOM_NODE:
			return N - held_out_map.size() - test_map.size();
		case strategy::STRATIFIED_RANDOM_PAIR:
			return mini_batch_size + 1;
		case strategy::STRATIFIED_RANDOM_NODE:
			return std::max(mini_batch_size + 1, get_max_fan_out(1));
			// return fan_out_cumul_distro[mini_batch_size];
		default:
			throw MCMCException("Invalid sampling strategy");
		}
	}

	::size_t get_num_linked_edges() const {
		return num_total_edges;
	}

	::size_t get_held_out_size() const {
		return held_out_size;
	}

	int get_num_nodes() const {
		return N;
	}

	const NetworkGraph &get_linked_edges() const {
		return *linked_edges;
	}

	const EdgeMap &get_held_out_set() const {
		return held_out_map;
	}

	const EdgeMap &get_test_set() const {
		return test_map;
	}

#ifdef UNUSED
	void set_num_pieces(::size_t num_pieces) {
		this->num_pieces = num_pieces;
	}
#endif

	/**
	 * sample list of edges from the whole training network uniformly, regardless
	 * of links or non-links edges.The sampling approach is pretty simple: randomly generate
	 * one edge and then check if that edge passes the conditions. The iteration
	 * stops until we get enough (mini_batch_size) edges.
	 *
	 * @return the caller must delete the result
	 */
	EdgeSample random_pair_sampling(::size_t mini_batch_size) const {

		MinibatchSet *mini_batch_set = new MinibatchSet();

		// iterate until we get $p$ valid edges.
		for (::size_t p = mini_batch_size; p > 0; p--) {
			int firstIdx = Random::random->randint(0, N - 1);
			int secondIdx = Random::random->randint(0, N - 1);
			if (firstIdx == secondIdx) {
				continue;
			}

			// make sure the first index is smaller than the second one, since
			// we are dealing with undirected graph.
			Edge edge(std::min(firstIdx, secondIdx), std::max(firstIdx, secondIdx));

			// the edge should not be in  1)hold_out set, 2)test_set  3) mini_batch_set (avoid duplicate)
			if (edge.in(held_out_map) || edge.in(test_map) || edge.in(*mini_batch_set)) {
				continue;
			}

			// great, we put it into the mini_batch list.
			edge.insertMe(mini_batch_set);
		}

		double scale = ((N * (N - 1)) / 2) / mini_batch_size;

		return EdgeSample(mini_batch_set, scale);
	}


	/**
	 * A set consists of all the pairs that involve one of the N nodes: we first sample one of
	 * the node from N nodes, and sample all the edges for that node. h(x) = 1/N
	 */
	EdgeSample random_node_sampling() const {
		MinibatchSet *mini_batch_set = new MinibatchSet();

		// randomly select the node ID
		int nodeId = Random::random->randint(0, N - 1);
		for (int i = 0; i < N; i++) {
			// make sure the first index is smaller than the second one, since
			// we are dealing with undirected graph.
			Edge edge(std::min(nodeId, i), std::max(nodeId, i));
			if (edge.in(held_out_map) || edge.in(test_map) || edge.in(*mini_batch_set)) {
				continue;
			}

			edge.insertMe(mini_batch_set);
		}

		return EdgeSample(mini_batch_set, N);
	}


	/**
	 * We divide the edges into linked and non-linked edges, and each time either sample
	 * mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
	 * 1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
	 */
	EdgeSample stratified_random_pair_sampling(::size_t mini_batch_size) const {
#ifdef EDGESET_IS_ADJACENCY_LIST
		throw UnimplementedException("Port stratified random pair sampling to AdjacencyList implementation");
#else
		int p = (int)mini_batch_size;

		MinibatchSet *mini_batch_set = new MinibatchSet();

		int flag = Random::random->randint(0, 1);

		if (flag == 0) {
			// sample mini-batch from linked edges
#ifdef RANDOM_FOLLOWS_PYTHON
			std::cerr << __func__ << ": FIXME: replace EdgeList w/ (unordered) EdgeSet again" << std::endl;
			auto sampled_linked_edges = Random::random->sampleList(linked_edges, mini_batch_size * 2);
#else
			auto sampled_linked_edges = Random::random->sample(linked_edges, mini_batch_size * 2);
#endif
			for (auto edge : *sampled_linked_edges) {
				if (p < 0) {
					std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					break;
				}

				if (edge.in(held_out_map) || edge.in(test_map) || edge.in(*mini_batch_set)) {
					continue;
				}

				edge.insertMe(mini_batch_set);
				p--;
			}

			delete sampled_linked_edges;

			return EdgeSample(mini_batch_set, get_num_linked_edges() / (double)mini_batch_size);

		} else {
			// sample mini-batch from non-linked edges
			while (p > 0) {
				int firstIdx = Random::random->randint(0, N - 1);
				int secondIdx = Random::random->randint(0, N - 1);

				if (firstIdx == secondIdx) {
					continue;
				}

				// ensure the first index is smaller than the second one.
				Edge edge(std::min(firstIdx, secondIdx), std::max(firstIdx, secondIdx));

				// check conditions:
				if (edge.in(*linked_edges) || edge.in(held_out_map) ||
						edge.in(test_map) || edge.in(*mini_batch_set)) {
					continue;
				}

				edge.insertMe(mini_batch_set);
				p--;
			}

			return EdgeSample(mini_batch_set,
							  (N * (N - 1)) / 2 - get_num_linked_edges() / (double)mini_batch_size);
		}
#endif
	}


	/**
	 * stratified sampling approach gives more attention to link edges (the edge is connected by two
	 * nodes). The sampling process works like this:
	 * a) randomly choose one node $i$ from all nodes (1,....N)
	 * b) decide to choose link edges or non-link edges with (50%, 50%) probability.
	 * c) if we decide to sample link edge:
	 *         return all the link edges for the chosen node $i$
	 *    else
	 *         sample edges from all non-links edges for node $i$. The number of edges
	 *         we sample equals to  number of all non-link edges / num_pieces
	 */
	EdgeSample stratified_random_node_sampling(::size_t num_pieces) const {
		while (true) {
			// randomly select the node ID
			int nodeId = Random::random->randint(0, N - 1);
			// decide to sample links or non-links
			int flag = Random::random->randint(0, 1);	// flag=0: non-link edges  flag=1: link edges
			// std::cerr << "num_pieces " << num_pieces << " flag " << flag << std::endl;

			MinibatchSet *mini_batch_set = new MinibatchSet();

			if (flag == 0) {
				/* sample non-link edges */
				// this is approximation, since the size of self.train_link_map[nodeId]
				// greatly smaller than N.
				// ::size_t mini_batch_size = (int)((N - train_link_map[nodeId].size()) / num_pieces);
				::size_t mini_batch_size = (int)(N / num_pieces);
				int p = (int)mini_batch_size;

				while (p > 0) {
					// because of the sparsity, when we sample $mini_batch_size*2$ nodes, the list likely
					// contains at least mini_batch_size valid nodes.
#ifdef EFFICIENCY_FOLLOWS_PYTHON
					auto nodeList = Random::random->sample(np::xrange(0, N), mini_batch_size * 2);
#else
					auto nodeList = Random::random->sampleRange(N, mini_batch_size * 2);
#endif
					for (std::vector<int>::iterator neighborId = nodeList->begin();
							neighborId != nodeList->end();
							neighborId++) {
						// std::cerr << "random neighbor " << *neighborId << std::endl;
						if (p < 0) {
							// std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
							break;
						}
						if (*neighborId == nodeId) {
							continue;
						}

						// check condition, and insert into mini_batch_set if it is valid.
						Edge edge(std::min(nodeId, *neighborId), std::max(nodeId, *neighborId));
						if (edge.in(*linked_edges) || edge.in(held_out_map) ||
								edge.in(test_map) || edge.in(*mini_batch_set)) {
							continue;
						}

						edge.insertMe(mini_batch_set);
						p--;
					}

					delete nodeList;
				}

				if (false) {
					std::cerr << "A Create mini batch size " << mini_batch_set->size() << " scale " << (N * num_pieces) << std::endl;
				}

				return EdgeSample(mini_batch_set, N * num_pieces);

			} else {
				/* sample linked edges */
				// return all linked edges
#ifdef EDGESET_IS_ADJACENCY_LIST
				for (auto neighborId : (*linked_edges)[nodeId]) {
					Edge e(std::min(nodeId, neighborId), std::max(nodeId, neighborId));
					if (! e.in(test_map) && ! e.in(held_out_map)) {
						e.insertMe(mini_batch_set);
					}
				}
#else
				if (false) {
					std::cerr << "train_link_map[" << nodeId << "] size " << train_link_map[nodeId].size() << std::endl;
				}
				for (VertexSet::const_iterator neighborId = train_link_map[nodeId].begin();
						neighborId != train_link_map[nodeId].end();
						neighborId++) {
					Edge edge(std::min(nodeId, *neighborId),
							  std::max(nodeId, *neighborId));
					edge.insertMe(mini_batch_set);
				}
#endif

				if (false) {
					std::cerr << "B Create mini batch size " << mini_batch_set->size() << " scale " << N << std::endl;
				}

				return EdgeSample(mini_batch_set, N);
			}
		}
	}


protected:
	/**
	 * create a set for each node, which contains list of
	 * nodes. i.e {0: Set[2,3,4], 1: Set[3,5,6]...}
	 * is used for sub-sampling
	 * the later.
	 */
	void init_train_link_map() {
#ifdef EDGESET_IS_ADJACENCY_LIST
		std::cerr << "train_link_map is gone; calculate membership for each edge as linked_edges - held_out_map - test_map" << std::endl;
#else
		train_link_map = std::vector<VertexSet>(N);
		for (auto edge = linked_edges->begin();
			 	edge != linked_edges->end();
				edge++) {
			train_link_map[edge->first].insert(edge->second);
			train_link_map[edge->second].insert(edge->first);
		}
#endif
	}


#ifdef EDGESET_IS_ADJACENCY_LIST

	// FIXME: move into np/
	template <typename T>
	static void prefix_sum(std::vector<T> *a) {
#ifndef NDEBUG
		std::vector<T> orig(a->size());
#pragma omp parallel for schedule(static, 1)
		for (::size_t i = 0; i < a->size(); ++i) {
			orig[i] = (*a)[i];
		}
#endif
		std::cerr << "omp max threads " << omp_get_max_threads() << std::endl;
		::size_t chunk = (a->size() + omp_get_max_threads() - 1) /
							omp_get_max_threads();
		std::vector<::size_t> chunk_sum(omp_get_max_threads());
#pragma omp parallel for
		for (::size_t t = 0; t < static_cast<::size_t>(omp_get_max_threads()); ++t) {
			for (::size_t i = chunk * t + 1;
				 	i < std::min(a->size(), chunk * (t + 1));
				   	++i) {
				(*a)[i] += (*a)[i - 1];
			}
		}
		chunk_sum[0] = 0;
		for (::size_t t = 1; t < static_cast<::size_t>(omp_get_max_threads()); ++t) {
			chunk_sum[t] = chunk_sum[t - 1] + (*a)[t * chunk - 1];
		}
#pragma omp parallel for
		for (::size_t t = 0; t < static_cast<::size_t>(omp_get_max_threads()); ++t) {
			for (::size_t i = chunk * t;
				 	i < std::min(a->size(), chunk * (t + 1));
				   	++i) {
				(*a)[i] += chunk_sum[t];
			}
		}
#ifndef NDEBUG
		assert((*a)[0] == orig[0]);
#pragma omp parallel for schedule(static, 1)
		for (::size_t i = 1; i < a->size(); ++i) {
			assert((*a)[i] == (*a)[i - 1] + orig[i]);
		}
#endif
	}


	void adjacency_list_init() {
		cumulative_edges.resize(N);

		if (linked_edges->size() != static_cast<::size_t>(N)) {
			throw MCMCException("AdjList size and/or cumul size corrupt");
		}

#pragma omp parallel for
		for (int32_t i = 0; i < N; ++i) {
			cumulative_edges[i] = (*linked_edges)[i].size();
		}
		prefix_sum(&cumulative_edges);
		std::cerr << "Found prefix sum for edges in AdjacencyList graph: top bucket edge " << cumulative_edges[cumulative_edges.size() - 1] << " max edge " << (cumulative_edges[cumulative_edges.size() - 1] + (*linked_edges)[cumulative_edges.size() - 1].size()) << std::endl;
 
		// std::cerr << "Initializing the held-out set on multiple machines will create randomness bugs" << std::endl;
		thread_random.resize(omp_get_max_threads());
		for (::size_t i = 0; i < thread_random.size(); ++i) {
			thread_random[i] = new Random::Random(i, 47);
		}
	}

	void adjacency_list_end() {
		for (auto r : thread_random) {
			delete r;
		}
	}


	void sample_random_edges(const NetworkGraph *linked_edges,
							 ::size_t p, std::vector<Edge> *edges) {
		std::unordered_set<Edge> collector;

		// The graph has a hit for (a,b) as well as (b,a). Therefore it contains
		// duplicates. Trust on unordered_set to filter out the duplicates. We need to
		// iterate until the requested number of edges has been inserted though.
		while (collector.size() < p) {
			std::vector<std::unordered_set<Edge>> thread_edges(omp_get_max_threads());

#pragma omp parallel for
			for (::size_t i = 0; i < p - collector.size(); ++i) {
				// Draw from 2 |Edges| since each edge is represented twice
				::size_t edge_index = thread_random[omp_get_thread_num()]->randint(0, 2 * num_total_edges);
				// locate the vertex where this edge lives
				// Actually search for find_ge, so do edge_index + 1
				int v1 = np::find_le(cumulative_edges, edge_index + 1);
				if (v1 == -1) {
					throw MCMCException("Cannot find edge " + to_string(edge_index) + " max is " + to_string(cumulative_edges[cumulative_edges.size() - 1]));
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

#endif // def EDGESET_IS_ADJACENCY_LIST


	/**
	 * Sample held out set. we draw equal number of
	 * links and non-links from the whole graph.
	 */
	void init_held_out_set() {
		::size_t p = held_out_size / 2;

		// Sample p linked-edges from the network.
		if (get_num_linked_edges() < p) {
			throw MCMCException("There are not enough linked edges that can sample from. "
							    "please use smaller held out ratio.");
		}

		::size_t count = 0;
		// FIXME make sampled_linked_edges an out param
		print_mem_usage(std::cerr);
		while (count < p) {
#if defined RANDOM_FOLLOWS_CPP_WENZHE || defined RANDOM_FOLLOWS_PYTHON
			std::cerr << __func__ << ": FIXME: replace EdgeList w/ (unordered) EdgeSet again" << std::endl;
			auto sampled_linked_edges = Random::random->sampleList(linked_edges, p);
#elif defined EDGESET_IS_ADJACENCY_LIST
			std::vector<Edge> *sampled_linked_edges = new std::vector<Edge>();
			sample_random_edges(linked_edges, p, sampled_linked_edges);
#else
			auto sampled_linked_edges = Random::random->sample(linked_edges, p);
#endif
			for (auto edge : *sampled_linked_edges) {
				// EdgeMap is an undirected graph, unfit for partitioning
				assert(edge.first < edge.second);
				held_out_map[edge] = true;
#ifndef EDGESET_IS_ADJACENCY_LIST
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
				dump(*sampled_linked_edges);
				std::cout << "held_out_set:" << std::endl;
				dump(held_out_map);
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
			std::cerr << "Edges in held-out set " << count << std::endl;
			print_mem_usage(std::cerr);
		}
	}


	/**
	 * sample test set. we draw equal number of samples for
	 * linked and non-linked edges
	 */
	void init_test_set() {
		int p = (int)(held_out_size / 2);
		// sample p linked edges from the network
		::size_t count = 0;
		while (p > 0) {
			// Because we already used some of the linked edges for held_out sets,
			// here we sample twice as much as links, and select among them, which
			// is likely to contain valid p linked edges.
#if defined RANDOM_FOLLOWS_CPP_WENZHE || defined RANDOM_FOLLOWS_PYTHON
			std::cerr << __func__ << ": FIXME: replace EdgeList w/ (unordered) EdgeSet again" << std::endl;
			// FIXME make sampled_linked_edges an out param
			auto sampled_linked_edges = Random::random->sampleList(linked_edges, 2 * p);
#elif defined EDGESET_IS_ADJACENCY_LIST
			std::vector<Edge> *sampled_linked_edges = new std::vector<Edge>();
			sample_random_edges(linked_edges, 2 * p, sampled_linked_edges);
#else
			// FIXME make sampled_linked_edges an out param
			auto sampled_linked_edges = Random::random->sample(linked_edges, 2 * p);
#endif
			for (auto edge : *sampled_linked_edges) {
				if (p < 0) {
					// std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					break;
				}

				// check whether it is already used in hold_out set
				if (edge.in(held_out_map) || edge.in(test_map)) {
					continue;
				}

				test_map[edge] = true;
#ifndef EDGESET_IS_ADJACENCY_LIST
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
		p = held_out_size / 2;
		while (p > 0) {
			Edge edge = sample_non_link_edge_for_test();
			assert(! edge.in(test_map));
			assert(! edge.in(held_out_map));
			test_map[edge] = false;
			p--;
			if (progress != 0 && count % progress == 0) {
				std::cerr << "Edges/out in test set " << count << std::endl;
				print_mem_usage(std::cerr);
			}
			count++;
		}

		if (progress != 0) {
			std::cerr << "Edges in test set " << count << std::endl;
			print_mem_usage(std::cerr);
		}
	}

	template <class T>
	static bool descending(T i, T j) { return (i > j); }

	void calc_max_fan_out() {
#ifdef EDGESET_IS_ADJACENCY_LIST
		// The AdjacencyList is a directed link representation; an edge
		// <me, other> in adj_list[me] is matched by an edge <other, me> in
		// adj_list[other]
		// Need to count edges only for
		//    train_link_map = linked_edges - held_out_map - test_map.
		std::vector<int> fan_out(N, 0);
		std::cerr << "FIXME: can't I parallelize by not doing the order check?" << std::endl;
		// abort();

		for (::size_t i = 0; i < linked_edges->size(); ++i) {
			for (auto n : (*linked_edges)[i]) {
				if (n >= static_cast<int>(i)) {	// don't count (a,b) as well as (b,a)
					Edge e(i, n);
					if (! e.in(held_out_map) && ! e.in(test_map)) {
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

#else // ifdef EDGESET_IS_ADJACENCY_LIST
		std::unordered_map<int, ::size_t> fan_out;

		::size_t i = 0;
		for (auto e: train_link_map) {
			fan_out[i] = e.size();
			i++;
		}

		std::transform(fan_out.begin(), fan_out.end(),
					   std::back_inserter(fan_out_cumul_distro),
					   boost::bind(&std::unordered_map<int, ::size_t>::value_type::second, _1));
#endif

		std::sort(fan_out_cumul_distro.begin(), fan_out_cumul_distro.end(), descending< ::size_t>);
		std::partial_sum(fan_out_cumul_distro.begin(), fan_out_cumul_distro.end(),
						 fan_out_cumul_distro.begin());

		std::cerr << "max_fan_out " << get_max_fan_out() << std::endl;
	}

public:
	::size_t get_max_fan_out() const {
		return get_max_fan_out(1);
	}


	::size_t get_max_fan_out(::size_t batch_size) const {
		if (batch_size == 0) {
			return 0;
		}

		return fan_out_cumul_distro[batch_size - 1];
	}


	::size_t get_fan_out(Vertex i) {
#ifdef EDGESET_IS_ADJACENCY_LIST
		return (*linked_edges)[i].size();
#else
		throw MCMCException(std::string(__func__) + "() not implemented for this graph representation");
#endif
	}


	::size_t marshall_edges_from(Vertex node, Vertex *marshall_area) {
#ifdef EDGESET_IS_ADJACENCY_LIST
		::size_t i = 0;
		for (auto n : (*linked_edges)[node]) {
			marshall_area[i] = n;
			i++;
		}

		return i;
#else
		throw MCMCException(std::string(__func__) + "() not implemented for this graph representation");
#endif
	}


protected:
	/**
	 * sample one non-link edge for held out set from the network. We should make sure the edge is not
	 * been used already, so we need to check the condition before we add it into
	 * held out sets
	 * TODO: add condition for checking the infinit-loop
	 */
	Edge sample_non_link_edge_for_held_out() {
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


	/**
	 * Sample one non-link edge for test set from the network. We first randomly generate one
	 * edge, then check conditions. If that edge passes all the conditions, return that edge.
	 * TODO prevent the infinit loop
	 */
	Edge sample_non_link_edge_for_test() {
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


protected:
	const Data *data_ = NULL;
	int32_t		N;					// number of nodes in the graph
	const NetworkGraph *linked_edges;	// all pair of linked edges.
	::size_t	num_total_edges;	// number of total edges.
   	double		held_out_ratio_;	// percentage of held-out data size
	::size_t	held_out_size;

#ifdef EDGESET_IS_ADJACENCY_LIST
	std::vector<::size_t> cumulative_edges;
	std::vector<Random::Random *> thread_random;
#endif

	// The map stores all the neighboring nodes for each node, within the training
	// set. The purpose of keeping this object is to make the stratified sampling
	// process easier, in which case we need to sample all the neighboring nodes
	// given the current one. The object looks like this:
	// {
	//     0: [1,3,1000,4000]
	//     1: [0,4,999]
	//   .............
	// 10000: [0,441,9000]
	//                         }
#ifndef EDGESET_IS_ADJACENCY_LIST
	std::vector<VertexSet> train_link_map;	//
#endif
	EdgeMap held_out_map;			// store all held out edges
	EdgeMap test_map;				// store all test edges

	std::vector< ::size_t> fan_out_cumul_distro;
	::size_t progress = 0;
};

}; // namespace mcmc

#endif	// ndef MCMC_MSB_NETWORK_H__
