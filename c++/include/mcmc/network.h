#ifndef MCMC_MSB_NETWORK_H__
#define MCMC_MSB_NETWORK_H__

namespace mcmc {

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
template
class Network<class Vertex> {

public:
	typedef std::pair<EdgeSet *, float>		EdgeSample;

	Network(const Data<Vertex> *data, float held_out_ratio)
   		: data(data), held_out_ratio(held_out_ratio) {
	}

	virtual ~Network() {
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
	EdgeSample sample_mini_batch(::size_t mini_batch_size, strategy_t strategy) {
		switch (strategy) {
		case STRATEGY_RANDOM_PAIR:
			return random_pair_sampling(mini_batch_size);
		case STRATEGY_RANDOM_NODE:
			return random_node_sampling();
		case STRATEGY_STRATIFIED_RANDOM_PAIR:
			return stratified_random_pair_sampling(mini_batch_size);
		case STRATEGY_STRATIFIED_RANDOM_NODE:
			return stratified_random_node_sampling(10);
		default:
			throw MCMCException("Invalid sampling strategy");
		}
	}

	::size_t get_num_linked_edges() const {
		return linked_edges.size();
	}

	::size_t get_num_total_edges() const {
		return num_total_edges;
	}

	::size_t get_num_nodes() const {
		return N;
	}

	<datatype> get_linked_edges() const {
		return linked_edges;
	}

	<datatype> get_held_out_set() const {
		return held_out_map;
	}

	<datatype> get_test_set() const {
		return test_map;
	}

	void set_num_pieces(::size_t num_pieces) {
		this->num_pieces = num_pieces;
	}

	/**
	 * sample list of edges from the whole training network uniformly, regardless
	 * of links or non-links edges.The sampling approach is pretty simple: randomly generate 
	 * one edge and then check if that edge passes the conditions. The iteration
	 * stops until we get enough (mini_batch_size) edges.
	 *
	 * @return the caller must delete the result
	 */
	EdgeSample random_pair_sampling(::size_t mini_batch_size) const {
		EdgeSet *mini_batch_set = new EdgeSet();

		// iterate until we get $p$ valid edges.
		while (::size_t p = mini_batch_size; p > 0; p--) {
			int firstIdx = random.randint(0, N - 1);
			int secondIdx = random.randint(0, N - 1);
			if (firstIdx == secondIdx) {
				continue;
			}

			// make sure the first index is smaller than the second one, since
			// we are dealing with undirected graph.
			Edge edge(std::min(firstIdx, secondIdx), std::max(firstIdx, secondIdx));

			// the edge should not be in  1)hold_out set, 2)test_set  3) mini_batch_set (avoid duplicate)
			if (held_out_map.contains(edge) ||
					test_map.contains(edge) ||
					mini_batch_set.contains(edge)) {
				continue;
			}

			// great, we put it into the mini_batch list.
			mini_batch_set.add(edge);
		}

		float scale = ((N * (N - 1)) / 2) / mini_batch_size;

		return EdgeSample(edge, scale);
	}


	/**
	 * A set consists of all the pairs that involve one of the N nodes: we first sample one of 
	 * the node from N nodes, and sample all the edges for that node. h(x) = 1/N
	 */
	EdgeSample random_node_sampling() const {
		EdgeSet *mini_batch_set = new EdgeSet();

		// randomly select the node ID
		int nodeId = random.randint(0, N - 1);
		for (::size_t i = 0; i < N; i++) {
			// make sure the first index is smaller than the second one, since
			// we are dealing with undirected graph.
			int edge = Edge(std::min(nodeId, i), std::max(nodeId, i));
			if (held_out_map.contains(edge) ||
					test_map.contains(edge) ||
					mini_batch_set.contains(edge)) {
				continue;
			}

			mini_batch_set.add(edge);
		}

		return EdgeSample(edge, N);
	}


	/**
	 * We divide the edges into linked and non-linked edges, and each time either sample
	 * mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
	 * 1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
	 */
	EdgeSample stratified_random_pair_sampling(::size_t mini_batch_size) {
		::size_t p = mini_batch_size;

		EdgeSet *mini_batch_set = new EdgeSet();
		int flag = random.randint(0, 1);
		if (flag == 0) {
			// sample mini-batch from linked edges
			EdgeSet *sampled_linked_edges = random.sample(linked_edges, mini_batch_size * 2);
			for (const EdgeSet::iterator &edge = sampled_linked_edges.start;
				 	edge != sampled_linked_edges.stop;
					edge++) {
				if (p == 0) {
					break;
				}

				if (held_out_map.contains(edge) ||
						test_map.contains(edge) ||
						mini_batch_set.contains(edge)) {
					continue;
				}

				mini_batch_set.add(edge);
				p--;
			}

			delete sampled_linked_edges;

			return EdgeSample(mini_batch_set, linked_edges.size() / (float)mini_batch_size);

		} else {
			// sample mini-batch from non-linked edges
			while (p > 0) {
				p--;
			}
		}
		...
	}


	EdgeSample stratified_random_node_sampling(::size_t num_pieces) {
		...
	}


protected:
	void init_train_link_map() {
		...
	}


	void init_held_out_set() {
		...
	}


	void init_test_set() {
		...
	}


protected:
	Edge sample_non_link_edge_for_held_out() const {
		...
	}


	Edge sample_non_link_edge_for_test() const {
		...
	}


protected:
	::size_t	N;					// number of nodes in the graph
	const EdgeSet *linked_edges;	// all pair of linked edges.
	::size_t num_total_edges;		// number of total edges.
	float held_out_ratio;			// percentage of held-out data size

	std::vector<EdgeSet> train_link_map;	//
	std::vector<EdgeSet> held_out_map;		// store all held out edges
	std::vector<EdgeSet> test_map;			// store all test edges

};

}; // namespace mcmc

#endif	// ndef MCMC_MSB_NETWORK_H__
