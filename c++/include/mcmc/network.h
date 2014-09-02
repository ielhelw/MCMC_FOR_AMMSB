#ifndef MCMC_MSB_NETWORK_H__
#define MCMC_MSB_NETWORK_H__

#include <algorithm>
#include <set>
#include <unordered_set>

#include "mcmc/types.h"
#include "mcmc/data.h"
#include "mcmc/random.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {

typedef std::pair<EdgeSet *, float>		EdgeSample;

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
class Network {

public:

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
	 *     vlaidation_ratio:  the percentage of data used for validation and testing.
	 */
	Network(const Data *data, float held_out_ratio) {
		N = data->N;							// number of nodes in the graph
		linked_edges = data->E;					// all pair of linked edges.
		num_total_edges = linked_edges->size(); // number of total edges.
	   	this->held_out_ratio = held_out_ratio;	// percentage of held-out data size

		// Based on the a-MMSB paper, it samples equal number of
		// linked edges and non-linked edges.
		held_out_size = held_out_ratio * linked_edges->size();

		// initialize train_link_map
		init_train_link_map();
		// randomly sample hold-out and test sets.
		init_held_out_set();
		init_test_set();
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
	EdgeSample sample_mini_batch(::size_t mini_batch_size, strategy::strategy strategy) const {
		switch (strategy) {
		case strategy::RANDOM_PAIR:
			return random_pair_sampling(mini_batch_size);
		case strategy::RANDOM_NODE:
			return random_node_sampling();
		case strategy::STRATIFIED_RANDOM_PAIR:
			return stratified_random_pair_sampling(mini_batch_size);
		case strategy::STRATIFIED_RANDOM_NODE:
			return stratified_random_node_sampling(10);
		default:
			throw MCMCException("Invalid sampling strategy");
		}
	}

	::size_t get_num_linked_edges() const {
		return linked_edges->size();
	}

	::size_t get_num_total_edges() const {
		return num_total_edges;
	}

	int get_num_nodes() const {
		return N;
	}

	const EdgeSet &get_linked_edges() const {
		return *linked_edges;
	}

	const EdgeMap &get_held_out_set() const {
		return held_out_map;
	}

	const EdgeMap &get_test_set() const {
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
			mini_batch_set->insert(edge);
		}

		float scale = ((N * (N - 1)) / 2) / mini_batch_size;

		return EdgeSample(mini_batch_set, scale);
	}


	/**
	 * A set consists of all the pairs that involve one of the N nodes: we first sample one of
	 * the node from N nodes, and sample all the edges for that node. h(x) = 1/N
	 */
	EdgeSample random_node_sampling() const {
		EdgeSet *mini_batch_set = new EdgeSet();

		// randomly select the node ID
		int nodeId = Random::random->randint(0, N - 1);
		for (int i = 0; i < N; i++) {
			// make sure the first index is smaller than the second one, since
			// we are dealing with undirected graph.
			Edge edge(std::min(nodeId, i), std::max(nodeId, i));
			if (edge.in(held_out_map) || edge.in(test_map) || edge.in(*mini_batch_set)) {
				continue;
			}

			mini_batch_set->insert(edge);
		}

		return EdgeSample(mini_batch_set, N);
	}


	/**
	 * We divide the edges into linked and non-linked edges, and each time either sample
	 * mini-batch from linked-edges or non-linked edges.  g(x) = 1/N_0 for non-link and
	 * 1/N_1 for link, where N_0-> number of non-linked edges, N_1-> # of linked edges.
	 */
	EdgeSample stratified_random_pair_sampling(::size_t mini_batch_size) const {
		int p = (int)mini_batch_size;

		EdgeSet *mini_batch_set = new EdgeSet();

		int flag = Random::random->randint(0, 1);

		if (flag == 0) {
			// sample mini-batch from linked edges
			EdgeSet *sampled_linked_edges = Random::random->sample(linked_edges, mini_batch_size * 2);
			for (EdgeSet::iterator edge = sampled_linked_edges->begin();
				 	edge != sampled_linked_edges->end();
					edge++) {
				if (p < 0) {
					std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					break;
				}

				if (edge->in(held_out_map) || edge->in(test_map) || edge->in(*mini_batch_set)) {
					continue;
				}

				mini_batch_set->insert(*edge);
				p--;
			}

			delete sampled_linked_edges;

			return EdgeSample(mini_batch_set, linked_edges->size() / (float)mini_batch_size);

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

				mini_batch_set->insert(edge);
				p--;
			}

			return EdgeSample(mini_batch_set,
							  (N * (N - 1)) / 2 - linked_edges->size() / (float)mini_batch_size);
		}
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
		// randomly select the node ID
		int nodeId = Random::random->randint(0, N - 1);
		// decide to sample links or non-links
		int flag = Random::random->randint(0, 1);	// flag=0: non-link edges  flag=1: link edges

		EdgeSet *mini_batch_set = new EdgeSet();

		if (flag == 0) {
			/* sample non-link edges */
			// this is approximation, since the size of self.train_link_map[nodeId]
			// greatly smaller than N.
			::size_t mini_batch_size = (int)((N - train_link_map[nodeId].size()) / num_pieces);
			int p = (int)mini_batch_size;

			while (p > 0) {
				// because of the sparsity, when we sample $mini_batch_size*2$ nodes, the list likely
				// contains at least mini_batch_size valid nodes.
				std::vector<int> *nodeList = Random::random->sample(np::xrange(0, N), mini_batch_size * 2);
				for (std::vector<int>::iterator neighborId = nodeList->begin();
					 	neighborId != nodeList->end();
						neighborId++) {
					if (p < 0) {
						std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
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

					mini_batch_set->insert(edge);
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
			if (false) {
				std::cerr << "train_link_map[" << nodeId << "] size " << train_link_map[nodeId].size() << std::endl;
			}
			for (VertexSet::const_iterator neighborId = train_link_map[nodeId].begin();
				 	neighborId != train_link_map[nodeId].end();
					neighborId++) {
				mini_batch_set->insert(Edge(std::min(nodeId, *neighborId),
										   	std::max(nodeId, *neighborId)));
			}

			if (false) {
				std::cerr << "B Create mini batch size " << mini_batch_set->size() << " scale " << N << std::endl;
			}
			return EdgeSample(mini_batch_set, N);
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
		train_link_map = std::vector<VertexSet>(N);
		for (EdgeSet::const_iterator edge = linked_edges->begin();
			 	edge != linked_edges->end();
				edge++) {
			train_link_map[edge->first].insert(edge->second);
			train_link_map[edge->second].insert(edge->first);
		}
	}


	/**
	 * Sample held out set. we draw equal number of
	 * links and non-links from the whole graph.
	 */
	void init_held_out_set() {
		::size_t p = held_out_size / 2;

		// Sample p linked-edges from the network.
		if (linked_edges->size() < p) {
			throw MCMCException("There are not enough linked edges that can sample from. "
							    "please use smaller held out ratio.");
		}

		EdgeSet *sampled_linked_edges = Random::random->sample(linked_edges, p);
		for (EdgeSet::iterator edge = sampled_linked_edges->begin();
			 	edge != sampled_linked_edges->end();
				edge++) {
			held_out_map[*edge] = true;
			train_link_map[edge->first].erase(edge->second);
			train_link_map[edge->second].erase(edge->first);
		}

		// sample p non-linked edges from the network
		while (p > 0) {
			Edge edge = sample_non_link_edge_for_held_out();
			held_out_map[edge] = false;
			p--;
		}

		if (false) {
			std::cout << "sampled_linked_edges:" << std::endl;
			dump(*sampled_linked_edges);
			std::cout << "held_out_set:" << std::endl;
			dump(held_out_map);
		}

		delete sampled_linked_edges;
	}


	/**
	 * sample test set. we draw equal number of samples for
	 * linked and non-linked edges
	 */
	void init_test_set() {
		int p = (int)(held_out_size / 2);
		// sample p linked edges from the network
		while (p > 0) {
			// Because we already used some of the linked edges for held_out sets,
			// here we sample twice as much as links, and select among them, which
			// is likely to contain valid p linked edges.
			EdgeSet *sampled_linked_edges = Random::random->sample(linked_edges, 2 * p);
			for (EdgeSet::iterator edge = sampled_linked_edges->begin();
				 	edge != sampled_linked_edges->end();
					edge++) {
				if (p < 0) {
					std::cerr << __func__ << ": Are you sure p < 0 is a good idea?" << std::endl;
					break;
				}

				// check whether it is already used in hold_out set
				if (edge->in(held_out_map) || edge->in(test_map)) {
					continue;
				}

				test_map[*edge] = true;
				train_link_map[edge->first].erase(edge->second);
				train_link_map[edge->second].erase(edge->first);
				p--;
			}
		}

		// sample p non-linked edges from the network
		p = held_out_size / 2;
		while (p > 0) {
			Edge edge = sample_non_link_edge_for_test();
			test_map[edge] = false;
			p--;
		}
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
	int			N;					// number of nodes in the graph
	const EdgeSet *linked_edges;	// all pair of linked edges.
	::size_t	num_total_edges;	// number of total edges.
	float		held_out_ratio;		// percentage of held-out data size
	::size_t	held_out_size;

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
	std::vector<VertexSet> train_link_map;	//
	EdgeMap held_out_map;			// store all held out edges
	EdgeMap test_map;				// store all test edges

	::size_t	num_pieces;

};

}; // namespace mcmc

#endif	// ndef MCMC_MSB_NETWORK_H__
