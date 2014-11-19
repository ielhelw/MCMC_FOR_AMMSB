/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_RELATIVITY_H__
#define MCMC_PREPROCESS_RELATIVITY_H__

#include <fstream>
#include <set>
#include <unordered_set>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

#include "mcmc/timer.h"

namespace mcmc {
namespace preprocess {

// FIXME: identical: hep_ph relativity ...

/**
 * Process relativity data set
 */
class Relativity : public DataSet {
public:
	Relativity(const std::string &filename) : DataSet(filename == "" ? "datasets/CA-GrQc.txt" : filename) {
	}

	virtual ~Relativity() {
	}

	/**
	 * The data is stored in .txt file. The format of data is as follows, the first column
	 * is line number. Within each line, it is tab separated.
	 *
	 * [1] some texts
	 * [2] some texts
	 * [3] some texts
	 * [4] some texts
	 * [5] 1    100
	 * [6] 1    103
	 * [7] 4    400
	 * [8] ............
	 *
	 * However, the node ID is not increasing by 1 every time. Thus, we re-format
	 * the node ID first.
	 */
	virtual const Data *process() {
		timer::Timer t_readline("  read lines");
		timer::Timer t_parse(   "  parse lines");
		timer::Timer t_renumber("  renumber");
		timer::Timer t_set(     "  create set");
		timer::Timer::setTabular(true);

		std::ifstream infile(filename);
		if (! infile) {
			throw mcmc::IOException("Cannot open " + filename);
		}

		std::string line;
		for (int i = 0; i < 4; i++) {
			std::getline(infile, line);
		}

		t_readline.start();
		// start from the 5th line.
		std::set<int> vertex;	// ordered set
		// std::unordered_set<int> vertex;	// ordered set
#if 0
		std::vector<mcmc::Edge> edge;
#else
		mcmc::EdgeSet *edge = new EdgeSet();
#endif
		int max_vertex = -1;
		while (std::getline(infile, line)) {
			int a;
			int b;
			t_parse.start();
#if 0
			std::istringstream iss(line);
			if (! (iss >> a >> b)) {
				throw mcmc::IOException("Fail to parse int");
			}
#else
			if (sscanf(line.c_str(), " %d %d", &a, &b) != 2) {
				throw mcmc::IOException("Fail to parse int");
			}
#endif
			t_parse.stop();
			vertex.insert(a);
			vertex.insert(b);
			max_vertex = std::max(max_vertex, a);
			max_vertex = std::max(max_vertex, b);
#if 0
			edge.push_back(Edge(a, b));
#else
			edge->insert(Edge(std::min(a, b), std::max(a, b)));
#endif
		}
		t_readline.stop();

		std::vector<int> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

		::size_t N = nodelist.size();
		if (( ::size_t)max_vertex == N) {
			std::cerr << "No need to renumber..." << std::endl;

			timer::Timer::printHeader(std::cout);
			std::cout << t_readline << std::endl;
			std::cout << t_parse << std::endl;
			std::cout << t_renumber << std::endl;
			std::cout << t_set << std::endl;

			return new Data(NULL, edge, N);
		}

		t_renumber.start();
		// change the node ID to make it start from 0
		std::unordered_map<int, int> node_id_map;
		int i = 0;
		for (std::vector<int>::iterator node_id = nodelist.begin();
			 	node_id != nodelist.end();
				node_id++) {
			node_id_map[*node_id] = i;
			i++;
		}
		t_renumber.stop();

		t_set.start();
		mcmc::EdgeSet *E = new mcmc::EdgeSet();	// store all pair of edges.
		for (auto i: *edge) {
			int node1 = node_id_map[i.first];
			int node2 = node_id_map[i.second];
			if (node1 == node2) {
				continue;
			}
			E->insert(Edge(std::min(node1, node2), std::max(node1, node2)));
		}
		t_set.stop();

		timer::Timer::printHeader(std::cout);
		std::cout << t_readline << std::endl;
		std::cout << t_parse << std::endl;
		std::cout << t_renumber << std::endl;
		std::cout << t_set << std::endl;

		return new Data(NULL, E, N);
	}

};

}	// namespace preprocess
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_RELATIVITY_H__
