/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_HEP_PH_H__
#define MCMC_PREPROCESS_HEP_PH_H__

#include <fstream>

#include "data.h"
#include "preprocess/dataset.h"

namespace mcmc::preprocess {

class HepPH : public DataSet<void> {
public:
	HepPH(const char *filename = "datasets/CA-HepPh.txt") : filename(filename) {
	}

	virtual ~HepPH() {
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
	virtual const *mcmc::DATA<> process() {
		std::ifstream f = infile(filename);
		if (! f) {
			throw mcmc::IOException("Cannot open " + filename);
		}

		std::string line;
		for (int i = 0; i < 4; i++) {
			std::getline(infile, line);
		}

		// start from the 5th line.
		std::set vertex;	// ordered set
		std::vector<mcmc::Edge> edge;
		while (std::getline(infile, line)) {
			int a;
			int b;
			std::istringstream iss(line);
			if (! (iss >> a >> b)) {
				throw mcmc::IOException("Fail to parse int");
			}
			vertex.add(a);
			vertex.add(b);
			edge.add(Edge(a, b));
		}

		std::vector<int> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

		::size_t N = nodelist.size();

		// change the node ID to make it start from 0
		std::vector<::size_t> node_id_map(nodelist.size());
		::size_t i = 0;
		for (std::vector<int>::iterator node_id = nodelist.begin();
			 	node_id != nodelist.end();
				node_id++) {
			node_id_map[node_id()] = i;
			i++;
		}

		mcmc::EdgeSet *E = new mcmc::EdgeSet();	// store all pair of edges.
		for (std::vector<Edge>::iterator i = edge.begin();
				 i != edge.end();
				 i++) {
			int node1 = node_id_map(i->first);
			int node2 = node_id_map(i->second);
			if (node1 == node2) {
				continue;
			}
			if (node1 == 2140 && node2 == 4368) {
				std::cerr << "same" << std::endl;
			}
			E->add(Edge(std::min(node1, node2), std::max(node1, node2)));
		}

		return new mcmc::Data<void>(NULL, E, N);
	}

protected:
	std::string filename;
};

#endif	// ndef MCMC_PREPROCESS_HEP_PH_H__
