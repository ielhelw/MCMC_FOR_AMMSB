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
#include <chrono>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {
namespace preprocess {

// FIXME: identical: hep_ph relativity ...

/**
 * Process relativity data set
 */
class Relativity : public DataSet {
public:
	Relativity(const std::string &filename, bool contiguous = false)
			: DataSet(filename == "" ? "datasets/CA-GrQc.txt" : filename, contiguous) {
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
using namespace std::chrono;
auto start = system_clock::now();

		std::ifstream infile(filename);
		if (! infile) {
			throw mcmc::IOException("Cannot open " + filename);
		}

std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms open file" << std::endl;
		std::string line;
		for (int i = 0; i < 4; i++) {
			std::getline(infile, line);
		}

		// start from the 5th line.

		mcmc::EdgeSet *E = new mcmc::EdgeSet();	// store all pair of edges.
		::size_t N;

		if (contiguous) {
			int max = -1;
			std::unordered_set<int> vertex;
			while (std::getline(infile, line)) {
				int a;
				int b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse int");
				}
				a--;
				b--;
				vertex.insert(a);
				vertex.insert(b);
				max = std::max(a, max);
				max = std::max(b, max);
				E->insert(Edge(std::min(a, b), std::max(a, b)));
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read unordered set" << std::endl;
			N = vertex.size();
			if (max + 1 != (int)N) {
				std::ostringstream s;
				s << "# vertices " << N << " max vertex " << max;
				throw OutOfRangeException(s.str());
			}

		} else {
			std::set<int> vertex;	// ordered set
			std::vector<mcmc::Edge> edge;
			while (std::getline(infile, line)) {
				int a;
				int b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse int");
				}
				vertex.insert(a);
				vertex.insert(b);
				edge.push_back(Edge(a, b));
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read ordered set" << std::endl;

			std::vector<int> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

			N = nodelist.size();

			// change the node ID to make it start from 0
			std::unordered_map<int, int> node_id_map;
			int i = 0;
			for (auto node_id: nodelist) {
				node_id_map[node_id] = i;
				i++;
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create map" << std::endl;

			for (auto i: edge) {
				int node1 = node_id_map[i.first];
				int node2 = node_id_map[i.second];
				if (node1 == node2) {
					continue;
				}
				E->insert(Edge(std::min(node1, node2), std::max(node1, node2)));
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create EdgeSet" << std::endl;
		}

		return new Data(NULL, E, N);
	}

};

}	// namespace preprocess
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_RELATIVITY_H__
