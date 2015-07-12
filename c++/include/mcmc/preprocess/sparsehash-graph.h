/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 *
 * @date 2015-07-11
 */

#ifndef MCMC_PREPROCESS_SPARSEHASH_H__
#define MCMC_PREPROCESS_SPARSEHASH_H__

#include "mcmc/data.h"
#include "mcmc/fileio.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {
namespace preprocess {

/**
 * Process relativity data set
 */
class SparseHashGraph : public DataSet {
public:
	SparseHashGraph(const std::string &filename = "")
			: DataSet(filename) {
	}

	virtual ~SparseHashGraph() {
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
		FileHandle f(filename_, compressed_, "r");

		// Read linked_edges
		int32_t N;
		f.read_fully(&N, sizeof N);
		std::vector<GoogleHashSet>* E = new std::vector<GoogleHashSet>(N);
		std::vector<GoogleHashSet>& data = *const_cast<std::vector<GoogleHashSet> *>(E);
		::size_t num_edges = 0;
		for (int32_t i = 0; i < N; ++i) {
			if (progress_ != 0 && i % progress_ == 0) {
				std::cerr << "Node + edgeset read " << i << std::endl;
				print_mem_usage(std::cerr);
			}
			data[i].read_metadata(f.handle());
			data[i].read_nopointer_data(f.handle());
			num_edges += data[i].size();
		}

		print_mem_usage(std::cerr);

		std::string header;
		header += "# Undirected graph " + filename_ + "\n";
		header += "# Unspecified provenance\n";
		header += "# Nodes " + to_string(N) + " Edges: " + to_string(num_edges) + "\n";
		header += "# FromNodeId\tToNodeId\n";

		return new Data(NULL, E, N, header);
	}

};

}
}

#endif	// ndef MCMC_PREPROCESS_SPARSEHASH_H__
