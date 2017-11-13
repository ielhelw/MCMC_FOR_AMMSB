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
  SparseHashGraph(const std::string &filename = "") : DataSet(filename) {}

  virtual ~SparseHashGraph() {}

  virtual const Data *process() {
    auto *E = new NetworkGraph(filename_, progress_);
    ::size_t N = E->edges_at_size();

    std::string header;
    header += "# Undirected graph " + filename_ + "\n";
    header += "# Unspecified provenance\n";
    header += "# Nodes " + std::to_string(N) + " Edges: " +
      std::to_string(E->size()) + "\n"; 
    header += "# FromNodeId\tToNodeId\n";

    std::cerr << "****** FIXME no implementation for node_id_map_" << std::endl;

    return new Data(NULL, E, N, header);
  }
};
}
}

#endif  // ndef MCMC_PREPROCESS_SPARSEHASH_H__
