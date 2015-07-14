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

#include <unordered_set>
#include <fstream>
#include <chrono>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {
namespace preprocess {

/**
 * Process relativity data set
 */
class Relativity : public DataSet {
 public:
  Relativity(const std::string &filename);

  virtual ~Relativity();

  /**
   * The data is stored in .txt file. The format of data is as follows, the
   *first column
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
  virtual const Data *process();

 private:
  Vertex contiguous_offset_ = 0;
};

}  // namespace preprocess
}  // namespace mcmc

#endif  // ndef MCMC_PREPROCESS_RELATIVITY_H__
