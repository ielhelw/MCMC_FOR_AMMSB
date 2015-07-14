/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_DATASET_H__
#define MCMC_PREPROCESS_DATASET_H__

#include <iostream>
#include <string>

#include "mcmc/data.h"

namespace mcmc {
namespace preprocess {

/**
 * Served as the abstract base class for different types of data sets.
 * For each data set, we should inherit from this class.
 */
class DataSet {
 public:
  DataSet(const std::string &filename);

  virtual ~DataSet();

  /**
   * Function to process the document. The document can be in any format. (i.e
   * txt, xml,..)
   * The subclass will implement this function to handle specific format of
   * document. Finally, return the Data object can be consumed by any learner.
   */
  /**
   * @return the caller must delete() the result
   */
  virtual const ::mcmc::Data *process() = 0;

  void setCompressed(bool on);

  void setContiguous(bool on);

  void setProgress(::size_t progress);

 protected:
  std::string filename_;
  bool compressed_;
  bool contiguous_;
  ::size_t progress_;
};

}  // namespace preprocess
}  // namespace mcmc

#endif  // ndef MCMC_PREPROCESS_DATASET_H__
