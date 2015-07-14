#ifndef MCMC_PREPROCESS_DATA_FACTORY_H__
#define MCMC_PREPROCESS_DATA_FACTORY_H__

#include "mcmc/data.h"
#include "mcmc/options.h"
#include "mcmc/preprocess/dataset.h"
#include "mcmc/preprocess/netscience.h"
#include "mcmc/preprocess/relativity.h"

namespace mcmc {
namespace preprocess {

class DataFactory {
 public:
  DataFactory(const mcmc::Options &options);

  virtual ~DataFactory();

  const mcmc::Data *get_data() const;

  void setCompressed(bool on);

  void setContiguous(bool on);

  void setProgress(::size_t progress);

  void deleteData(const mcmc::Data *data);

 protected:
  std::string dataset_class_;
  std::string filename_;
  bool compressed_;
  bool contiguous_;
  ::size_t progress_;
};

};  // namespace preprocess
};  // namespace mcmc

#endif  // ndef MCMC_PREPROCESS_DATA_FACTORY_H__
