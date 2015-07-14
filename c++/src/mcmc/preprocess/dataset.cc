#include "mcmc/preprocess/dataset.h"

namespace mcmc {
namespace preprocess {

DataSet::DataSet(const std::string &filename) : filename_(filename) {
  std::cerr << "Handle input dataset from file " << filename_ << std::endl;
}

DataSet::~DataSet() {}

void DataSet::setCompressed(bool on) { compressed_ = on; }

void DataSet::setContiguous(bool on) { contiguous_ = on; }

void DataSet::setProgress(::size_t progress) { progress_ = progress; }

}  // namespace preprocess
}  // namespace mcmc
